#!/usr/bin/env python3
"""
Dispatch distributed loading jobs via SkyPilot pools.

Discovers parquet files on S3, creates a pool of CPU workers, and submits
parallel loading jobs. Each job discovers files and picks its shard via
--num-jobs + $SKYPILOT_JOB_RANK.

Also manages the Qdrant indexing lifecycle: defers indexing before loading,
enables it after with --finalize.

Reads the same configs as `nova load` (configs/loader/*.yaml). The single-machine
loader ignores `dispatch:` / `resources:` blocks; this CLI consumes them.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import click
import yaml

from supernova.cli.config_resolve import resolve_config
from supernova.cli.skypilot_utils import (
    build_env_dict,
    build_rust_worker_setup,
    config_mount,
    launch_pool_and_jobs,
    make_run_dir,
    print_dry_run,
    print_monitor,
    referenced_env_vars,
    resolve_binary,
    resolve_resources,
    rust_worker_run,
)
from supernova.destinations import datasource_to_destination
from supernova.metrics import make_run_id

logger = logging.getLogger(__name__)

# CPU loader workers. Used when the config omits a `resources:` block; spot is
# fine because each shard's upserts are idempotent (point id is content-derived)
# and SkyPilot re-queues a preempted job. `--on-demand` flips use_spot off.
DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory": 16,
    "cloud": "aws",
    "use_spot": True,
}


def discover_parquet_files(ds_cfg: dict) -> list[str]:
    """
    List all corpus .parquet files for the loader's datasource block,
    excluding eval/ artifacts. Returns absolute URIs (s3:// or hf://...).
    """
    from supernova.destinations import (
        datasource_to_destination,
        discover_corpus_parquets,
    )

    dest = datasource_to_destination(ds_cfg)
    return discover_corpus_parquets(dest)


@click.command(name="load-dist", help="Distribute loading across SkyPilot instances.")
@click.argument("config")
@click.option(
    "--dry-run", is_flag=True, help="Generate configs and print plan, don't launch."
)
@click.option(
    "--num-workers",
    "--num-shards",
    "num_workers",
    type=int,
    default=None,
    help="Number of workers; the corpus is split into one shard each. "
    "(--num-shards: deprecated alias.)",
)
@click.option(
    "--pool-name", default=None, help="SkyPilot pool name (default: auto-generated)."
)
@click.option(
    "--on-demand",
    is_flag=True,
    help="Use on-demand instances instead of spot (higher cost, no preemption, "
    "separate AWS quota).",
)
@click.option("--cloud", default=None, help="Override resources.cloud (e.g. aws, gcp).")
@click.option("--cpus", type=int, default=None, help="Override resources.cpus per worker.")
@click.option(
    "--ramp",
    is_flag=True,
    help="Let SkyPilot's autoscaler bring workers up gradually (min_workers=0). "
    "Default is burst.",
)
@click.option(
    "--finalize",
    is_flag=True,
    help="Enable Qdrant indexing (run after all jobs complete).",
)
def load_dist(config, dry_run, num_workers, pool_name, on_demand, cloud, cpus, ramp, finalize):
    """Dispatch distributed loading via SkyPilot pools."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("supernova").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    resolved_config = resolve_config(cfg)

    vectors_spec = resolved_config.get("vectors")
    if not vectors_spec:
        raise click.UsageError("config is missing required top-level 'vectors:' block")

    # just enable indexing and exit — delegated to the loader binary's control
    # plane (it owns the Qdrant lifecycle now; the controller only orchestrates).
    if finalize:
        logger.info("Enabling Qdrant indexing (nova-load --finalize)...")
        t0 = time.perf_counter()
        subprocess.run([resolve_binary("nova-load"), "--finalize", config], check=True)
        logger.info(f"Indexing complete in {time.perf_counter() - t0:.1f}s")
        return

    # dispatch / resources are optional: the worker count can come from
    # --num-workers and resources from ~/.nova/resources.yaml or the CPU default,
    # so a plain `nova load` config (which has neither) also works with load-dist.
    dispatch_cfg = cfg.get("dispatch") or {}
    # Layered: built-in default < ~/.nova resources file < config `resources:` <
    # flags. --on-demand only forces spot off (load defaults to spot — shards are
    # idempotent and SkyPilot re-queues a preemption).
    overrides = {"cloud": cloud, "cpus": cpus}
    if on_demand:
        overrides["use_spot"] = False
    resources = resolve_resources("load", cfg.get("resources"), overrides, DEFAULT_RESOURCES)
    # Accept dispatch.num_workers, or the legacy dispatch.num_shards key.
    num_workers_eff = (
        num_workers or dispatch_cfg.get("num_workers") or dispatch_cfg.get("num_shards")
    )
    if not num_workers_eff:
        raise click.UsageError(
            "worker count is required: pass --num-workers N or set dispatch.num_workers in the config"
        )
    config_name = Path(config).stem
    run_name = dispatch_cfg.get("run_name", config_name)
    pool_name_eff = pool_name or f"nova-load-{run_name}"
    # Mint ONE run id on the controller and forward it (NOVA_RUN_ID) to every
    # shard, so the partitioned fleet writes into a single run (each worker is a
    # node_id = SKYPILOT_JOB_RANK) instead of each shard minting its own. Mirrors
    # storm-dist; nova-load's resolve_run_id picks this up.
    metrics_run_id = make_run_id(run_name)

    # create run directory
    run_dir = make_run_dir(run_name)
    run_id = run_dir.name

    # discover parquet files
    ds_cfg = resolved_config["datasource"]

    dest = datasource_to_destination(ds_cfg)
    logger.info(f"Discovering parquet files at {dest.root_uri}/...")
    files = discover_parquet_files(ds_cfg)
    logger.info(f"Found {len(files)} parquet files")

    if not files:
        logger.error("No parquet files found. Exiting.")
        return

    click.echo("=" * 60)
    click.echo("supernova distributed loading plan")
    click.echo("=" * 60)
    click.echo(f"  Destination:  {dest.root_uri}")
    click.echo(f"  Total files:  {len(files)}")
    click.echo(f"  Num workers:  {num_workers_eff}")
    click.echo(f"  Files/worker: ~{len(files) // num_workers_eff}")
    click.echo(f"  Pool name:    {pool_name_eff}")
    click.echo(f"  Metrics run:  {metrics_run_id}")
    click.echo(
        f"  Provision:    {'ramp (autoscaler)' if ramp else 'burst (all workers at startup)'}"
    )
    click.echo(f"  Resources:    {resources}")
    click.echo(f"  Run dir:      {run_dir}")
    click.echo("=" * 60)

    # Stage the (raw) config so workers get it without mounting the repo;
    # secrets ride along as forwarded env vars and `nova` resolves ${VAR} at run time.
    cfg_mounts, remote_cfg = config_mount(run_dir, config)

    # generate pool YAML — burst by default (provision all workers at startup);
    # SkyPilot's autoscaler ramp is too slow when we know the target count up front.
    pool_yaml = {
        "pool": {
            "min_workers": 0 if ramp else num_workers_eff,
            "max_workers": num_workers_eff,
        },
        "resources": resources,
        "file_mounts": cfg_mounts,
        "setup": build_rust_worker_setup("nova-load"),
    }
    pool_path = run_dir / "pool.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)

    # generate job YAML
    job_yaml = {
        "name": f"load-{run_name}",
        "resources": resources,
        "run": rust_worker_run(
            "nova-load",
            f"{remote_cfg} --num-jobs {num_workers_eff} --no-manage-indexing",
        ),
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    # write manifest
    manifest = {
        "run_id": run_id,
        "config": config,
        "num_workers": num_workers_eff,
        "total_files": len(files),
        "pool_name": pool_name_eff,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated configs in {run_dir}/")

    if dry_run:
        print_dry_run(pool_name_eff, num_workers_eff, pool_path, job_path)
        return

    # Create the collection + defer indexing ONCE, here on the controller, by
    # running the loader binary in control-plane mode. Workers then load with
    # --no-manage-indexing. nova-load probes vector dims and applies the
    # configured collection params itself, so this is the single create call.
    logger.info("Setting up Qdrant collection (nova-load --setup-only)...")
    subprocess.run([resolve_binary("nova-load"), "--setup-only", config], check=True)
    logger.info("Qdrant collection ready (indexing deferred)")

    # Forward exactly the secrets the config references via ${VAR} (vendor-agnostic),
    # plus HF_TOKEN (an HF datasource reads it from the env, not from the YAML).
    envs = build_env_dict(referenced_env_vars(config) + ["HF_TOKEN"])
    envs["NOVA_RUN_ID"] = metrics_run_id  # every shard writes into this one run
    # When launched under `nova experiment`, tag this run so Grafana can group the
    # write phase with the concurrent reads on one timeline.
    exp_id = os.environ.get("NOVA_EXPERIMENT_ID")
    if exp_id:
        envs["NOVA_EXPERIMENT_ID"] = exp_id
    logger.info(f"Creating pool '{pool_name_eff}'...")
    logger.info(f"Submitting {num_workers_eff} jobs to pool '{pool_name_eff}'...")
    launch_pool_and_jobs(pool_name_eff, pool_path, job_path, num_workers_eff, envs)

    click.echo(f"\nSubmitted {num_workers_eff} loading jobs to pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)
    click.echo("\nAfter all jobs complete, enable Qdrant indexing:")
    click.echo(f"  nova load-dist {config} --finalize")


if __name__ == "__main__":
    load_dist()
