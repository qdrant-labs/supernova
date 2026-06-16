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

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import click
import yaml

from supernova.cli.skypilot_utils import (
    build_env_dict,
    build_rust_worker_setup,
    config_mount,
    launch_pool_and_jobs,
    make_run_dir,
    print_dry_run,
    print_monitor,
    rust_worker_run,
)

from supernova.loader.vectorstore.qdrant import QdrantVectorStore
from supernova.destinations import datasource_to_destination

logger = logging.getLogger(__name__)


def resolve_env_vars(value: str) -> str:
    """
    Replace ${VAR_NAME} references with environment variable values.
    """

    def _replace(match):
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")
        return val

    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", _replace, value)
    return value


def resolve_config(obj):
    """
    Recursively resolve env vars in config values.
    """
    if isinstance(obj, str):
        return resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_config(v) for v in obj]
    return obj


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


async def _setup_collection(store: QdrantVectorStore, dimensions: dict[str, int]):
    await store.ensure_collection(dimensions)
    await store.defer_indexing()
    await store.close()


async def _enable_and_wait(store: QdrantVectorStore):
    await store.enable_indexing()
    await store.wait_for_indexing()
    await store.close()


@click.command(name="load-dist", help="Distribute loading across SkyPilot instances.")
@click.argument("config")
@click.option(
    "--dry-run", is_flag=True, help="Generate configs and print plan, don't launch."
)
@click.option("--num-shards", type=int, default=None, help="Override number of shards.")
@click.option(
    "--pool-name", default=None, help="SkyPilot pool name (default: auto-generated)."
)
@click.option(
    "--on-demand",
    is_flag=True,
    help="Use on-demand instances instead of spot (higher cost, no preemption, "
    "separate AWS quota).",
)
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
def load_dist(config, dry_run, num_shards, pool_name, on_demand, ramp, finalize):
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

    # just enable indexing and exit
    if finalize:
        logger.info("Enabling Qdrant indexing...")
        vs_cfg = dict(resolved_config["vectorstore"])
        vs_cfg.pop("type", None)
        vs_cfg.pop("params", None)
        store = QdrantVectorStore(vectors=vectors_spec, **vs_cfg)

        t0 = time.perf_counter()
        asyncio.run(_enable_and_wait(store))
        elapsed = time.perf_counter() - t0
        logger.info(f"Indexing complete in {elapsed:.1f}s")
        return

    dispatch_cfg = cfg["dispatch"]
    resources = dict(cfg["resources"])
    if on_demand:
        resources["use_spot"] = False
    num_shards_eff = num_shards or dispatch_cfg["num_shards"]
    config_name = Path(config).stem
    run_name = dispatch_cfg.get("run_name", config_name)
    pool_name_eff = pool_name or f"nova-load-{run_name}"

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
    click.echo(f"  Num shards:   {num_shards_eff}")
    click.echo(f"  Files/shard:  ~{len(files) // num_shards_eff}")
    click.echo(f"  Pool name:    {pool_name_eff}")
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
            "min_workers": 0 if ramp else num_shards_eff,
            "max_workers": num_shards_eff,
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
            f"{remote_cfg} --num-jobs {num_shards_eff} --no-manage-indexing",
        ),
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    # write manifest
    manifest = {
        "run_id": run_id,
        "config": config,
        "num_shards": num_shards_eff,
        "total_files": len(files),
        "pool_name": pool_name_eff,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated configs in {run_dir}/")

    if dry_run:
        print_dry_run(pool_name_eff, num_shards_eff, pool_path, job_path)
        return

    # Setup Qdrant: create collection + defer indexing. Keep `params` — this is
    # the ONLY create_collection call in the distributed path (workers run
    # --no-manage-indexing), so dropping it here silently creates the collection
    # with default shard_number / replication_factor instead of the configured ones.
    logger.info("Setting up Qdrant collection...")
    vs_cfg = dict(resolved_config["vectorstore"])
    vs_cfg.pop("type", None)
    store = QdrantVectorStore(vectors=vectors_spec, **vs_cfg)

    # Use the first corpus file to probe vector dimensions. Assumes all files
    # share the same schema (guaranteed by the embedding pipeline) and avoids
    # the glob hitting eval/ parquets that have a different schema entirely.
    from supernova.destinations import S3Destination, HfDestination

    if isinstance(dest, S3Destination):
        from supernova.loader.datasource.s3 import S3DataReader

        reader = S3DataReader(
            bucket=dest.bucket,
            prefix=dest.prefix,
            vectors=vectors_spec,
            file_list=[files[0]],
        )
    elif isinstance(dest, HfDestination):
        from supernova.loader.datasource.huggingface import HuggingFaceDataReader

        reader = HuggingFaceDataReader(
            repo_id=dest.repo_id,
            subdir=dest.subdir or None,
            vectors=vectors_spec,
            file_list=[files[0]],
        )
    else:
        raise ValueError(
            f"Unsupported destination for dim probe: {type(dest).__name__}"
        )
    dimensions = reader.get_dimensions()
    reader.close()

    asyncio.run(_setup_collection(store, dimensions))
    logger.info("Qdrant collection ready (indexing deferred)")

    envs = build_env_dict(["QDRANT_URL", "QDRANT_API_KEY", "HF_TOKEN"])
    logger.info(f"Creating pool '{pool_name_eff}'...")
    logger.info(f"Submitting {num_shards_eff} jobs to pool '{pool_name_eff}'...")
    launch_pool_and_jobs(pool_name_eff, pool_path, job_path, num_shards_eff, envs)

    click.echo(f"\nSubmitted {num_shards_eff} loading jobs to pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)
    click.echo("\nAfter all jobs complete, enable Qdrant indexing:")
    click.echo(f"  nova load-dist {config} --finalize")


if __name__ == "__main__":
    load_dist()
