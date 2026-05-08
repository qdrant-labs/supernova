#!/usr/bin/env python3
"""
Dispatch distributed `vf partition` jobs via SkyPilot pools.

Same shape as `vf embed-dist` but with CPU-only resources (no GPU) and the
no-op embedder, so each worker just splits/writes raw rows. Lets you validate
that ranks read non-overlapping slices and the S3 layout is what you expect
before committing to a real (expensive) embed run.
"""

import json
import logging
import math
from pathlib import Path

import click
import yaml

from cli.skypilot_utils import build_env_flags, make_run_dir, launch_pool_and_jobs, print_dry_run, print_monitor

logger = logging.getLogger(__name__)

# CPU-only -- no GPU is needed when the model isn't actually running.
DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory": 16,
    "cloud": "aws",
    "use_spot": True,
    "any_of": [
        {"region": "us-east-1"},
        {"region": "us-west-2"},
    ],
}


@click.command(name="partition-dist", help="Distributed partition via SkyPilot pool.")
@click.argument("config")
@click.option("--dry-run", is_flag=True, help="Generate configs and print plan, don't launch.")
@click.option("--num-jobs", type=int, default=None,
              help="Number of parallel jobs (default: auto from dataset size).")
@click.option("--chunk-size", type=int, default=None,
              help="Rows per job (used to auto-compute num-jobs).")
@click.option("--pool-name", default=None,
              help="SkyPilot pool name (default: auto-generated).")
@click.option("--max-workers", type=int, default=None,
              help="Max pool workers for autoscaling (default: num-jobs).")
@click.option("--on-demand", is_flag=True,
              help="Use on-demand instances instead of spot.")
@click.option("--ramp", is_flag=True,
              help="Use SkyPilot's gradual autoscaler (min_workers=0). Default is burst.")
def partition_dist(config, dry_run, num_jobs, chunk_size, pool_name, max_workers, on_demand, ramp):
    """Dispatch distributed `vf partition` via SkyPilot pools."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    source_cfg = cfg["source"]
    pipeline_cfg = cfg.get("pipeline", {})
    # `partition.resources:` may override CPU defaults; fall back to embed-style
    # `resources:` if set; finally fall back to DEFAULT_RESOURCES.
    partition_cfg = cfg.get("partition") or {}
    resources = partition_cfg.get("resources") or cfg.get("resources") or dict(DEFAULT_RESOURCES)
    if on_demand:
        resources = dict(resources)
        resources["use_spot"] = False

    # get dataset size (source-agnostic)
    from cli.run_embedder import build_source
    source = build_source(dict(source_cfg))
    total_rows = source.get_total_rows()

    chunk_size_eff = chunk_size or pipeline_cfg.get("chunk_size", 100_000)
    num_jobs_eff = num_jobs or math.ceil(total_rows / chunk_size_eff)
    max_workers_eff = max_workers or num_jobs_eff

    config_name = Path(config).stem
    pool_name_eff = pool_name or f"vf-partition-{config_name}"

    # create run directory
    run_dir = make_run_dir(pool_name_eff)
    run_id = run_dir.name

    click.echo("=" * 60)
    click.echo("vectorforge distributed partition plan (no-op embedder)")
    click.echo("=" * 60)
    click.echo(f"  Source:       {source.source_name}")
    click.echo(f"  Total rows:   {total_rows:,}")
    click.echo(f"  Num jobs:     {num_jobs_eff}")
    click.echo(f"  Rows/job:     ~{math.ceil(total_rows / num_jobs_eff):,}")
    click.echo(f"  Max workers:  {max_workers_eff}")
    click.echo(f"  Pool name:    {pool_name_eff}")
    click.echo(f"  Provision:    {'ramp (autoscaler)' if ramp else 'burst (all workers at startup)'}")
    click.echo(f"  Resources:    {resources}")
    click.echo(f"  Run dir:      {run_dir}")
    click.echo("=" * 60)

    # pool YAML -- CPU only. Setup mirrors embed-dist (uv install + sync).
    pool_yaml = {
        "pool": {
            "min_workers": 0 if ramp else max_workers_eff,
            "max_workers": max_workers_eff,
        },
        "resources": resources,
        "file_mounts": {
            "/app": ".",
        },
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync --extra partition",
    }
    pool_path = run_dir / "pool.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)

    job_yaml = {
        "name": f"partition-{config_name}",
        "resources": resources,
        "run": f"cd /app && uv run vf partition {config} --num-jobs {num_jobs_eff}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    manifest = {
        "run_id": run_id,
        "config": config,
        "source": source.source_name,
        "total_rows": total_rows,
        "num_jobs": num_jobs_eff,
        "max_workers": max_workers_eff,
        "pool_name": pool_name_eff,
        "mode": "partition",
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated configs in {run_dir}/")

    if dry_run:
        print_dry_run(pool_name_eff, num_jobs_eff, pool_path, job_path)
        return

    env_flags = build_env_flags(["HF_TOKEN"])
    logger.info(f"Creating pool '{pool_name_eff}'...")
    logger.info(f"Submitting {num_jobs_eff} jobs to pool '{pool_name_eff}'...")
    launch_pool_and_jobs(pool_name_eff, pool_path, job_path, num_jobs_eff, env_flags)

    click.echo(f"\nSubmitted {num_jobs_eff} partition jobs to pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)
    click.echo("")
    click.echo("Once all ranks finish, verify clean partitioning with:")
    click.echo("  uv run python scripts/verify_no_duplicates.py s3://<your-prefix>/ --content-columns text")


if __name__ == "__main__":
    partition_dist()
