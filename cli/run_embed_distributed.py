#!/usr/bin/env python3
"""
Dispatch distributed embedding jobs via SkyPilot pools.

Creates a pool of GPU workers and submits parallel embedding jobs. Each job
processes a slice of the dataset using --num-jobs/--job-rank for automatic
partitioning.
"""

import json
import logging
import math
from pathlib import Path

import click
import yaml

from cli.skypilot_utils import (
    build_env_flags,
    make_run_dir,
    launch_pool_and_jobs,
    print_dry_run,
    print_monitor,
)

logger = logging.getLogger(__name__)

DEFAULT_RESOURCES = {
    "accelerators": "A10G:1",
    "instance_type": "g5.4xlarge",  # 16 vCPU, 64GB RAM, 1x A10G — wider RAM headroom for long-text datasets
    "cloud": "aws",
    "use_spot": True,
    "disk_size": 150,
    "image_id": {
        "us-east-1": "ami-0038d79e7270bb987",
        "us-west-2": "ami-08a03808395c1b31f",
        "us-east-2": "ami-0a28b3d7e7c9192a7",
    },
    "any_of": [
        {"region": "us-east-1"},
        {"region": "us-west-2"},
        # {"region": "us-east-2"} # --- has a lower quote right now for some reason, so skipping.
    ],
}


@click.command(name="embed-dist", help="Embed distributed via SkyPilot pool.")
@click.argument("config")
@click.option(
    "--dry-run", is_flag=True, help="Generate configs and print plan, don't launch."
)
@click.option(
    "--num-jobs",
    type=int,
    default=None,
    help="Number of parallel jobs (default: auto from dataset size).",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Rows per job (used to auto-compute num-jobs).",
)
@click.option(
    "--pool-name", default=None, help="SkyPilot pool name (default: auto-generated)."
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Max pool workers for autoscaling (default: num-jobs).",
)
@click.option(
    "--on-demand",
    is_flag=True,
    help="Use on-demand instances instead of spot (higher cost, no preemption, separate AWS quota).",
)
@click.option(
    "--ramp",
    is_flag=True,
    help="Let SkyPilot's autoscaler bring workers up gradually (min_workers=0). "
    "Default is burst (min_workers=max_workers) since EC2 provisioning is "
    "slow and we know the target count up front.",
)
def embed_dist(
    config, dry_run, num_jobs, chunk_size, pool_name, max_workers, on_demand, ramp
):
    """Dispatch distributed embedding via SkyPilot pools."""
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
    resources = cfg.get("resources", dict(DEFAULT_RESOURCES))
    if on_demand:
        resources["use_spot"] = False

    # get dataset size (source-agnostic)
    from cli.run_embedder import build_source

    source = build_source(dict(source_cfg))
    total_rows = source.get_total_rows()

    chunk_size_eff = chunk_size or pipeline_cfg.get("chunk_size", 100_000)
    num_jobs_eff = num_jobs or math.ceil(total_rows / chunk_size_eff)
    max_workers_eff = max_workers or num_jobs_eff

    config_name = Path(config).stem
    pool_name_eff = pool_name or f"vf-embed-{config_name}"

    # create run directory
    run_dir = make_run_dir(pool_name_eff)
    run_id = run_dir.name

    click.echo("=" * 60)
    click.echo("vectorforge distributed embedding plan")
    click.echo("=" * 60)
    click.echo(f"  Source:       {source.source_name}")
    click.echo(f"  Total rows:   {total_rows:,}")
    click.echo(f"  Num jobs:     {num_jobs_eff}")
    click.echo(f"  Rows/job:     ~{math.ceil(total_rows / num_jobs_eff):,}")
    click.echo(f"  Max workers:  {max_workers_eff}")
    click.echo(f"  Pool name:    {pool_name_eff}")
    click.echo(
        f"  Provision:    {'ramp (autoscaler)' if ramp else 'burst (all workers at startup)'}"
    )
    click.echo(f"  Resources:    {resources}")
    click.echo(f"  Run dir:      {run_dir}")
    click.echo("=" * 60)

    # generate pool YAML — burst by default (provision all workers at startup);
    # SkyPilot's autoscaler ramp is too slow when we know the target count up front.
    pool_yaml = {
        "pool": {
            "min_workers": 0 if ramp else max_workers_eff,
            "max_workers": max_workers_eff,
        },
        "resources": resources,
        "file_mounts": {
            "/app": ".",
        },
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync --extra embed",
    }
    pool_path = run_dir / "pool.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)

    # generate job YAML — pool-submitted jobs MUST specify resources (per the
    # SkyPilot pools docs, else the job won't be able to use a GPU) but MUST
    # NOT specify setup / file_mounts / workdir (those come from the pool).
    job_yaml = {
        "name": f"embed-{config_name}",
        "resources": resources,
        "run": f"cd /app && uv run vf embed {config} --num-jobs {num_jobs_eff}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    # write manifest
    manifest = {
        "run_id": run_id,
        "config": config,
        "source": source.source_name,
        "total_rows": total_rows,
        "num_jobs": num_jobs_eff,
        "max_workers": max_workers_eff,
        "pool_name": pool_name_eff,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated configs in {run_dir}/")

    if dry_run:
        print_dry_run(pool_name_eff, num_jobs_eff, pool_path, job_path)
        return

    env_flags = build_env_flags(["HF_TOKEN", "OPENAI_API_KEY"])
    logger.info(f"Creating pool '{pool_name_eff}'...")
    logger.info(f"Submitting {num_jobs_eff} jobs to pool '{pool_name_eff}'...")
    launch_pool_and_jobs(pool_name_eff, pool_path, job_path, num_jobs_eff, env_flags)

    click.echo(f"\nSubmitted {num_jobs_eff} jobs to pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)


if __name__ == "__main__":
    embed_dist()
