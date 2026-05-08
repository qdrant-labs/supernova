#!/usr/bin/env python3
"""
Dispatch distributed HuggingFace Hub upload jobs via SkyPilot pools.

Each worker downloads its assigned S3 parquet files (fast — S3 to EC2 is
in-region and effectively free) and streams them up to HuggingFace directly
from the datacenter, bypassing your local machine's upload bandwidth entirely.
"""

import json
import logging

import click
import yaml

from cli.skypilot_utils import (
    build_env_flags,
    make_run_dir,
    launch_pool_and_jobs,
    print_dry_run,
    print_monitor,
)
from vectorforge.destinations import S3Destination, discover_corpus_parquets

logger = logging.getLogger(__name__)

DEFAULT_RESOURCES = {
    "cpus": 2,
    "memory": 4,
    "cloud": "aws",
    "use_spot": True,
    "any_of": [
        {"region": "us-east-1"},
        {"region": "us-west-2"},
    ],
}


def list_s3_parquets(bucket: str, prefix: str) -> list[str]:
    """Return bare S3 keys (no scheme, no bucket) for every corpus parquet."""
    dest = S3Destination(bucket=bucket, prefix=prefix.rstrip("/"))
    scheme_prefix = f"s3://{bucket}/"
    return [u[len(scheme_prefix) :] for u in discover_corpus_parquets(dest)]


@click.command(
    name="push-hf-dist", help="Distribute HF upload across SkyPilot instances."
)
@click.argument("s3_uri")
@click.argument("repo_id")
@click.option(
    "--num-jobs",
    type=int,
    default=None,
    help="Number of parallel workers (default: auto from file count).",
)
@click.option(
    "--files-per-job",
    type=int,
    default=20,
    show_default=True,
    help="Files per worker when auto-computing num-jobs.",
)
@click.option(
    "--subfolder", default="data", show_default=True, help="Folder inside the HF repo."
)
@click.option("--private", is_flag=True, help="Create HF repo as private.")
@click.option(
    "--pool-name", default=None, help="SkyPilot pool name (default: auto-generated)."
)
@click.option("--on-demand", is_flag=True, help="Use on-demand instead of spot.")
@click.option(
    "--ramp",
    is_flag=True,
    help="Ramp workers gradually (min_workers=0). Default is burst.",
)
@click.option(
    "--dry-run", is_flag=True, help="Print plan and generate configs, don't launch."
)
def push_hf_dist(
    s3_uri,
    repo_id,
    num_jobs,
    files_per_job,
    subfolder,
    private,
    pool_name,
    on_demand,
    ramp,
    dry_run,
):
    """Dispatch distributed HF Hub uploads via SkyPilot."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger(__name__).setLevel(logging.INFO)

    if not s3_uri.startswith("s3://"):
        raise click.UsageError("s3_uri must start with s3://")
    without_scheme = s3_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    logger.info("Listing parquet files at s3://%s/%s/...", bucket, prefix)
    keys = list_s3_parquets(bucket, prefix)
    if not keys:
        logger.error("No parquet files found at %s", s3_uri)
        return

    num_jobs_eff = num_jobs or max(1, len(keys) // files_per_job)
    resources = dict(DEFAULT_RESOURCES)
    if on_demand:
        resources["use_spot"] = False

    repo_slug = repo_id.replace("/", "--")
    pool_name_eff = pool_name or f"vf-push-hf-{repo_slug}"

    run_dir = make_run_dir(pool_name_eff)
    run_id = run_dir.name

    click.echo("=" * 60)
    click.echo("vectorforge distributed HuggingFace push plan")
    click.echo("=" * 60)
    click.echo(f"  Source:       s3://{bucket}/{prefix}")
    click.echo(f"  Total files:  {len(keys)}")
    click.echo(f"  Num workers:  {num_jobs_eff}")
    click.echo(f"  Files/worker: ~{len(keys) // num_jobs_eff}")
    click.echo(f"  HF repo:      {repo_id}")
    click.echo(f"  Subfolder:    {subfolder}")
    click.echo(f"  Pool name:    {pool_name_eff}")
    click.echo(
        f"  Provision:    {'ramp (autoscaler)' if ramp else 'burst (all workers at startup)'}"
    )
    click.echo(f"  Resources:    {resources}")
    click.echo(f"  Run dir:      {run_dir}")
    click.echo("=" * 60)

    # Workers only need base deps (boto3 + huggingface_hub) — no extras required.
    # hf_transfer gives ~5x faster uploads; HF_HUB_ENABLE_HF_TRANSFER activates it.
    pool_yaml = {
        "pool": {
            "min_workers": 0 if ramp else num_jobs_eff,
            "max_workers": num_jobs_eff,
        },
        "resources": resources,
        "file_mounts": {"/app": "."},
        "setup": (
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "cd /app && uv sync && "
            "uv pip install hf_transfer"
        ),
    }
    pool_path = run_dir / "pool.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)

    push_flags = f"--num-jobs {num_jobs_eff} --subfolder {subfolder}"
    if private:
        push_flags += " --private"

    job_yaml = {
        "name": f"push-hf-{repo_slug}",
        "resources": resources,
        "envs": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        "run": f"cd /app && uv run vf push-hf {s3_uri} {repo_id} {push_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    manifest = {
        "run_id": run_id,
        "s3_uri": s3_uri,
        "repo_id": repo_id,
        "total_files": len(keys),
        "num_jobs": num_jobs_eff,
        "pool_name": pool_name_eff,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if dry_run:
        print_dry_run(pool_name_eff, num_jobs_eff, pool_path, job_path)
        return

    env_flags = build_env_flags(["HF_TOKEN"])
    logger.info("Creating pool '%s'...", pool_name_eff)
    logger.info("Submitting %d jobs to pool '%s'...", num_jobs_eff, pool_name_eff)
    launch_pool_and_jobs(pool_name_eff, pool_path, job_path, num_jobs_eff, env_flags)

    click.echo(f"\nSubmitted {num_jobs_eff} upload jobs to pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)
    click.echo(f"Dataset:      https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    push_hf_dist()
