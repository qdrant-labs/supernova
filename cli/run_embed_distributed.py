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
    build_env_dict,
    launch_pool_and_jobs,
    launch_single_job_to_pool,
    make_run_dir,
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


def _find_latest_run_dir(config_path: str) -> Path | None:
    """
    Return the most recent ``runs/<timestamp>_<name>/`` whose manifest.json was
    written for this config. Matches against the manifest's ``config`` field
    rather than the dir name so custom ``--pool-name`` runs are found too.
    """
    runs_root = Path("runs")
    if not runs_root.exists():
        return None
    candidates: list[Path] = []
    for run_dir in runs_root.iterdir():
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("config") == config_path:
            candidates.append(run_dir)
    if not candidates:
        return None
    # Run dirs are timestamped (YYYY-MM-DDTHH-MM_*) so lexical sort = chronological.
    return sorted(candidates)[-1]


def _retry_one_rank(
    config: str,
    retry_rank: int,
    run_dir_override: str | None,
    dry_run: bool,
) -> None:
    """
    Re-launch a single job rank against the existing pool. Reads the original
    run's manifest + job.yaml so the resources, env, and num_jobs match exactly.
    """
    if run_dir_override:
        run_dir = Path(run_dir_override)
        if not run_dir.exists():
            raise click.UsageError(f"--run-dir {run_dir} does not exist.")
    else:
        found = _find_latest_run_dir(config)
        if found is None:
            raise click.UsageError(
                f"No run dir found whose manifest points at {config}. "
                "Pass --run-dir explicitly."
            )
        run_dir = found

    manifest_path = run_dir / "manifest.json"
    job_yaml_path = run_dir / "job.yaml"
    if not manifest_path.exists():
        raise click.UsageError(f"manifest.json missing at {manifest_path}")
    if not job_yaml_path.exists():
        raise click.UsageError(f"job.yaml missing at {job_yaml_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(job_yaml_path) as f:
        original_job = yaml.safe_load(f)

    num_jobs_orig = int(manifest["num_jobs"])
    pool_name = manifest["pool_name"]

    if retry_rank < 0 or retry_rank >= num_jobs_orig:
        raise click.UsageError(
            f"--retry-rank must be in [0, {num_jobs_orig - 1}] (run had num_jobs={num_jobs_orig})"
        )

    rank_width = max(2, len(str(num_jobs_orig - 1)))
    rank_str = f"{retry_rank:0{rank_width}d}"

    # Reuse the original job.yaml's resources + envs verbatim so the retry runs
    # on the same hardware with the same secrets. We only override the run line
    # to pin --job-rank.
    retry_job = {
        "name": f"{original_job.get('name', 'embed')}-retry-rank{rank_str}",
        "resources": original_job["resources"],
        "envs": original_job.get("envs", {"HF_HUB_ENABLE_HF_TRANSFER": "1"}),
        "run": (
            f"cd /app && uv run vf embed {config} "
            f"--num-jobs {num_jobs_orig} --job-rank {retry_rank}"
        ),
    }
    retry_path = run_dir / f"retry_rank{rank_str}.yaml"
    with open(retry_path, "w") as f:
        yaml.dump(retry_job, f, default_flow_style=False, sort_keys=False)

    click.echo("=" * 60)
    click.echo("vectorforge embed-dist retry plan")
    click.echo("=" * 60)
    click.echo(f"  Config:      {config}")
    click.echo(f"  Run dir:     {run_dir}")
    click.echo(f"  Pool:        {pool_name}")
    click.echo(f"  Num jobs:    {num_jobs_orig}")
    click.echo(f"  Retry rank:  {retry_rank}  (of 0..{num_jobs_orig - 1})")
    click.echo(f"  Retry yaml:  {retry_path}")
    click.echo("=" * 60)

    if dry_run:
        click.echo(f"\n[dry run] Would submit {retry_path} to pool '{pool_name}'.")
        click.echo(f"To run manually: sky jobs launch -p {pool_name} -y {retry_path}")
        return

    envs = build_env_dict(["HF_TOKEN", "OPENAI_API_KEY"])
    logger.info("Submitting retry for rank %d to pool '%s'...", retry_rank, pool_name)
    launch_single_job_to_pool(pool_name, retry_path, envs)
    click.echo(f"\nSubmitted retry for rank {retry_rank} to pool '{pool_name}'")
    click.echo("Monitor: sky jobs queue   |   Logs: sky jobs logs <job-id>")


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
@click.option(
    "--retry-rank",
    type=int,
    default=None,
    help="Re-launch one specific job rank to the existing pool. Reads num_jobs and "
    "pool_name from the most recent run for this config (or --run-dir). Use after "
    "a transient failure (HF flake, spot preemption) to redo just that slice without "
    "restarting the whole pool.",
)
@click.option(
    "--run-dir",
    "run_dir_override",
    default=None,
    help="With --retry-rank: explicit path to the runs/<dir>/ to retry against. "
    "Defaults to the most recent run dir whose manifest.json points at this config.",
)
def embed_dist(
    config,
    dry_run,
    num_jobs,
    chunk_size,
    pool_name,
    max_workers,
    on_demand,
    ramp,
    retry_rank,
    run_dir_override,
):
    """Dispatch distributed embedding via SkyPilot pools."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    if retry_rank is not None:
        _retry_one_rank(config, retry_rank, run_dir_override, dry_run)
        return

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
    # HF_HUB_ENABLE_HF_TRANSFER activates the Rust hf_transfer client for
    # multi-part parallel downloads — required if the source is a large
    # huggingface_parquet dataset and prefetch=true. The hf_transfer wheel is
    # pulled in via the `embed` extra.
    job_yaml = {
        "name": f"embed-{config_name}",
        "resources": resources,
        "envs": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
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

    envs = build_env_dict(["HF_TOKEN", "OPENAI_API_KEY"])
    logger.info(f"Creating pool '{pool_name_eff}'...")
    logger.info(f"Submitting {num_jobs_eff} jobs to pool '{pool_name_eff}'...")
    launch_pool_and_jobs(pool_name_eff, pool_path, job_path, num_jobs_eff, envs)

    click.echo(f"\nSubmitted {num_jobs_eff} jobs to pool '{pool_name_eff}'")
    print_monitor(pool_name_eff)


if __name__ == "__main__":
    embed_dist()
