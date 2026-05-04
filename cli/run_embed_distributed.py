#!/usr/bin/env python3
"""
Dispatch distributed embedding jobs via SkyPilot pools.

Creates a pool of GPU workers and submits parallel embedding jobs. Each job
processes a slice of the dataset using --num-jobs/--job-rank for automatic
partitioning.

Then wrapper that joins sky pilot pool management with the existing embedding pipeline. The embedding
pipeline is designed to be run in parallel across multiple workers.

Usage:
  vectorforge-embed-distributed configs/embedder/finewiki_gte_multi/en.yaml
  vectorforge-embed-distributed configs/embedder/finewiki_gte_multi/en.yaml --dry-run
  vectorforge-embed-distributed configs/embedder/finewiki_gte_multi/en.yaml --num-jobs 20
  vectorforge-embed-distributed configs/embedder/finewiki_gte_multi/en.yaml --pool-name my-pool
"""

import argparse
import json
import logging
import math
import os
import subprocess

from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ENV_VARS_TO_FORWARD = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "HF_TOKEN",
    "OPENAI_API_KEY",
]

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


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Dispatch distributed embedding via SkyPilot pools")
    parser.add_argument("config", help="Path to embedder YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and print plan, don't launch")
    parser.add_argument("--num-jobs", type=int, help="Number of parallel jobs (default: auto from dataset size)")
    parser.add_argument("--chunk-size", type=int, help="Rows per job (used to auto-compute num-jobs)")
    parser.add_argument("--pool-name", type=str, help="SkyPilot pool name (default: auto-generated)")
    parser.add_argument("--max-workers", type=int, help="Max pool workers for autoscaling (default: num-jobs)")
    parser.add_argument("--on-demand", action="store_true", help="Use on-demand instances instead of spot (higher cost, no preemption, separate AWS quota)")
    parser.add_argument("--ramp", action="store_true", help="Let SkyPilot's autoscaler bring workers up gradually (min_workers=0). Default is burst (min_workers=max_workers) since EC2 provisioning is slow and we know the target count up front.")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    source_cfg = config["source"]
    pipeline_cfg = config.get("pipeline", {})
    resources = config.get("resources", dict(DEFAULT_RESOURCES))
    if args.on_demand:
        resources["use_spot"] = False

    # get dataset size (source-agnostic)
    from cli.run_embedder import build_source
    source = build_source(dict(source_cfg))
    total_rows = source.get_total_rows()

    chunk_size = args.chunk_size or pipeline_cfg.get("chunk_size", 100_000)
    num_jobs = args.num_jobs or math.ceil(total_rows / chunk_size)
    max_workers = args.max_workers or num_jobs

    config_name = Path(args.config).stem
    pool_name = args.pool_name or f"vf-embed-{config_name}"

    # create run directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_id = f"{timestamp}_{pool_name}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("vectorforge distributed embedding plan")
    print("=" * 60)
    print(f"  Source:       {source.source_name}")
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Num jobs:     {num_jobs}")
    print(f"  Rows/job:     ~{math.ceil(total_rows / num_jobs):,}")
    print(f"  Max workers:  {max_workers}")
    print(f"  Pool name:    {pool_name}")
    print(f"  Provision:    {'ramp (autoscaler)' if args.ramp else 'burst (all workers at startup)'}")
    print(f"  Resources:    {resources}")
    print(f"  Run dir:      {run_dir}")
    print("=" * 60)

    # generate pool YAML — burst by default (provision all workers at startup);
    # SkyPilot's autoscaler ramp is too slow when we know the target count up front.
    pool_yaml = {
        "pool": {
            "min_workers": 0 if args.ramp else max_workers,
            "max_workers": max_workers,
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
        "run": f"cd /app && uv run vf embed {args.config} --num-jobs {num_jobs}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    # write manifest
    manifest = {
        "run_id": run_id,
        "config": args.config,
        "source": source.source_name,
        "total_rows": total_rows,
        "num_jobs": num_jobs,
        "max_workers": max_workers,
        "pool_name": pool_name,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated configs in {run_dir}/")

    if args.dry_run:
        print(f"\n[dry run] Would create pool '{pool_name}' and submit {num_jobs} jobs")
        print(f"  Pool config: {pool_path}")
        print(f"  Job config:  {job_path}")
        print("\nTo run manually:")
        print(f"  sky jobs pool apply -p {pool_name} {pool_path}")
        print(f"  sky jobs launch -p {pool_name} --num-jobs {num_jobs} {job_path}")
        return

    # build env flags
    env_flags = []
    for var in ENV_VARS_TO_FORWARD:
        val = os.environ.get(var)
        if val:
            env_flags.extend(["--env", f"{var}={val}"])

    # create pool
    logger.info(f"Creating pool '{pool_name}'...")
    subprocess.run(
        ["sky", "jobs", "pool", "apply", "-p", pool_name, str(pool_path), *env_flags],
        check=True,
    )

    # submit jobs
    logger.info(f"Submitting {num_jobs} jobs to pool '{pool_name}'...")
    subprocess.run(
        ["sky", "jobs", "launch", "-p", pool_name, "--num-jobs", str(num_jobs), "-y", str(job_path), *env_flags],
        check=True,
    )

    print(f"\nSubmitted {num_jobs} jobs to pool '{pool_name}'")
    print(f"Monitor with: sky jobs pool status {pool_name}")
    print(f"View logs:    sky jobs pool logs {pool_name}")
    print(f"Tear down:    sky jobs pool down {pool_name}")


if __name__ == "__main__":
    main()
