#!/usr/bin/env python3
"""
Dispatch distributed embedding jobs via SkyPilot pools.

Creates a pool of GPU workers and submits parallel embedding jobs. Each job
processes a slice of the dataset using --num-jobs/--job-rank for automatic
partitioning.

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
    "cloud": "aws",
    "use_spot": True,
}


def main():
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
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    source_cfg = config["source"]
    pipeline_cfg = config.get("pipeline", {})
    resources = config.get("resources", DEFAULT_RESOURCES)

    dataset_name = source_cfg["dataset_name"]
    hf_config = source_cfg.get("config")
    split = source_cfg.get("split", "train")

    # Get dataset size
    from datasets import load_dataset_builder
    builder = load_dataset_builder(dataset_name, hf_config)
    total_rows = builder.info.splits[split].num_examples

    chunk_size = args.chunk_size or pipeline_cfg.get("chunk_size", 100_000)
    num_jobs = args.num_jobs or math.ceil(total_rows / chunk_size)
    max_workers = args.max_workers or num_jobs

    config_name = Path(args.config).stem
    pool_name = args.pool_name or f"vf-embed-{config_name}"

    # Create run directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_id = f"{timestamp}_{pool_name}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("vectorforge distributed embedding plan")
    print("=" * 60)
    print(f"  Dataset:      {dataset_name}")
    print(f"  Split:        {split}")
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Num jobs:     {num_jobs}")
    print(f"  Rows/job:     ~{math.ceil(total_rows / num_jobs):,}")
    print(f"  Max workers:  {max_workers}")
    print(f"  Pool name:    {pool_name}")
    print(f"  Resources:    {resources}")
    print(f"  Run dir:      {run_dir}")
    print("=" * 60)

    # Generate pool YAML
    pool_yaml = {
        "pool": {
            "min_workers": 0,
            "max_workers": max_workers,
        },
        "resources": resources,
        "file_mounts": {
            "/app": ".",
        },
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync",
    }
    pool_path = run_dir / "pool.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)

    # Generate job YAML
    job_yaml = {
        "name": f"embed-{config_name}",
        "resources": resources,
        "run": f"cd /app && vectorforge {args.config} --num-jobs {num_jobs}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    # Write manifest
    manifest = {
        "run_id": run_id,
        "config": args.config,
        "dataset": dataset_name,
        "split": split,
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
        print(f"\nTo run manually:")
        print(f"  sky jobs pool apply -p {pool_name} {pool_path}")
        print(f"  sky jobs launch -p {pool_name} --num-jobs {num_jobs} {job_path}")
        return

    # Build env flags
    env_flags = []
    for var in ENV_VARS_TO_FORWARD:
        val = os.environ.get(var)
        if val:
            env_flags.extend(["--env", f"{var}={val}"])

    # Create pool
    logger.info(f"Creating pool '{pool_name}'...")
    subprocess.run(
        ["sky", "jobs", "pool", "apply", "-p", pool_name, str(pool_path), *env_flags],
        check=True,
    )

    # Submit jobs
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
