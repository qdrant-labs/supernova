#!/usr/bin/env python3
"""
Dispatch distributed `vf partition` jobs via SkyPilot pools.

Same shape as `vf embed-dist` but with CPU-only resources (no GPU) and the
no-op embedder, so each worker just splits/writes raw rows. Lets you validate
that ranks read non-overlapping slices and the S3 layout is what you expect
before committing to a real (expensive) embed run.

Usage:
  vf partition-dist configs/embedder/ccnews_2016.yaml
  vf partition-dist configs/embedder/ccnews_2016.yaml --dry-run
  vf partition-dist configs/embedder/ccnews_2016.yaml --num-jobs 10
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
]

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


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Dispatch distributed `vf partition` via SkyPilot pools")
    parser.add_argument("config", help="Path to embedder YAML config (same schema as `vf embed`)")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and print plan, don't launch")
    parser.add_argument("--num-jobs", type=int, help="Number of parallel jobs (default: auto from dataset size)")
    parser.add_argument("--chunk-size", type=int, help="Rows per job (used to auto-compute num-jobs)")
    parser.add_argument("--pool-name", type=str, help="SkyPilot pool name (default: auto-generated)")
    parser.add_argument("--max-workers", type=int, help="Max pool workers for autoscaling (default: num-jobs)")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use on-demand instances instead of spot")
    parser.add_argument("--ramp", action="store_true",
                        help="Use SkyPilot's gradual autoscaler (min_workers=0). Default is burst.")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    source_cfg = config["source"]
    pipeline_cfg = config.get("pipeline", {})
    # `partition.resources:` may override CPU defaults; fall back to embed-style
    # `resources:` if set; finally fall back to DEFAULT_RESOURCES.
    partition_cfg = config.get("partition") or {}
    resources = partition_cfg.get("resources") or config.get("resources") or dict(DEFAULT_RESOURCES)
    if args.on_demand:
        resources = dict(resources)
        resources["use_spot"] = False

    # get dataset size (source-agnostic)
    from cli.run_embedder import build_source
    source = build_source(dict(source_cfg))
    total_rows = source.get_total_rows()

    chunk_size = args.chunk_size or pipeline_cfg.get("chunk_size", 100_000)
    num_jobs = args.num_jobs or math.ceil(total_rows / chunk_size)
    max_workers = args.max_workers or num_jobs

    config_name = Path(args.config).stem
    pool_name = args.pool_name or f"vf-partition-{config_name}"

    # create run directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_id = f"{timestamp}_{pool_name}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("vectorforge distributed partition plan (no-op embedder)")
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

    # pool YAML -- CPU only. Setup mirrors embed-dist (uv install + sync).
    pool_yaml = {
        "pool": {
            "min_workers": 0 if args.ramp else max_workers,
            "max_workers": max_workers,
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
        "run": f"cd /app && uv run vf partition {args.config} --num-jobs {num_jobs}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    manifest = {
        "run_id": run_id,
        "config": args.config,
        "source": source.source_name,
        "total_rows": total_rows,
        "num_jobs": num_jobs,
        "max_workers": max_workers,
        "pool_name": pool_name,
        "mode": "partition",
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

    env_flags = []
    for var in ENV_VARS_TO_FORWARD:
        val = os.environ.get(var)
        if val:
            env_flags.extend(["--env", f"{var}={val}"])

    logger.info(f"Creating pool '{pool_name}'...")
    subprocess.run(
        ["sky", "jobs", "pool", "apply", "-p", pool_name, str(pool_path), *env_flags],
        check=True,
    )

    logger.info(f"Submitting {num_jobs} jobs to pool '{pool_name}'...")
    subprocess.run(
        ["sky", "jobs", "launch", "-p", pool_name, "--num-jobs", str(num_jobs), "-y", str(job_path), *env_flags],
        check=True,
    )

    print(f"\nSubmitted {num_jobs} partition jobs to pool '{pool_name}'")
    print(f"Monitor with: sky jobs pool status {pool_name}")
    print(f"View logs:    sky jobs pool logs {pool_name}")
    print(f"Tear down:    sky jobs pool down {pool_name}")
    print()
    print("Once all ranks finish, verify clean partitioning with:")
    print("  uv run python scripts/verify_no_duplicates.py s3://<your-prefix>/ --content-columns text")


if __name__ == "__main__":
    main()
