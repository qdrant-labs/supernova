#!/usr/bin/env python3
"""
Dispatch distributed HuggingFace Hub upload jobs via SkyPilot pools.

Each worker downloads its assigned S3 parquet files (fast — S3 to EC2 is
in-region and effectively free) and streams them up to HuggingFace directly
from the datacenter, bypassing your local machine's upload bandwidth entirely.

Usage:
  vf push-hf-dist s3://qdrant--vectorforge/fineweb/embed-bge-large/ nleroy917/fineweb-bge-large
  vf push-hf-dist s3://... username/repo --num-jobs 50 --dry-run
  vf push-hf-dist s3://... username/repo --private
"""

import argparse
import json
import logging
import os
import subprocess

from datetime import datetime
from pathlib import Path

import boto3
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
    from vectorforge.utils import discover_corpus_parquets
    return discover_corpus_parquets(bucket, prefix)


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Dispatch distributed HF Hub uploads via SkyPilot")
    parser.add_argument("s3_uri", help="s3://bucket/prefix — source parquet files")
    parser.add_argument("repo_id", help="HF repo id, e.g. 'username/dataset-name'")
    parser.add_argument("--num-jobs", type=int, help="Number of parallel workers (default: auto from file count)")
    parser.add_argument("--files-per-job", type=int, default=20,
                        help="Files per worker when auto-computing num-jobs (default: 20)")
    parser.add_argument("--subfolder", default="data", help="Folder inside the HF repo (default: data)")
    parser.add_argument("--private", action="store_true", help="Create HF repo as private")
    parser.add_argument("--pool-name", type=str, help="SkyPilot pool name (default: auto-generated)")
    parser.add_argument("--on-demand", action="store_true", help="Use on-demand instead of spot")
    parser.add_argument("--ramp", action="store_true",
                        help="Ramp workers gradually (min_workers=0). Default is burst.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and generate configs, don't launch")
    args = parser.parse_args(argv)

    if not args.s3_uri.startswith("s3://"):
        parser.error("s3_uri must start with s3://")
    without_scheme = args.s3_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    logger.info("Listing parquet files at s3://%s/%s/...", bucket, prefix)
    keys = list_s3_parquets(bucket, prefix)
    if not keys:
        logger.error("No parquet files found at %s", args.s3_uri)
        return

    num_jobs = args.num_jobs or max(1, len(keys) // args.files_per_job)
    resources = dict(DEFAULT_RESOURCES)
    if args.on_demand:
        resources["use_spot"] = False

    repo_slug = args.repo_id.replace("/", "--")
    pool_name = args.pool_name or f"vf-push-hf-{repo_slug}"

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_id = f"{timestamp}_{pool_name}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("vectorforge distributed HuggingFace push plan")
    print("=" * 60)
    print(f"  Source:       s3://{bucket}/{prefix}")
    print(f"  Total files:  {len(keys)}")
    print(f"  Num workers:  {num_jobs}")
    print(f"  Files/worker: ~{len(keys) // num_jobs}")
    print(f"  HF repo:      {args.repo_id}")
    print(f"  Subfolder:    {args.subfolder}")
    print(f"  Pool name:    {pool_name}")
    print(f"  Provision:    {'ramp (autoscaler)' if args.ramp else 'burst (all workers at startup)'}")
    print(f"  Resources:    {resources}")
    print(f"  Run dir:      {run_dir}")
    print("=" * 60)

    # Workers only need base deps (boto3 + huggingface_hub) — no extras required.
    # hf_transfer gives ~5x faster uploads; HF_HUB_ENABLE_HF_TRANSFER activates it.
    pool_yaml = {
        "pool": {
            "min_workers": 0 if args.ramp else num_jobs,
            "max_workers": num_jobs,
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

    push_flags = f"--num-jobs {num_jobs} --subfolder {args.subfolder}"
    if args.private:
        push_flags += " --private"

    job_yaml = {
        "name": f"push-hf-{repo_slug}",
        "resources": resources,
        "envs": {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        "run": f"cd /app && uv run vf push-hf {args.s3_uri} {args.repo_id} {push_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    manifest = {
        "run_id": run_id,
        "s3_uri": args.s3_uri,
        "repo_id": args.repo_id,
        "total_files": len(keys),
        "num_jobs": num_jobs,
        "pool_name": pool_name,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

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

    logger.info("Creating pool '%s'...", pool_name)
    subprocess.run(
        ["sky", "jobs", "pool", "apply", "-p", pool_name, str(pool_path), *env_flags],
        check=True,
    )

    logger.info("Submitting %d jobs to pool '%s'...", num_jobs, pool_name)
    subprocess.run(
        ["sky", "jobs", "launch", "-p", pool_name, "--num-jobs", str(num_jobs), "-y", str(job_path), *env_flags],
        check=True,
    )

    print(f"\nSubmitted {num_jobs} upload jobs to pool '{pool_name}'")
    print(f"Monitor with: sky jobs pool status {pool_name}")
    print(f"View logs:    sky jobs pool logs {pool_name}")
    print(f"Tear down:    sky jobs pool down {pool_name}")
    print(f"Dataset:      https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()