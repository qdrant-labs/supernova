#!/usr/bin/env python3
"""
Dispatch distributed loading jobs via SkyPilot pools.

Discovers parquet files on S3, creates a pool of CPU workers, and submits
parallel loading jobs. Each job discovers files and picks its shard via
--num-jobs + $SKYPILOT_JOB_RANK.

Also manages the Qdrant indexing lifecycle: defers indexing before loading,
enables it after with --finalize.

Usage:
  vf load-dist configs/dispatch/cohere200M.yaml
  vf load-dist configs/dispatch/cohere200M.yaml --dry-run
  vf load-dist configs/dispatch/cohere200M.yaml --num-shards 20
  vf load-dist configs/dispatch/cohere200M.yaml --finalize   # enable indexing after jobs complete
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import time

from datetime import datetime
from pathlib import Path

import boto3
import yaml

from vectorforge.loader.datasource.s3 import S3DataReader
from vectorforge.loader.vectorstore.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

ENV_VARS_TO_FORWARD = [
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "HF_TOKEN",
]


def resolve_env_vars(value):
    """Replace ${VAR_NAME} references with environment variable values."""
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
    """Recursively resolve env vars in config values."""
    if isinstance(obj, str):
        return resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_config(v) for v in obj]
    return obj


def discover_parquet_files(bucket: str, prefix: str) -> list[str]:
    """List all .parquet files under an S3 prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                files.append(f"s3://{bucket}/{key}")
    return sorted(files)


async def _setup_collection(store: QdrantVectorStore, dimension: int):
    await store.ensure_collection(dimension)
    await store.defer_indexing()
    await store.close()


async def _enable_and_wait(store: QdrantVectorStore):
    await store.enable_indexing()
    await store.wait_for_indexing()
    await store.close()


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Dispatch distributed loading via SkyPilot pools")
    parser.add_argument("config", help="Path to dispatch YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and print plan, don't launch")
    parser.add_argument("--num-shards", type=int, help="Override number of shards")
    parser.add_argument("--pool-name", type=str, help="SkyPilot pool name (default: auto-generated)")
    parser.add_argument("--finalize", action="store_true",
                        help="Enable Qdrant indexing (run after all jobs complete)")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    resolved_config = resolve_config(config)

    # --finalize: just enable indexing and exit
    if args.finalize:
        logger.info("Enabling Qdrant indexing...")
        vs_cfg = dict(resolved_config["vectorstore"])
        vs_cfg.pop("type", None)
        vs_cfg.pop("params", None)
        store = QdrantVectorStore(**vs_cfg)

        t0 = time.perf_counter()
        asyncio.run(_enable_and_wait(store))
        elapsed = time.perf_counter() - t0
        logger.info(f"Indexing complete in {elapsed:.1f}s")
        return

    dispatch_cfg = config["dispatch"]
    resources = config["resources"]
    num_shards = args.num_shards or dispatch_cfg["num_shards"]
    config_name = Path(args.config).stem
    run_name = dispatch_cfg.get("run_name", config_name)
    pool_name = args.pool_name or f"vf-load-{run_name}"

    # Create run directory
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_id = f"{timestamp}_{run_name}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Discover parquet files
    ds_cfg = config["datasource"]
    bucket = ds_cfg["s3_bucket"]
    prefix = ds_cfg["s3_prefix"]
    logger.info(f"Discovering parquet files at s3://{bucket}/{prefix}/...")
    files = discover_parquet_files(bucket, prefix)
    logger.info(f"Found {len(files)} parquet files")

    if not files:
        logger.error("No parquet files found. Exiting.")
        return

    print("=" * 60)
    print("vectorforge distributed loading plan")
    print("=" * 60)
    print(f"  S3 prefix:    s3://{bucket}/{prefix}")
    print(f"  Total files:  {len(files)}")
    print(f"  Num shards:   {num_shards}")
    print(f"  Files/shard:  ~{len(files) // num_shards}")
    print(f"  Pool name:    {pool_name}")
    print(f"  Resources:    {resources}")
    print(f"  Run dir:      {run_dir}")
    print("=" * 60)

    # Generate pool YAML
    pool_yaml = {
        "pool": {
            "min_workers": 0,
            "max_workers": num_shards,
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
        "name": f"load-{run_name}",
        "resources": resources,
        "run": f"cd /app && vf load {args.config} --num-jobs {num_shards} --no-manage-indexing",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    # Write manifest
    manifest = {
        "run_id": run_id,
        "config": args.config,
        "num_shards": num_shards,
        "total_files": len(files),
        "pool_name": pool_name,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated configs in {run_dir}/")

    if args.dry_run:
        print(f"\n[dry run] Would create pool '{pool_name}' and submit {num_shards} jobs")
        print(f"  Pool config: {pool_path}")
        print(f"  Job config:  {job_path}")
        print(f"\nTo run manually:")
        print(f"  sky jobs pool apply -p {pool_name} {pool_path}")
        print(f"  sky jobs launch -p {pool_name} --num-jobs {num_shards} {job_path}")
        return

    # Setup Qdrant: create collection + defer indexing
    logger.info("Setting up Qdrant collection...")
    vs_cfg = dict(resolved_config["vectorstore"])
    vs_cfg.pop("type", None)
    vs_cfg.pop("params", None)
    store = QdrantVectorStore(**vs_cfg)

    reader = S3DataReader(s3_bucket=bucket, s3_prefix=prefix)
    dimension = reader.get_dimensions()
    reader.close()

    asyncio.run(_setup_collection(store, dimension))
    logger.info("Qdrant collection ready (indexing deferred)")

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
    logger.info(f"Submitting {num_shards} jobs to pool '{pool_name}'...")
    subprocess.run(
        ["sky", "jobs", "launch", "-p", pool_name, "--num-jobs", str(num_shards), "-y", str(job_path), *env_flags],
        check=True,
    )

    print(f"\nSubmitted {num_shards} loading jobs to pool '{pool_name}'")
    print(f"\nMonitor:    sky jobs pool status {pool_name}")
    print(f"View logs:  sky jobs pool logs {pool_name}")
    print(f"Tear down:  sky jobs pool down {pool_name}")
    print(f"\nAfter all jobs complete, enable Qdrant indexing:")
    print(f"  vf load-dist {args.config} --finalize")


if __name__ == "__main__":
    main()
