#!/usr/bin/env python3
"""
Dispatch distributed loading jobs via SkyPilot.

Discovers parquet files on S3, divides them into shards, generates per-shard
loader + SkyPilot configs, manages the Qdrant indexing lifecycle, and fans
out SkyPilot spot instance jobs.

Usage:
  vectorforge-dispatch configs/dispatch/cohere200M.yaml
  vectorforge-dispatch configs/dispatch/cohere200M.yaml --dry-run
  vectorforge-dispatch configs/dispatch/cohere200M.yaml --num-shards 20
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


def shard_files(files: list[str], num_shards: int) -> list[list[str]]:
    """Distribute files round-robin across N shards."""
    shards = [[] for _ in range(num_shards)]
    for i, f in enumerate(files):
        shards[i % num_shards].append(f)
    return shards


def generate_loader_yaml(
    base_config: dict,
    shard_files: list[str],
    shard_id: int,
    run_dir: Path,
) -> Path:
    """Write a per-shard loader config with an explicit file_list."""
    loader_config = {
        "datasource": {
            **base_config["datasource"],
            "file_list": shard_files,
        },
        "vectorstore": base_config["vectorstore"],
        "loader": base_config.get("loader", {}),
    }

    path = run_dir / f"shard_{shard_id:03d}_loader.yaml"
    with open(path, "w") as f:
        yaml.dump(loader_config, f, default_flow_style=False, sort_keys=False)
    return path


def generate_sky_yaml(
    shard_id: int,
    loader_config_path: Path,
    resources: dict,
    run_dir: Path,
) -> Path:
    """Write a per-shard SkyPilot job config. No env vars — injected at launch."""
    sky_config = {
        "resources": resources,
        "file_mounts": {
            "/app": ".",
        },
        "setup": "cd /app && pip install -e .",
        "run": f"cd /app && vectorforge-load {loader_config_path} --no-manage-indexing",
    }

    path = run_dir / f"shard_{shard_id:03d}_sky.yaml"
    with open(path, "w") as f:
        yaml.dump(sky_config, f, default_flow_style=False, sort_keys=False)
    return path


def launch_jobs(sky_configs: list[Path], run_name: str) -> list[str]:
    """Launch SkyPilot jobs with env vars forwarded from the current shell."""
    job_names = []
    env_flags = []
    for var in ENV_VARS_TO_FORWARD:
        val = os.environ.get(var)
        if val:
            env_flags.extend(["--env", f"{var}={val}"])

    for i, sky_yaml in enumerate(sky_configs):
        name = f"{run_name}-shard-{i:03d}"
        cmd = [
            "sky", "jobs", "launch", str(sky_yaml),
            "--async", "-y",
            "--name", name,
            *env_flags,
        ]
        logger.info(f"Launching {name}")
        subprocess.run(cmd, check=True)
        job_names.append(name)

    return job_names


def wait_for_jobs(job_names: list[str], poll_interval: float = 30.0) -> dict[str, str]:
    """Poll sky jobs queue until all named jobs reach a terminal state."""
    pending = set(job_names)
    results = {}
    terminal_states = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "CANCELLED"}

    while pending:
        result = subprocess.run(
            ["sky", "jobs", "queue", "--all"],
            capture_output=True, text=True,
        )
        output = result.stdout

        for name in list(pending):
            for line in output.split("\n"):
                if name in line:
                    for state in terminal_states:
                        if state in line:
                            pending.discard(name)
                            results[name] = state
                            logger.info(f"  {name}: {state}")
                            break
                    break

        if pending:
            done = len(job_names) - len(pending)
            logger.info(f"  {done}/{len(job_names)} complete, {len(pending)} pending...")
            time.sleep(poll_interval)

    return results


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Dispatch distributed loading via SkyPilot")
    parser.add_argument("config", help="Path to dispatch YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't launch")
    parser.add_argument("--num-shards", type=int, help="Override number of shards")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dispatch_cfg = config["dispatch"]
    resources = config["resources"]
    num_shards = args.num_shards or dispatch_cfg["num_shards"]
    config_name = Path(args.config).stem
    run_name = dispatch_cfg.get("run_name", config_name)

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

    # Shard files
    shards = shard_files(files, num_shards)
    logger.info(f"Divided into {num_shards} shards "
                f"({min(len(s) for s in shards)}-{max(len(s) for s in shards)} files each)")

    # Write manifest
    manifest = {
        "run_id": run_id,
        "config": args.config,
        "num_shards": num_shards,
        "total_files": len(files),
        "shards": {f"shard_{i:03d}": s for i, s in enumerate(shards)},
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate per-shard configs
    sky_configs = []
    for i, shard in enumerate(shards):
        loader_path = generate_loader_yaml(config, shard, i, run_dir)
        sky_path = generate_sky_yaml(i, loader_path, resources, run_dir)
        sky_configs.append(sky_path)

    logger.info(f"Generated configs in {run_dir}/")

    if args.dry_run:
        print(f"\n[dry run] Would launch {num_shards} SkyPilot jobs")
        print(f"  Run directory: {run_dir}")
        print(f"  Total files:   {len(files)}")
        print(f"  Shards:        {num_shards}")
        for i, shard in enumerate(shards):
            print(f"    shard_{i:03d}: {len(shard)} files")
        return

    # Resolve env vars for Qdrant setup
    resolved_config = resolve_config(config)

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

    # Launch SkyPilot jobs
    logger.info(f"Launching {num_shards} SkyPilot jobs...")
    t0 = time.perf_counter()
    job_names = launch_jobs(sky_configs, run_name)

    # Wait for all jobs
    logger.info("Waiting for jobs to complete...")
    results = wait_for_jobs(job_names)

    upload_elapsed = time.perf_counter() - t0
    succeeded = sum(1 for v in results.values() if v == "SUCCEEDED")
    failed = len(results) - succeeded

    logger.info(f"Upload phase: {succeeded} succeeded, {failed} failed in {upload_elapsed:.1f}s")

    if failed > 0:
        logger.warning(f"{failed} shards failed. Check logs with: sky jobs logs <job_name>")
        for name, status in results.items():
            if status != "SUCCEEDED":
                logger.warning(f"  {name}: {status}")

    # Enable indexing + wait
    logger.info("Enabling indexing...")
    t1 = time.perf_counter()
    asyncio.run(_enable_and_wait(store, dimension))
    index_elapsed = time.perf_counter() - t1

    # Write report
    report = {
        "run_id": run_id,
        "total_files": len(files),
        "num_shards": num_shards,
        "succeeded": succeeded,
        "failed": failed,
        "upload_seconds": round(upload_elapsed, 1),
        "index_seconds": round(index_elapsed, 1),
        "total_seconds": round(upload_elapsed + index_elapsed, 1),
        "jobs": results,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Done. Report: {run_dir / 'report.json'}")
    logger.info(f"  Upload:   {upload_elapsed:.1f}s")
    logger.info(f"  Indexing: {index_elapsed:.1f}s")
    logger.info(f"  Total:    {upload_elapsed + index_elapsed:.1f}s")


async def _setup_collection(store: QdrantVectorStore, dimension: int):
    await store.ensure_collection(dimension)
    await store.defer_indexing()
    await store.close()


async def _enable_and_wait(store: QdrantVectorStore, dimension: int):
    # Need a fresh client since the previous one was closed
    store._client = type(store._client)(url=store.url, api_key=store.api_key)
    await store.enable_indexing()
    await store.wait_for_indexing()
    await store.close()


if __name__ == "__main__":
    main()
