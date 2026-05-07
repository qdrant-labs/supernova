#!/usr/bin/env python3
"""
Dispatch distributed loading jobs via SkyPilot pools.

Discovers parquet files on S3, creates a pool of CPU workers, and submits
parallel loading jobs. Each job discovers files and picks its shard via
--num-jobs + $SKYPILOT_JOB_RANK.

Also manages the Qdrant indexing lifecycle: defers indexing before loading,
enables it after with --finalize.

Reads the same configs as `vf load` (configs/loader/*.yaml). The single-machine
loader ignores `dispatch:` / `resources:` blocks; this CLI consumes them.

Usage:
  vf load-dist configs/loader/ccnews_bge_large.yaml
  vf load-dist configs/loader/ccnews_bge_large.yaml --dry-run
  vf load-dist configs/loader/ccnews_bge_large.yaml --num-shards 20
  vf load-dist configs/loader/ccnews_bge_large.yaml --finalize   # enable indexing after jobs complete
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import yaml

from cli.skypilot_utils import build_env_flags, make_run_dir, launch_pool_and_jobs, print_dry_run, print_monitor
from vectorforge.loader.datasource.s3 import S3DataReader
from vectorforge.loader.vectorstore.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


def resolve_env_vars(value: str) -> str:
    """
    Replace ${VAR_NAME} references with environment variable values.
    """
    def _replace(match):
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")
        return val

    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", _replace, value)
    return value


def resolve_config(obj: str | dict | list):
    """
    Recursively resolve env vars in config values.
    """
    if isinstance(obj, str):
        return resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_config(v) for v in obj]
    return obj


def discover_parquet_files(bucket: str, prefix: str) -> list[str]:
    """List all corpus .parquet files under an S3 prefix, excluding eval/ artifacts."""
    from vectorforge.utils import discover_corpus_parquets
    return [f"s3://{bucket}/{k}" for k in discover_corpus_parquets(bucket, prefix)]


async def _setup_collection(store: QdrantVectorStore, dimensions: dict[str, int]):
    await store.ensure_collection(dimensions)
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
    parser.add_argument("--on-demand", action="store_true", help="Use on-demand instances instead of spot (higher cost, no preemption, separate AWS quota)")
    parser.add_argument("--ramp", action="store_true", help="Let SkyPilot's autoscaler bring workers up gradually (min_workers=0). Default is burst (min_workers=max_workers) since EC2 provisioning is slow and we know the target count up front.")
    parser.add_argument("--finalize", action="store_true", help="Enable Qdrant indexing (run after all jobs complete)")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    resolved_config = resolve_config(config)

    vectors_spec = resolved_config.get("vectors")
    if not vectors_spec:
        raise SystemExit("config is missing required top-level 'vectors:' block")

    # just enable indexing and exit
    if args.finalize:
        logger.info("Enabling Qdrant indexing...")
        vs_cfg = dict(resolved_config["vectorstore"])
        vs_cfg.pop("type", None)
        vs_cfg.pop("params", None)
        store = QdrantVectorStore(vectors=vectors_spec, **vs_cfg)

        t0 = time.perf_counter()
        asyncio.run(_enable_and_wait(store))
        elapsed = time.perf_counter() - t0
        logger.info(f"Indexing complete in {elapsed:.1f}s")
        return

    dispatch_cfg = config["dispatch"]
    resources = dict(config["resources"])
    if args.on_demand:
        resources["use_spot"] = False
    num_shards = args.num_shards or dispatch_cfg["num_shards"]
    config_name = Path(args.config).stem
    run_name = dispatch_cfg.get("run_name", config_name)
    pool_name = args.pool_name or f"vf-load-{run_name}"

    # create run directory
    run_dir = make_run_dir(run_name)
    run_id = run_dir.name

    # discover parquet files
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
    print(f"  Provision:    {'ramp (autoscaler)' if args.ramp else 'burst (all workers at startup)'}")
    print(f"  Resources:    {resources}")
    print(f"  Run dir:      {run_dir}")
    print("=" * 60)

    # generate pool YAML — burst by default (provision all workers at startup);
    # SkyPilot's autoscaler ramp is too slow when we know the target count up front.
    pool_yaml = {
        "pool": {
            "min_workers": 0 if args.ramp else num_shards,
            "max_workers": num_shards,
        },
        "resources": resources,
        "file_mounts": {
            "/app": ".",
        },
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync --extra load",
    }
    pool_path = run_dir / "pool.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)

    # generate job YAML
    job_yaml = {
        "name": f"load-{run_name}",
        "resources": resources,
        "run": f"cd /app && uv run vf load {args.config} --num-jobs {num_shards} --no-manage-indexing",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    # write manifest
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
        print_dry_run(pool_name, num_shards, pool_path, job_path)
        return

    # Setup Qdrant: create collection + defer indexing
    logger.info("Setting up Qdrant collection...")
    vs_cfg = dict(resolved_config["vectorstore"])
    vs_cfg.pop("type", None)
    vs_cfg.pop("params", None)
    store = QdrantVectorStore(vectors=vectors_spec, **vs_cfg)

    # Use the first corpus file to probe vector dimensions. Assumes all files
    # share the same schema (guaranteed by the embedding pipeline) and avoids
    # the glob hitting eval/ parquets that have a different schema entirely.
    reader = S3DataReader(s3_bucket=bucket, s3_prefix=prefix, vectors=vectors_spec, file_list=[files[0]])
    dimensions = reader.get_dimensions()
    reader.close()

    asyncio.run(_setup_collection(store, dimensions))
    logger.info("Qdrant collection ready (indexing deferred)")

    env_flags = build_env_flags(["QDRANT_URL", "QDRANT_API_KEY", "HF_TOKEN"])
    logger.info(f"Creating pool '{pool_name}'...")
    logger.info(f"Submitting {num_shards} jobs to pool '{pool_name}'...")
    launch_pool_and_jobs(pool_name, pool_path, job_path, num_shards, env_flags)

    print(f"\nSubmitted {num_shards} loading jobs to pool '{pool_name}'")
    print_monitor(pool_name)
    print("\nAfter all jobs complete, enable Qdrant indexing:")
    print(f"  vf load-dist {args.config} --finalize")


if __name__ == "__main__":
    main()
