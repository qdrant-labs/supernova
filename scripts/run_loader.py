import argparse
import asyncio
import logging
import os
import re

import yaml

from vectorforge.loader.datasource.s3 import S3DataReader
from vectorforge.loader.datasource.huggingface import HuggingFaceDataReader
from vectorforge.loader.vectorstore.qdrant import QdrantVectorStore
from vectorforge.loader.runner import run_loader


DATASOURCE_REGISTRY = {
    "s3": S3DataReader,
    "huggingface": HuggingFaceDataReader,
}

VECTORSTORE_REGISTRY = {
    "qdrant": QdrantVectorStore,
}


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


def resolve_config(obj):
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


def build_reader(cfg: dict):
    source_type = cfg.pop("type", "s3")
    cls = DATASOURCE_REGISTRY.get(source_type)
    if cls is None:
        raise ValueError(f"Unknown datasource type: {source_type}. Available: {list(DATASOURCE_REGISTRY)}")
    return cls(**cfg)


def build_vectorstore(cfg: dict):
    store_type = cfg.pop("type")
    cls = VECTORSTORE_REGISTRY.get(store_type)
    if cls is None:
        raise ValueError(f"Unknown vectorstore type: {store_type}. Available: {list(VECTORSTORE_REGISTRY)}")

    # Extract known top-level fields, pass rest as params
    url = cfg.pop("url", None)
    api_key = cfg.pop("api_key", None)
    collection_name = cfg.pop("collection_name", None)
    params = cfg.pop("params", {})

    kwargs = {"params": params}
    if url is not None:
        kwargs["url"] = url
    if api_key is not None:
        kwargs["api_key"] = api_key
    if collection_name is not None:
        kwargs["collection_name"] = collection_name

    return cls(**kwargs)


def _discover_and_shard(ds_cfg: dict, num_jobs: int, job_rank: int) -> list[str]:
    """Discover parquet files on S3 and return this job's shard."""
    import boto3

    bucket = ds_cfg["s3_bucket"]
    prefix = ds_cfg["s3_prefix"].rstrip("/")

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                files.append(f"s3://{bucket}/{key}")
    files.sort()

    # Round-robin assignment
    shard = [f for i, f in enumerate(files) if i % num_jobs == job_rank]
    logging.getLogger("vectorforge").info(
        "Job %d/%d: %d files (of %d total)", job_rank, num_jobs, len(shard), len(files),
    )
    return shard


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Load pre-embedded data into a vector store")
    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Parse config and print info without loading")
    parser.add_argument("--no-manage-indexing", action="store_true", default=False,
                        help="Skip collection creation and indexing lifecycle (for distributed workers)")
    parser.add_argument("--num-jobs", type=int, default=None,
                        help="Total number of parallel jobs (auto-shards files by rank)")
    parser.add_argument("--job-rank", type=int, default=None,
                        help="This job's rank (0-indexed, used with --num-jobs)")
    args = parser.parse_args(argv)

    config_path = args.config or os.environ.get("LOADER_CONFIG_PATH")
    if not config_path:
        parser.error("Provide a config path as argument or set LOADER_CONFIG_PATH env var")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve environment variables throughout the config
    config = resolve_config(config)

    # Rank-based file sharding for distributed loading
    if args.num_jobs is not None:
        job_rank = args.job_rank
        if job_rank is None:
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))

        shard_files = _discover_and_shard(config["datasource"], args.num_jobs, job_rank)
        if not shard_files:
            logging.getLogger("vectorforge").info("No files assigned to this shard, exiting.")
            return
        config["datasource"]["file_list"] = shard_files

    reader = build_reader(config["datasource"])
    store = build_vectorstore(dict(config["vectorstore"]))

    loader_cfg = config.get("loader", {})

    if args.dry_run:
        print("Config parsed successfully. Reader and VectorStore instances created.")
        print(f"Reader: {reader}")
        print(f"VectorStore: {store}")
        return

    asyncio.run(
        run_loader(
            reader=reader,
            store=store,
            batch_size=loader_cfg.get("batch_size", 1000),
            prefetch_size=loader_cfg.get("prefetch_size"),
            concurrency=loader_cfg.get("concurrency", 8),
            manage_indexing=not args.no_manage_indexing,
        )
    )


if __name__ == "__main__":
    main()
