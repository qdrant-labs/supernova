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


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Load pre-embedded data into a vector store")
    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Parse config and print info without loading")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("LOADER_CONFIG_PATH")
    if not config_path:
        parser.error("Provide a config path as argument or set LOADER_CONFIG_PATH env var")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve environment variables throughout the config
    config = resolve_config(config)

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
        )
    )


if __name__ == "__main__":
    main()
