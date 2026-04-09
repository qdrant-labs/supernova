#!/usr/bin/env python3
"""CLI entrypoint for vectorforge pipelines."""

import argparse
import asyncio
import logging
import os
import yaml

from vectorforge.sources.huggingface import HuggingFaceSource
from vectorforge.embedders.openai import OpenAIEmbedder
from vectorforge.embedders.sentence_transformer import SentenceTransformerEmbedder
from vectorforge.storage.s3 import S3Backend
from vectorforge.storage.huggingface import HuggingFaceBackend
from vectorforge.storage.local import LocalBackend
from vectorforge.pipeline.runner import run


SOURCE_REGISTRY = {
    "huggingface": HuggingFaceSource,
}

EMBEDDER_REGISTRY = {
    "openai": OpenAIEmbedder,
    "sentence_transformer": SentenceTransformerEmbedder,
}


def build_source(cfg: dict):
    source_type = cfg.pop("type")
    cls = SOURCE_REGISTRY[source_type]
    return cls(**cfg)


def build_embedder(cfg: dict) -> OpenAIEmbedder | SentenceTransformerEmbedder:
    embedder_type = cfg.pop("type")
    cls = EMBEDDER_REGISTRY[embedder_type]
    return cls(**cfg)


def build_storage(cfg: dict):
    storage_type = cfg.pop("type", "s3")
    if storage_type == "s3":
        return S3Backend(
            bucket=cfg["s3_bucket"],
            prefix=cfg["s3_prefix"],
        )
    elif storage_type == "hf":
        return HuggingFaceBackend(
            repo_id=cfg["repo_id"],
            token=cfg.get("token"),
            private=cfg.get("private", True),
        )
    elif storage_type == "local":
        return LocalBackend(
            output_dir=cfg.get("output_dir", "/tmp/vectorforge"),
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Run a vectorforge embedding pipeline")
    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("CONFIG_PATH")
    if not config_path:
        parser.error("Provide a config path as argument or set CONFIG_PATH env var")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    source = build_source(dict(config["source"]))
    embedder = build_embedder(dict(config["embedder"]))
    storage = build_storage(dict(config["storage"]))

    pipeline_cfg = config.get("pipeline", {})
    storage_cfg = config.get("storage", {})

    asyncio.run(
        run(
            source=source,
            embedder=embedder,
            storage=storage,
            chunk_size=pipeline_cfg.get("chunk_size", 10_000),
            num_workers=pipeline_cfg.get("num_workers", 8),
            flush_threshold=pipeline_cfg.get("flush_threshold", 100_000),
            output_dir=storage_cfg.get("output_dir", "/tmp/vectorforge"),
            max_text_length=pipeline_cfg.get("max_text_length"),
        )
    )


if __name__ == "__main__":
    main()