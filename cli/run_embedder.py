import argparse
import asyncio
import logging
import math
import os
import yaml

from vectorforge.sources.huggingface import HuggingFaceSource
from vectorforge.embedders.dense.openai import OpenAIEmbedder
from vectorforge.embedders.dense.sentence_transformer import SentenceTransformerDenseEmbedder
from vectorforge.embedders.sparse.sentence_transformer import SentenceTransformerSparseEmbedder
from vectorforge.embedders.hybrid import SentenceTransformerHybridEmbedder
from vectorforge.embedders.engine import EmbeddingEngine
from vectorforge.storage.s3 import S3Backend
from vectorforge.storage.huggingface import HuggingFaceBackend
from vectorforge.storage.local import LocalBackend
from vectorforge.pipeline.runner import run

# available sources, embedders, and storage backends. Used to construct from config.
# mapping from string identifiers in config → actual classes. Factored out to avoid circular imports and keep main() clean.
SOURCE_REGISTRY = {
    "huggingface": HuggingFaceSource,
}

DENSE_EMBEDDER_REGISTRY = {
    "openai": OpenAIEmbedder,
    "sentence_transformer": SentenceTransformerDenseEmbedder,
}

SPARSE_EMBEDDER_REGISTRY = {
    "sentence_transformer": SentenceTransformerSparseEmbedder,
}


def build_source(cfg: dict):
    source_type = cfg.pop("type")
    cls = SOURCE_REGISTRY[source_type]
    return cls(**cfg)


def build_dense_embedder(cfg: dict):
    embedder_type = cfg.pop("type")
    cls = DENSE_EMBEDDER_REGISTRY.get(embedder_type)
    if cls is None:
        raise ValueError(f"Unknown dense embedder type: {embedder_type}. Available: {list(DENSE_EMBEDDER_REGISTRY)}")
    return cls(**cfg)


def build_sparse_embedder(cfg: dict):
    embedder_type = cfg.pop("type")
    cls = SPARSE_EMBEDDER_REGISTRY.get(embedder_type)
    if cls is None:
        raise ValueError(f"Unknown sparse embedder type: {embedder_type}. Available: {list(SPARSE_EMBEDDER_REGISTRY)}")
    return cls(**cfg)


def _can_hybrid(dense_cfg: dict, sparse_cfg: dict) -> bool:
    """
    Check if dense and sparse configs point to the same sentence_transformer model.
    """
    return (
        dense_cfg.get("type") == sparse_cfg.get("type") == "sentence_transformer"
        and dense_cfg.get("model") == sparse_cfg.get("model")
    )


def build_engine(config: dict) -> EmbeddingEngine:
    """
    Build an EmbeddingEngine from config.

    Supports:
      - dense_embedder only
      - sparse_embedder only
      - both (auto-detects hybrid when same model)
      - legacy 'embedder' key (treated as dense_embedder)
    """
    dense_cfg = dict(config.get("dense_embedder") or {})
    sparse_cfg = dict(config.get("sparse_embedder") or {})

    if not dense_cfg and not sparse_cfg:
        raise ValueError("Config must specify at least one of: dense_embedder, sparse_embedder")

    # Detect hybrid case: same model for both → single forward pass
    if dense_cfg and sparse_cfg and _can_hybrid(dense_cfg, sparse_cfg):
        hybrid_cfg = dict(dense_cfg)
        hybrid_cfg.pop("type")
        hybrid = SentenceTransformerHybridEmbedder(**hybrid_cfg)
        return EmbeddingEngine(hybrid=hybrid)

    # Build separately
    dense = build_dense_embedder(dense_cfg) if dense_cfg else None
    sparse = build_sparse_embedder(sparse_cfg) if sparse_cfg else None
    return EmbeddingEngine(dense=dense, sparse=sparse)


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


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Run a vectorforge embedding pipeline")
    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    parser.add_argument("--offset", type=int, default=None, help="Skip this many rows (for distributed slicing)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many rows (for distributed slicing)")
    parser.add_argument("--num-jobs", type=int, default=None, help="Total number of parallel jobs (auto-computes offset/limit from dataset size)")
    parser.add_argument("--job-rank", type=int, default=None, help="This job's rank (0-indexed, used with --num-jobs)")
    args = parser.parse_args(argv)

    config_path = args.config or os.environ.get("VF_CONFIG_PATH")
    if not config_path:
        parser.error("Provide a config path as argument or set VF_CONFIG_PATH env var")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # support both explicit offset/limit and rank-based slicing
    if args.num_jobs is not None:
        job_rank = args.job_rank
        if job_rank is None:
            # SkyPilot pools set these env vars automatically
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))
            logging.getLogger("vectorforge").info(
                f"Auto-detected job rank {job_rank} from SKYPILOT_JOB_RANK env var"
            )

        # Build source to query total rows (source-agnostic)
        source_for_count = build_source(dict(config["source"]))
        total_rows = source_for_count.get_total_rows()

        rows_per_job = math.ceil(total_rows / args.num_jobs)
        offset = job_rank * rows_per_job
        limit = min(rows_per_job, total_rows - offset)

        logging.getLogger("vectorforge").info(
            "Job %d/%d: offset=%d limit=%d (total=%d)",
            job_rank, args.num_jobs, offset, limit, total_rows,
        )
        config["source"]["offset"] = offset
        config["source"]["limit"] = limit

    elif args.offset is not None or args.limit is not None:
        if args.offset is not None:
            # just use what was provided, no auto-computation
            config["source"]["offset"] = args.offset
        if args.limit is not None:
            # just use what was provided, no auto-computation
            config["source"]["limit"] = args.limit

    source = build_source(dict(config["source"]))
    engine = build_engine(config)
    storage = build_storage(dict(config["storage"]))

    pipeline_cfg = config.get("pipeline", {})
    storage_cfg = config.get("storage", {})

    dense_column = pipeline_cfg.get("dense_embedding_column", "dense_embedding") if engine.has_dense else None
    sparse_column = pipeline_cfg.get("sparse_embedding_column", "sparse_embedding") if engine.has_sparse else None

    asyncio.run(
        run(
            source=source,
            engine=engine,
            storage=storage,
            chunk_size=pipeline_cfg.get("chunk_size", 10_000),
            num_workers=pipeline_cfg.get("num_workers", 8),
            flush_threshold=pipeline_cfg.get("flush_threshold", 100_000),
            output_dir=storage_cfg.get("output_dir", "/tmp/vectorforge"),
            max_text_length=pipeline_cfg.get("max_text_length"),
            dense_column=dense_column,
            sparse_column=sparse_column,
        )
    )


if __name__ == "__main__":
    main()
