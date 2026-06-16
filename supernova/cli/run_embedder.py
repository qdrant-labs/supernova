import asyncio
import logging
import math
import os

import click
import yaml

from supernova.sources.huggingface import HuggingFaceSource
from supernova.embedders.dense.openai import OpenAIEmbedder
from supernova.embedders.dense.sentence_transformer import (
    SentenceTransformerDenseEmbedder,
)
from supernova.embedders.sparse.sentence_transformer import (
    SentenceTransformerSparseEmbedder,
)
from supernova.embedders.sparse.fastembed import FastEmbedSparseEmbedder
from supernova.embedders.multivector.bge_m3 import BGEM3MultiVectorEmbedder
from supernova.embedders.hybrid import SentenceTransformerHybridEmbedder
from supernova.embedders.engine import EmbeddingEngine
from supernova.chunkers import build_chunker
from supernova.storage.s3 import S3Backend
from supernova.storage.huggingface import HuggingFaceBackend
from supernova.storage.local import LocalBackend
from supernova.embedders.runner import run_embedder

# available sources, embedders, and storage backends. Used to construct from config.
# mapping from string identifiers in config → actual classes. Factored out to avoid circular imports and keep main() clean.
SOURCE_REGISTRY = {
    "huggingface": HuggingFaceSource,
    # Legacy alias: pre-rename the parquet streamer was registered as
    # "huggingface_parquet". Existing configs keep working unchanged.
    "huggingface_parquet": HuggingFaceSource,
}

DENSE_EMBEDDER_REGISTRY = {
    "openai": OpenAIEmbedder,
    "sentence_transformer": SentenceTransformerDenseEmbedder,
}

SPARSE_EMBEDDER_REGISTRY = {
    "sentence_transformer": SentenceTransformerSparseEmbedder,
    "fastembed": FastEmbedSparseEmbedder,
}

MULTIVECTOR_EMBEDDER_REGISTRY = {
    "bge_m3": BGEM3MultiVectorEmbedder,
}


def build_source(cfg: dict):
    source_type = cfg.pop("type")
    cls = SOURCE_REGISTRY[source_type]
    return cls(**cfg)


def build_dense_embedder(cfg: dict):
    embedder_type = cfg.pop("type")
    cls = DENSE_EMBEDDER_REGISTRY.get(embedder_type)
    if cls is None:
        raise ValueError(
            f"Unknown dense embedder type: {embedder_type}. Available: {list(DENSE_EMBEDDER_REGISTRY)}"
        )
    return cls(**cfg)


def build_sparse_embedder(cfg: dict):
    embedder_type = cfg.pop("type")
    cls = SPARSE_EMBEDDER_REGISTRY.get(embedder_type)
    if cls is None:
        raise ValueError(
            f"Unknown sparse embedder type: {embedder_type}. Available: {list(SPARSE_EMBEDDER_REGISTRY)}"
        )
    return cls(**cfg)


def build_multivector_embedder(cfg: dict):
    embedder_type = cfg.pop("type")
    cls = MULTIVECTOR_EMBEDDER_REGISTRY.get(embedder_type)
    if cls is None:
        raise ValueError(
            f"Unknown multivector embedder type: {embedder_type}. Available: {list(MULTIVECTOR_EMBEDDER_REGISTRY)}"
        )
    return cls(**cfg)


def _can_hybrid(dense_cfg: dict, sparse_cfg: dict) -> bool:
    """
    Check if dense and sparse configs point to the same sentence_transformer model.
    """
    return dense_cfg.get("type") == sparse_cfg.get(
        "type"
    ) == "sentence_transformer" and dense_cfg.get("model") == sparse_cfg.get("model")


def build_engine(config: dict) -> EmbeddingEngine:
    """
    Build an EmbeddingEngine from config.

    Supports any combination of:
      - dense_embedder
      - sparse_embedder
      - multivector_embedder
      - pooling (derive a dense column from the multivector output)

    dense + sparse with the same sentence_transformer model are auto-combined
    into a single hybrid forward pass. Multivector is always built separately.
    """
    dense_cfg = dict(config.get("dense_embedder") or {})
    sparse_cfg = dict(config.get("sparse_embedder") or {})
    multivector_cfg = dict(config.get("multivector_embedder") or {})

    if not dense_cfg and not sparse_cfg and not multivector_cfg:
        raise ValueError(
            "Config must specify at least one of: dense_embedder, sparse_embedder, multivector_embedder"
        )

    # pooling lives inside multivector_embedder (it only applies in that context).
    # pop it off so it isn't passed to the embedder constructor as an unknown kwarg.
    pooling_cfg = dict(multivector_cfg.pop("pooling", None) or {})
    if config.get("pooling"):
        import warnings

        warnings.warn(
            "Top-level 'pooling:' key is ignored. Nest it under 'multivector_embedder:' instead.",
            stacklevel=2,
        )

    multivector = (
        build_multivector_embedder(multivector_cfg) if multivector_cfg else None
    )

    pooling_type = pooling_cfg.get("type") if pooling_cfg else None
    pooling_normalize = pooling_cfg.get("normalize", True) if pooling_cfg else True

    # detect hybrid case: same model for both --> optimize for a single forward pass
    if dense_cfg and sparse_cfg and _can_hybrid(dense_cfg, sparse_cfg):
        hybrid_cfg = dict(dense_cfg)
        hybrid_cfg.pop("type")  # remove the type
        hybrid = SentenceTransformerHybridEmbedder(**hybrid_cfg)
        return EmbeddingEngine(
            hybrid=hybrid,
            multivector=multivector,
            multivector_pooling=pooling_type,
            multivector_pooling_normalize=pooling_normalize,
        )

    # build separately (two distinct models, no optimization is possible)
    dense = build_dense_embedder(dense_cfg) if dense_cfg else None
    sparse = build_sparse_embedder(sparse_cfg) if sparse_cfg else None
    return EmbeddingEngine(
        dense=dense,
        sparse=sparse,
        multivector=multivector,
        multivector_pooling=pooling_type,
        multivector_pooling_normalize=pooling_normalize,
    )


def build_storage(cfg: dict):
    storage_type = cfg.pop("type", "s3")
    if storage_type == "s3":
        return S3Backend(
            bucket=cfg["bucket"],
            prefix=cfg["prefix"],
        )
    elif storage_type == "hf":
        bucket_id = cfg.get("bucket_id") or cfg.get("repo_id")
        if not bucket_id:
            raise ValueError("storage.type='hf' requires 'bucket_id' (HF bucket like 'owner/name')")
        return HuggingFaceBackend(
            bucket_id=bucket_id,
            prefix=cfg.get("prefix", ""),
            token=cfg.get("token"),
            private=cfg.get("private", True),
        )
    elif storage_type == "local":
        return LocalBackend(
            output_dir=cfg.get("output_dir", "/tmp/supernova"),
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


@click.command(name="embed", help="Embed a dataset locally.")
@click.argument("config", required=False)
@click.option(
    "--num-jobs",
    type=int,
    default=None,
    help="Total number of parallel jobs (auto-computes per-rank slice from dataset size).",
)
@click.option(
    "--job-rank",
    type=int,
    default=None,
    help="This job's rank (0-indexed, used with --num-jobs).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate configs and print plan, don't run the pipeline.",
)
def embed(config, num_jobs, job_rank, dry_run):
    """Run a supernova embedding pipeline."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("supernova").setLevel(logging.INFO)

    config_path = config or os.environ.get("NOVA_CONFIG_PATH")
    if not config_path:
        raise click.UsageError(
            "Provide a config path as argument or set NOVA_CONFIG_PATH env var"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "offset" in cfg.get("source", {}) or "limit" in cfg.get("source", {}):
        raise click.UsageError(
            "source.offset / source.limit are no longer supported in YAML. "
            "Use --num-jobs / --job-rank for distributed slicing."
        )

    filename_prefix = ""
    if num_jobs is not None:
        if job_rank is None:
            # SkyPilot pools set these env vars automatically
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))
            logging.getLogger("supernova").info(
                f"Auto-detected job rank {job_rank} from SKYPILOT_JOB_RANK env var"
            )

        # build source to query total rows (source-agnostic)
        source_for_count = build_source(dict(cfg["source"]))
        dataset_total = source_for_count.get_total_rows()

        rows_per_job = math.ceil(dataset_total / num_jobs)
        slice_offset = job_rank * rows_per_job
        slice_limit = min(rows_per_job, dataset_total - slice_offset)

        logging.getLogger("supernova").info(
            "Job %d/%d: offset=%d limit=%d (dataset_total=%d)",
            job_rank + 1,
            num_jobs,
            slice_offset,
            slice_limit,
            dataset_total,
        )
        cfg["source"]["offset"] = slice_offset
        cfg["source"]["limit"] = slice_limit

        rank_width = max(2, len(str(num_jobs - 1)))
        shard_by_rank = bool(cfg.get("pipeline", {}).get("shard_by_rank"))
        # shard_by_rank=true  -> "rank00/batch_*.parquet" (50 subdirs, ~N files each)
        # shard_by_rank=false -> "rank00_batch_*.parquet" (flat, all in one dir)
        separator = "/" if shard_by_rank else "_"
        filename_prefix = f"rank{job_rank:0{rank_width}d}{separator}"

    source = build_source(dict(cfg["source"]))
    engine = build_engine(cfg)
    chunker = build_chunker(cfg.get("chunking"))
    storage = build_storage(dict(cfg["storage"]))

    pipeline_cfg = cfg.get("pipeline", {})
    storage_cfg = cfg.get("storage", {})

    dense_column = (
        pipeline_cfg.get("dense_embedding_column", "dense_embedding")
        if engine.has_dense
        else None
    )
    sparse_column = (
        pipeline_cfg.get("sparse_embedding_column", "sparse_embedding")
        if engine.has_sparse
        else None
    )
    multivector_column = (
        pipeline_cfg.get("multivector_embedding_column", "multivector_embedding")
        if engine.has_multivector
        else None
    )

    # if pooling is configured (nested under multivector_embedder), its pooled_column_name
    # overrides the default dense column
    pooling_cfg = (cfg.get("multivector_embedder") or {}).get("pooling") or {}
    if pooling_cfg and pooling_cfg.get("pooled_column_name"):
        dense_column = pooling_cfg["pooled_column_name"]

    # expected_total_rows drives the progress bar's "X/Y chunks + pct" display.
    # prefer the per-job limit (set by --num-jobs slicing); fall back to source-level limit.
    expected_total_rows = cfg["source"].get("limit")

    if dry_run:
        click.echo("=" * 60)
        click.echo("supernova embedding pipeline DRY RUN")
        click.echo("=" * 60)
        click.echo(f"Config: {config_path}")
        click.echo(f"Source: {cfg['source']['type']}")
        click.echo(f"Engine: {', '.join(k for k in ['dense', 'sparse', 'multivector'] if getattr(engine, 'has_' + k))} embedding")
        click.echo(f"Chunking: {chunker.__class__.__name__}")
        click.echo(f"Storage: {cfg['storage']['type']}")
        click.echo(f"Filename prefix: '{filename_prefix}'")
        if num_jobs:
            click.echo(f"Distributed slicing: job_rank={job_rank} / num_jobs={num_jobs}")
        click.echo("=" * 60)
        return

    asyncio.run(
        run_embedder(
            source=source,
            engine=engine,
            storage=storage,
            chunker=chunker,
            chunk_size=pipeline_cfg.get("chunk_size", 10_000),
            num_workers=pipeline_cfg.get("num_workers", 8),
            flush_threshold=pipeline_cfg.get("flush_threshold", 100_000),
            row_group_size=pipeline_cfg.get("row_group_size"),
            output_dir=storage_cfg.get("output_dir", "/tmp/supernova"),
            max_text_length=pipeline_cfg.get("max_text_length"),
            dense_column=dense_column,
            sparse_column=sparse_column,
            multivector_column=multivector_column,
            rendered_text_column=pipeline_cfg.get("rendered_text_column", "text"),
            filename_prefix=filename_prefix,
            expected_total_rows=expected_total_rows,
        )
    )


if __name__ == "__main__":
    embed()
