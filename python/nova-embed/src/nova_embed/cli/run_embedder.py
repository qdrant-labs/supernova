import asyncio
import logging
import math
import os

import click

from nova_embed.config import load_config
from nova_embed.registry import (
    DENSE_EMBEDDERS,
    MULTIVECTOR_EMBEDDERS,
    SOURCES,
    SPARSE_EMBEDDERS,
    STORAGE,
)
from nova_embed.sources.base import files_in_window

# Import the component packages for their registration side-effects: the
# @*.register decorators on the concrete classes populate the registries above.
import nova_embed.sources  # noqa: F401
import nova_embed.embedders  # noqa: F401
import nova_embed.storage  # noqa: F401

from nova_embed.embedders.engine import EmbeddingEngine
from nova_embed.embedders.hybrid import SentenceTransformerHybridEmbedder
from nova_embed.embedders.runner import run_embedder
from nova_embed.chunkers import build_chunker


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
        MULTIVECTOR_EMBEDDERS.build(multivector_cfg) if multivector_cfg else None
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
    dense = DENSE_EMBEDDERS.build(dense_cfg) if dense_cfg else None
    sparse = SPARSE_EMBEDDERS.build(sparse_cfg) if sparse_cfg else None
    return EmbeddingEngine(
        dense=dense,
        sparse=sparse,
        multivector=multivector,
        multivector_pooling=pooling_type,
        multivector_pooling_normalize=pooling_normalize,
    )


def _configured_embedders(cfg) -> list[str]:
    """`kind=model` for each embedder set in the config (no models loaded)."""
    out = []
    for kind in ("dense", "sparse", "multivector"):
        section = getattr(cfg, f"{kind}_embedder", None)
        if section:
            d = section.build_dict()
            out.append(f"{kind}={d.get('model') or d.get('type') or '?'}")
    return out


def _print_dry_run(cfg, config_path: str, source_dict: dict, num_jobs: int | None) -> None:
    """
    Inspect the source and print how the dataset partitions across `num_jobs`
    workers — rows per rank, and (for file-based sources) how many parquet files
    each rank reads. Footers only: no data download, no embedding model loaded.
    """
    click.echo("=" * 70)
    click.echo("nova-embed DRY RUN")
    click.echo("=" * 70)
    click.echo(f"config:    {config_path}")
    click.echo(f"source:    {cfg.source.type}")
    click.echo(f"embedders: {', '.join(_configured_embedders(cfg)) or '(none configured!)'}")
    click.echo(f"chunking:  {cfg.chunking.type if cfg.chunking else 'passthrough'}")
    click.echo(f"storage:   {cfg.storage.type}")

    source = SOURCES.build(dict(source_dict))
    total = source.get_total_rows()
    list_files = getattr(source, "list_files", None)
    files = list_files() if callable(list_files) else None

    click.echo("-" * 70)
    if files is not None:
        click.echo(f"dataset:   {total:,} rows across {len(files):,} parquet files")
    else:
        click.echo(f"dataset:   {total:,} rows")

    jobs = num_jobs or 1
    rows_per_job = math.ceil(total / jobs) if total else 0
    click.echo(f"partition: {jobs} job(s), ~{rows_per_job:,} rows/job")
    click.echo("-" * 70)

    width = max(1, len(str(jobs - 1)))
    empty = 0
    for rank in range(jobs):
        offset = rank * rows_per_job
        limit = max(0, min(rows_per_job, total - offset))
        empty += limit == 0
        line = f"  rank {rank:>{width}}: rows [{offset:>12,} .. {offset + limit:>12,})  {limit:>12,} rows"
        if files is not None:
            line += f"  ·  {len(files_in_window(files, offset, limit)) if limit else 0} files"
        click.echo(line)

    if empty:
        click.echo(f"\n  ⚠  {empty} job(s) receive 0 rows — num_jobs exceeds the data; reduce it.")
    if files is not None and num_jobs and num_jobs > len(files):
        click.echo(
            f"\n  ⚠  num_jobs ({num_jobs}) > file count ({len(files)}): ranks splitting the "
            "same file each download that whole file (row windows don't split files)."
        )
    click.echo("=" * 70)


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
    """
    Run a nova_embed embedding pipeline.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nova_embed").setLevel(logging.INFO)

    config_path = config or os.environ.get("NOVA_CONFIG_PATH")
    if not config_path:
        raise click.UsageError(
            "Provide a config path as argument or set NOVA_CONFIG_PATH env var"
        )

    cfg = load_config(config_path)
    pipeline = cfg.pipeline

    source_dict = cfg.source.build_dict()
    if "offset" in source_dict or "limit" in source_dict:
        raise click.UsageError(
            "source.offset / source.limit are not supported in YAML. "
            "Use --num-jobs / --job-rank for distributed slicing."
        )

    # Dry run short-circuits BEFORE building the engine — we must not download or
    # load the embedding model just to print a plan. It inspects the source
    # (parquet footers only) to show how the dataset partitions across workers.
    if dry_run:
        _print_dry_run(cfg, config_path, source_dict, num_jobs)
        return

    filename_prefix = ""
    if num_jobs is not None:
        if job_rank is None:
            # SkyPilot pools set these env vars automatically
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))
            logging.getLogger("nova_embed").info(
                f"Auto-detected job rank {job_rank} from SKYPILOT_JOB_RANK env var"
            )

        # build source to query total rows (source-agnostic)
        source_for_count = SOURCES.build(dict(source_dict))
        dataset_total = source_for_count.get_total_rows()

        rows_per_job = math.ceil(dataset_total / num_jobs)
        slice_offset = job_rank * rows_per_job
        slice_limit = min(rows_per_job, dataset_total - slice_offset)

        logging.getLogger("nova_embed").info(
            "Job %d/%d: offset=%d limit=%d (dataset_total=%d)",
            job_rank + 1,
            num_jobs,
            slice_offset,
            slice_limit,
            dataset_total,
        )
        source_dict["offset"] = slice_offset
        source_dict["limit"] = slice_limit

        rank_width = max(2, len(str(num_jobs - 1)))
        # shard_by_rank=true  -> "rank00/batch_*.parquet" (subdir per rank)
        # shard_by_rank=false -> "rank00_batch_*.parquet" (flat)
        separator = "/" if pipeline.shard_by_rank else "_"
        filename_prefix = f"rank{job_rank:0{rank_width}d}{separator}"

    # Carry source provenance (source_file_name + source_row_number) into the
    # output when enabled. Injected only when on, so sources that don't support
    # it aren't forced to accept the kwarg.
    if pipeline.include_source_provenance:
        source_dict["include_provenance"] = True

    source = SOURCES.build(dict(source_dict))

    # build_engine still takes a dict keyed by *_embedder; feed it the validated
    # sections.
    engine_cfg: dict = {}
    if cfg.dense_embedder:
        engine_cfg["dense_embedder"] = cfg.dense_embedder.build_dict()
    if cfg.sparse_embedder:
        engine_cfg["sparse_embedder"] = cfg.sparse_embedder.build_dict()
    if cfg.multivector_embedder:
        engine_cfg["multivector_embedder"] = cfg.multivector_embedder.build_dict()
    engine = build_engine(engine_cfg)

    chunker = build_chunker(cfg.chunking.build_dict() if cfg.chunking else None)

    storage_dict = cfg.storage.build_dict()
    storage = STORAGE.build(dict(storage_dict))

    dense_column = pipeline.dense_embedding_column if engine.has_dense else None
    sparse_column = pipeline.sparse_embedding_column if engine.has_sparse else None
    multivector_column = (
        pipeline.multivector_embedding_column if engine.has_multivector else None
    )

    # pooling (nested under multivector_embedder) can override the dense column name
    mv_dict = cfg.multivector_embedder.build_dict() if cfg.multivector_embedder else {}
    pooling_cfg = mv_dict.get("pooling") or {}
    if pooling_cfg.get("pooled_column_name"):
        dense_column = pooling_cfg["pooled_column_name"]

    # prefer the per-job limit (set by --num-jobs slicing); else there's no cap
    expected_total_rows = source_dict.get("limit")

    asyncio.run(
        run_embedder(
            source=source,
            engine=engine,
            storage=storage,
            chunker=chunker,
            chunk_size=pipeline.chunk_size,
            num_workers=pipeline.num_workers,
            flush_threshold=pipeline.flush_threshold,
            row_group_size=pipeline.row_group_size,
            output_dir=storage_dict.get("output_dir", "/tmp/nova_embed"),
            max_text_length=pipeline.max_text_length,
            dense_column=dense_column,
            sparse_column=sparse_column,
            multivector_column=multivector_column,
            rendered_text_column=pipeline.rendered_text_column,
            filename_prefix=filename_prefix,
            expected_total_rows=expected_total_rows,
        )
    )


if __name__ == "__main__":
    embed()
