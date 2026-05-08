#!/usr/bin/env python3
"""
Run the embed pipeline with a no-op embedder.

Lets you validate sharding/partitioning logic (which rank reads which rows,
how files end up laid out in S3, whether ranks overlap) without spending
GPU time on the actual embedding model. Output parquets have every column
the real embed run produces, minus the float vectors -- so the same
`scripts/verify_no_duplicates.py` runs against the partition output to
confirm clean partitioning before committing to a real run.
"""

import asyncio
import logging
import math
import os

import click
import yaml

from cli.run_embedder import build_source, build_storage
from vectorforge.embedders.dense.noop import NoopDenseEmbedder
from vectorforge.embedders.engine import EmbeddingEngine
from vectorforge.pipeline.runner import run


def build_noop_engine(config: dict) -> EmbeddingEngine:
    """
    Build an engine that runs everything except the model forward pass.

    Reads `partition.chunk_chars` (optional) so chunking is approximated;
    otherwise treats each source row as one record. Per-rank file count won't
    match the real embed exactly, but row-distribution / non-overlap will.
    """
    partition_cfg = config.get("partition") or {}
    chunk_chars = partition_cfg.get("chunk_chars")
    max_tokens = partition_cfg.get("max_tokens", 1_000_000)

    # Surface the original embedder model for manifest readability.
    real_model = (
        (config.get("dense_embedder") or {}).get("model")
        or (config.get("multivector_embedder") or {}).get("model")
        or (config.get("sparse_embedder") or {}).get("model")
    )

    return EmbeddingEngine(
        dense=NoopDenseEmbedder(
            max_tokens=max_tokens,
            chunk_chars=chunk_chars,
            model=real_model,
        ),
    )


def _print_list_files_plan(config: dict, num_jobs: int | None) -> None:
    """Dry-run: list parquet files matched by the source, plus per-rank plan."""
    source_cfg = dict(config["source"])
    source_type = source_cfg.get("type")

    if source_type != "huggingface_parquet":
        # streaming HF source has no file-listing concept
        click.echo(f"--list-files only supports source.type=huggingface_parquet (got {source_type!r}).")
        click.echo("For streaming HF sources, partition assignment is offset-based -- run a partition")
        click.echo("with --num-jobs and inspect the S3 output instead.")
        return

    source = build_source(source_cfg)
    files = source.list_files()
    total_rows = sum(n for _, n in files)

    click.echo(f"Dataset:    {source.dataset_name}")
    click.echo(f"Filter:     path_filter={source_cfg.get('path_filter')!r}, split={source_cfg.get('split')!r}")
    click.echo(f"Files:      {len(files)} parquet files, {total_rows:,} total rows")
    click.echo("")
    sample = files[:10]
    for path, n in sample:
        click.echo(f"  {n:>12,}  {path}")
    if len(files) > len(sample):
        click.echo(f"  ... and {len(files) - len(sample)} more")
    click.echo("")

    if num_jobs is None:
        click.echo("Pass --num-jobs N to also see per-rank assignment.")
        return

    rows_per_job = math.ceil(total_rows / num_jobs)
    click.echo(f"With --num-jobs {num_jobs}: ~{rows_per_job:,} rows/rank")

    # cumulative file boundaries: (path, file_start, file_end) over the global row index
    cumulative = 0
    file_ranges: list[tuple[str, int, int]] = []
    for p, n in files:
        file_ranges.append((p, cumulative, cumulative + n))
        cumulative += n

    for rank in range(num_jobs):
        offset = rank * rows_per_job
        limit = min(rows_per_job, total_rows - offset)
        if limit <= 0:
            click.echo(f"  rank {rank:>3}: (empty)")
            continue
        end = offset + limit
        # files this rank touches
        touched = [
            os.path.basename(p)
            for p, fs, fe in file_ranges
            if fe > offset and fs < end
        ]
        sample_touched = ", ".join(touched[:3])
        if len(touched) > 3:
            sample_touched += f" (+{len(touched) - 3} more)"
        click.echo(f"  rank {rank:>3}: rows {offset:>10,}-{end - 1:>10,}  ({limit:>10,} rows)  files: {sample_touched}")


@click.command(name="partition",
               help="Run pipeline with no-op embedder (validate sharding without GPU).")
@click.argument("config", required=False)
@click.option("--offset", type=int, default=None,
              help="Skip this many rows (for explicit slicing).")
@click.option("--limit", type=int, default=None,
              help="Process at most this many rows.")
@click.option("--num-jobs", type=int, default=None,
              help="Total parallel jobs (auto-computes offset/limit per rank).")
@click.option("--job-rank", type=int, default=None,
              help="This job's rank (0-indexed; defaults to $SKYPILOT_JOB_RANK).")
@click.option("--list-files", "list_files_flag", is_flag=True,
              help="Dry-run: list matched files + per-rank plan and exit. No S3 writes.")
def partition(config, offset, limit, num_jobs, job_rank, list_files_flag):
    """Run the embed pipeline with the no-op embedder for partition validation."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    config_path = config or os.environ.get("VF_CONFIG_PATH")
    if not config_path:
        raise click.UsageError("Provide a config path as argument or set VF_CONFIG_PATH env var")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if list_files_flag:
        _print_list_files_plan(cfg, num_jobs)
        return

    # rank-based slicing (mirrors run_embedder.embed)
    filename_prefix = ""
    if num_jobs is not None:
        if job_rank is None:
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))
            logging.getLogger("vectorforge").info(
                f"Auto-detected job rank {job_rank} from SKYPILOT_JOB_RANK env var"
            )

        source_for_count = build_source(dict(cfg["source"]))
        dataset_total = source_for_count.get_total_rows()

        window_offset = cfg["source"].get("offset") or 0
        window_limit = cfg["source"].get("limit")
        window_size = (
            min(window_limit, dataset_total - window_offset) if window_limit
            else dataset_total - window_offset
        )

        rows_per_job = math.ceil(window_size / num_jobs)
        slice_offset = window_offset + job_rank * rows_per_job
        slice_limit = min(rows_per_job, window_size - job_rank * rows_per_job)

        logging.getLogger("vectorforge").info(
            "Job %d/%d: offset=%d limit=%d (window=[%d,%d), dataset_total=%d)",
            job_rank, num_jobs, slice_offset, slice_limit, window_offset, window_offset + window_size, dataset_total,
        )
        cfg["source"]["offset"] = slice_offset
        cfg["source"]["limit"] = slice_limit

        rank_width = max(2, len(str(num_jobs - 1)))
        shard_by_rank = bool(cfg.get("pipeline", {}).get("shard_by_rank"))
        separator = "/" if shard_by_rank else "_"
        filename_prefix = f"rank{job_rank:0{rank_width}d}{separator}"
    elif offset is not None or limit is not None:
        if offset is not None:
            cfg["source"]["offset"] = offset
        if limit is not None:
            cfg["source"]["limit"] = limit

    source = build_source(dict(cfg["source"]))
    engine = build_noop_engine(cfg)
    storage = build_storage(dict(cfg["storage"]))

    pipeline_cfg = cfg.get("pipeline", {})
    storage_cfg = cfg.get("storage", {})

    expected_total_rows = cfg["source"].get("limit")

    asyncio.run(
        run(
            source=source,
            engine=engine,
            storage=storage,
            chunk_size=pipeline_cfg.get("chunk_size", 10_000),
            num_workers=pipeline_cfg.get("num_workers", 8),
            flush_threshold=pipeline_cfg.get("flush_threshold", 100_000),
            row_group_size=pipeline_cfg.get("row_group_size"),
            output_dir=storage_cfg.get("output_dir", "/tmp/vectorforge"),
            max_text_length=pipeline_cfg.get("max_text_length"),
            # Skip all embedding columns -- writer omits them when None.
            dense_column=None,
            sparse_column=None,
            multivector_column=None,
            rendered_text_column=pipeline_cfg.get("rendered_text_column", "text"),
            filename_prefix=filename_prefix,
            expected_total_rows=expected_total_rows,
        )
    )


if __name__ == "__main__":
    partition()
