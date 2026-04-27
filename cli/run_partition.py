#!/usr/bin/env python3
"""
Run the embed pipeline with a no-op embedder.

Lets you validate sharding/partitioning logic (which rank reads which rows,
how files end up laid out in S3, whether ranks overlap) without spending
GPU time on the actual embedding model. Output parquets have every column
the real embed run produces, minus the float vectors -- so the same
`scripts/verify_no_duplicates.py` runs against the partition output to
confirm clean partitioning before committing to a real run.

Usage:
  vf partition configs/embedder/ccnews_2016.yaml
  vf partition configs/embedder/ccnews_2016.yaml --num-jobs 10 --job-rank 3
  vf partition configs/embedder/ccnews_2016.yaml --list-files       # dry-run
"""

import argparse
import asyncio
import logging
import math
import os

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
        print(f"--list-files only supports source.type=huggingface_parquet (got {source_type!r}).")
        print("For streaming HF sources, partition assignment is offset-based -- run a partition")
        print("with --num-jobs and inspect the S3 output instead.")
        return

    source = build_source(source_cfg)
    files = source.list_files()
    total_rows = sum(n for _, n in files)

    print(f"Dataset:    {source.dataset_name}")
    print(f"Filter:     path_filter={source_cfg.get('path_filter')!r}, split={source_cfg.get('split')!r}")
    print(f"Files:      {len(files)} parquet files, {total_rows:,} total rows")
    print()
    sample = files[:10]
    for path, n in sample:
        print(f"  {n:>12,}  {path}")
    if len(files) > len(sample):
        print(f"  ... and {len(files) - len(sample)} more")
    print()

    if num_jobs is None:
        print("Pass --num-jobs N to also see per-rank assignment.")
        return

    rows_per_job = math.ceil(total_rows / num_jobs)
    print(f"With --num-jobs {num_jobs}: ~{rows_per_job:,} rows/rank")

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
            print(f"  rank {rank:>3}: (empty)")
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
        print(f"  rank {rank:>3}: rows {offset:>10,}-{end - 1:>10,}  ({limit:>10,} rows)  files: {sample_touched}")


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run the embed pipeline with a no-op embedder for partition validation",
    )
    parser.add_argument("config", nargs="?", help="Path to YAML config file (same schema as `vf embed`)")
    parser.add_argument("--offset", type=int, default=None, help="Skip this many rows (for explicit slicing)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many rows")
    parser.add_argument("--num-jobs", type=int, default=None, help="Total parallel jobs (auto-computes offset/limit per rank)")
    parser.add_argument("--job-rank", type=int, default=None, help="This job's rank (0-indexed; defaults to $SKYPILOT_JOB_RANK)")
    parser.add_argument("--list-files", action="store_true",
                        help="Dry-run: list matched files + per-rank plan and exit. No S3 writes.")
    args = parser.parse_args(argv)

    config_path = args.config or os.environ.get("VF_CONFIG_PATH")
    if not config_path:
        parser.error("Provide a config path as argument or set VF_CONFIG_PATH env var")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.list_files:
        _print_list_files_plan(config, args.num_jobs)
        return

    # rank-based slicing (mirrors run_embedder.main)
    filename_prefix = ""
    if args.num_jobs is not None:
        job_rank = args.job_rank
        if job_rank is None:
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK", 0))
            logging.getLogger("vectorforge").info(
                f"Auto-detected job rank {job_rank} from SKYPILOT_JOB_RANK env var"
            )

        source_for_count = build_source(dict(config["source"]))
        dataset_total = source_for_count.get_total_rows()

        window_offset = config["source"].get("offset") or 0
        window_limit = config["source"].get("limit")
        window_size = (
            min(window_limit, dataset_total - window_offset) if window_limit
            else dataset_total - window_offset
        )

        rows_per_job = math.ceil(window_size / args.num_jobs)
        offset = window_offset + job_rank * rows_per_job
        limit = min(rows_per_job, window_size - job_rank * rows_per_job)

        logging.getLogger("vectorforge").info(
            "Job %d/%d: offset=%d limit=%d (window=[%d,%d), dataset_total=%d)",
            job_rank, args.num_jobs, offset, limit, window_offset, window_offset + window_size, dataset_total,
        )
        config["source"]["offset"] = offset
        config["source"]["limit"] = limit

        rank_width = max(2, len(str(args.num_jobs - 1)))
        shard_by_rank = bool(config.get("pipeline", {}).get("shard_by_rank"))
        separator = "/" if shard_by_rank else "_"
        filename_prefix = f"rank{job_rank:0{rank_width}d}{separator}"
    elif args.offset is not None or args.limit is not None:
        if args.offset is not None:
            config["source"]["offset"] = args.offset
        if args.limit is not None:
            config["source"]["limit"] = args.limit

    source = build_source(dict(config["source"]))
    engine = build_noop_engine(config)
    storage = build_storage(dict(config["storage"]))

    pipeline_cfg = config.get("pipeline", {})
    storage_cfg = config.get("storage", {})

    expected_total_rows = config["source"].get("limit")

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
    main()
