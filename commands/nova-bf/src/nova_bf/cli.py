"""`nova bf <compute|merge>` — exec'd by the `nova` dispatcher as `nova-bf`."""

from __future__ import annotations

import logging

import click

from nova_bf.config import load_config


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nova_bf").setLevel(logging.INFO)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Brute-force exact nearest-neighbor ground truth."""


@main.command()
@click.argument("config")
@click.option("--num-jobs", type=int, default=None, help="Total workers (enables distributed slicing).")
@click.option("--job-rank", type=int, default=None, help="This worker's rank; defaults to $SKYPILOT_JOB_RANK.")
@click.option("--io-workers", type=int, default=None, help="Override params.io_workers — concurrent corpus-file reader threads (for sweeping/tuning).")
@click.option("--io-thread-count", type=int, default=None, help="Override params.io_thread_count — pyarrow's global IO pool (true S3 fetch concurrency).")
@click.option("--max-files", type=int, default=None, help="Read only the first N corpus files of this slice. Benchmarking aid; output is PARTIAL.")
def compute(
    config: str,
    num_jobs: int | None,
    job_rank: int | None,
    io_workers: int | None,
    io_thread_count: int | None,
    max_files: int | None,
) -> None:
    """Search the corpus and write per-query top-K (one worker's slice)."""
    _setup_logging()
    from nova_bf.compute import run_compute

    run_compute(
        load_config(config),
        num_jobs=num_jobs,
        job_rank=job_rank,
        io_workers=io_workers,
        io_thread_count=io_thread_count,
        max_files=max_files,
    )


@main.command()
@click.argument("config")
def merge(config: str) -> None:
    """Merge per-rank partial results into each search's own top-K parquet."""
    _setup_logging()
    from nova_bf.merge import run_merge

    run_merge(load_config(config))


if __name__ == "__main__":
    main()
