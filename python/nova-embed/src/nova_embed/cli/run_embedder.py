import asyncio
import logging
import math
import os

import click

from nova_embed.config import ChunkingConfig, load_config
from nova_embed.registry import SOURCES, STORAGE
from nova_embed.sources.base import files_in_window

# Import the component packages for their registration side-effects: the
# @*.register decorators on the concrete classes populate the registries above.
import nova_embed.sources  # noqa: F401
import nova_embed.embedders  # noqa: F401
import nova_embed.storage  # noqa: F401

from nova_embed.embedders.engine import build_engine
from nova_embed.embedders.runner import run_embedder
from nova_embed.chunkers import build_chunker


def _configured_embedders(cfg) -> list[str]:
    """One line per embedder entry (no models loaded)."""
    return [
        f"{e.name}: kind={e.kind.value} type={e.type} model={e.model or '?'} "
        f"{e.input_column}[{e.modality.value}] -> {e.column}"
        for e in cfg.embedders
    ]


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
    for line in _configured_embedders(cfg):
        click.echo(f"embedder:  {line}")
    click.echo(f"chunking:  {cfg.chunking.strategy if cfg.chunking else 'passthrough'}")
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
    chunking = cfg.chunking or ChunkingConfig()

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

    # The fields being embedded must survive the source's read projection even
    # if the user's exclude_columns would drop them.
    source_dict.setdefault("required_columns", sorted(cfg.input_specs))

    source = SOURCES.build(dict(source_dict))

    engine = build_engine(cfg.embedders)

    # A splitting chunker operates on THE input column (config validation
    # guarantees there is exactly one when strategy != passthrough). Passthrough
    # needs no chunker at all: one row in, one row out.
    if chunking.splits:
        chunker = build_chunker(chunking.build_dict())
        split_column = next(iter(cfg.input_specs))
    else:
        chunker = None
        split_column = None

    storage_dict = cfg.storage.build_dict()
    storage = STORAGE.build(dict(storage_dict))

    # prefer the per-job limit (set by --num-jobs slicing); else there's no cap
    expected_total_rows = source_dict.get("limit")

    asyncio.run(
        run_embedder(
            source=source,
            engine=engine,
            storage=storage,
            chunker=chunker,
            split_column=split_column,
            chunk_size=pipeline.chunk_size,
            num_workers=pipeline.num_workers,
            flush_threshold=pipeline.flush_threshold,
            row_group_size=pipeline.row_group_size,
            output_dir=storage_dict.get("output_dir", "/tmp/nova_embed"),
            on_empty_input=pipeline.on_empty_input,
            drop_columns=pipeline.drop_columns,
            filename_prefix=filename_prefix,
            expected_total_rows=expected_total_rows,
            chunking_strategy=chunking.strategy,
        )
    )


if __name__ == "__main__":
    embed()
