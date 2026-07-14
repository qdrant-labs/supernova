import asyncio
import json
import logging
import os
import time

from dataclasses import asdict
from datetime import datetime, timezone

from tqdm import tqdm

from nova_embed.sources.base import DatasetSource, EmptyInputStats, iter_chunks
from nova_embed.chunkers import Chunker
from nova_embed.embedders.engine import EmbeddingEngine
from nova_embed.embedders.buffer import ResultBuffer
from nova_embed.embedders.worker import worker
from nova_embed.storage.base import StorageBackend
from nova_embed.storage.writer import write_batch

logger = logging.getLogger(__name__)

# Above this skip rate, "quiet" isn't good enough: a big fraction of empty
# inputs usually means the WRONG input_column, which is exactly the mistake
# the launch-time checks exist to surface.
SKIP_RATE_WARN_THRESHOLD = 0.01


async def run_embedder(
    source: DatasetSource,
    engine: EmbeddingEngine,
    storage: StorageBackend,
    chunker: Chunker | None = None,
    split_column: str | None = None,
    chunk_size: int = 10_000,
    num_workers: int = 8,
    flush_threshold: int = 100_000,
    output_dir: str = "/tmp/nova_embed",
    on_empty_input: str = "skip",
    drop_columns: list[str] | None = None,
    filename_prefix: str = "",
    expected_total_rows: int | None = None,
    row_group_size: int | None = None,
    chunking_strategy: str = "passthrough",
):
    logger.info(
        "Starting pipeline: source=%s outputs=%s storage=%s chunk_size=%d num_workers=%d flush_threshold=%d",
        source.source_name,
        [f"{s.column}<-{s.model_name}" for s in engine.output_specs],
        storage.destination,
        chunk_size,
        num_workers,
        flush_threshold,
    )
    start_time = time.time()
    total_records = 0

    await storage.ensure_ready()

    work_queue: asyncio.Queue = asyncio.Queue(maxsize=num_workers * 2)
    result_queue: asyncio.Queue = asyncio.Queue()

    batch_counter = 0
    empty_stats = EmptyInputStats()

    async def flush(records):
        nonlocal batch_counter, total_records
        local_path = write_batch(
            records,
            output_dir,
            batch_counter,
            output_specs=engine.output_specs,
            filename_prefix=filename_prefix,
            row_group_size=row_group_size,
        )
        logger.info(
            "Wrote batch %d (%d records) to %s", batch_counter, len(records), local_path
        )
        batch_counter += 1
        total_records += len(records)
        # preserve any subdir structure from filename_prefix (e.g. "rank00/") so
        # storage backends can replicate the layout remotely.
        remote_subpath = os.path.relpath(local_path, output_dir)
        # upload_file consumes local_path (cloud backends upload then delete the
        # staging copy; LocalBackend moves it into place / no-ops if it's already
        # there). Deleting it here would nuke LocalBackend's saved file.
        await storage.upload_file(local_path, remote_subpath=remote_subpath)

    buffer = ResultBuffer(flush_fn=flush, flush_threshold=flush_threshold)

    # chunker: feeds work queue, then sends sentinels to shut down workers
    async def run_chunker():
        try:
            for chunk_id, records in iter_chunks(
                source,
                input_groups=engine.input_groups,
                chunk_size=chunk_size,
                on_empty_input=on_empty_input,
                chunker=chunker,
                split_column=split_column,
                stats=empty_stats,
            ):
                await work_queue.put((chunk_id, records))
        finally:
            for _ in range(num_workers):
                await work_queue.put(None)

    # drain: pulls from result queue into buffer until all workers are done.
    # The bar counts *records* (not chunks), so tqdm shows the embedded count,
    # the %/ETA against the dataset, and a records/sec rate. `total` is the
    # per-job row count (from --num-jobs slicing or source.limit); None → a
    # count-up bar.
    progress = tqdm(
        total=expected_total_rows,
        unit=" records",
        unit_scale=True,
        desc="Embedding",
        smoothing=0.1,  # rate reflects recent throughput, not the whole run
    )

    async def drain_results():
        finished_workers = 0
        while finished_workers < num_workers:
            result = await result_queue.get()
            if result is None:  # worker finished sentinel
                finished_workers += 1
                continue
            await buffer.push(result)
            progress.update(len(result.records))
        await buffer.drain()
        progress.close()

    worker_tasks = [
        asyncio.create_task(
            worker(
                i,
                work_queue,
                result_queue,
                engine,
                drop_columns=frozenset(drop_columns or ()),
            )
        )
        for i in range(num_workers)
    ]

    await asyncio.gather(
        run_chunker(),
        *worker_tasks,
        drain_results(),
    )

    elapsed = time.time() - start_time
    logger.info(
        "Pipeline complete: %d records in %d batches, %.1fs elapsed (%.0f records/s)",
        total_records,
        batch_counter,
        elapsed,
        total_records / elapsed if elapsed > 0 else 0,
    )

    # Loud, not silent: skipped rows are counted into the manifest below, and a
    # high rate gets a warning — it usually means a wrong input_column.
    if empty_stats.rows_skipped:
        source_rows_seen = total_records + empty_stats.rows_skipped
        skip_rate = empty_stats.rows_skipped / source_rows_seen
        log = (
            logger.warning
            if skip_rate > SKIP_RATE_WARN_THRESHOLD
            else logger.info
        )
        log(
            "Skipped %d row(s) with empty input column(s) (%.2f%% of rows seen). "
            "If this is unexpected, check the configured input_column(s).",
            empty_stats.rows_skipped,
            skip_rate * 100,
        )

    manifest = {
        "source": source.source_name,
        "embedders": [asdict(spec) for spec in engine.output_specs],
        "chunk_size": chunk_size,
        "chunking_strategy": chunking_strategy,
        "num_workers": num_workers,
        "flush_threshold": flush_threshold,
        "on_empty_input": on_empty_input,
        "drop_columns": sorted(drop_columns or []),
        "total_records": total_records,
        "rows_skipped_empty_input": empty_stats.rows_skipped,
        "total_batches": batch_counter,
        "elapsed_seconds": round(elapsed, 2),
        "records_per_second": round(total_records / elapsed, 1) if elapsed > 0 else 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "destination": storage.destination,
    }
    await storage.upload_bytes(
        json.dumps(manifest, indent=2).encode(),
        f"{filename_prefix}_manifest.json" if filename_prefix else "_manifest.json",
    )
    logger.info("Uploaded manifest to %s", storage.destination)
