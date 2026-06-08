import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

from tqdm import tqdm

from supernova.sources.base import DatasetSource
from supernova.chunkers import Chunker
from supernova.embedders.engine import EmbeddingEngine
from supernova.pipeline.buffer import ResultBuffer
from supernova.pipeline.worker import worker
from supernova.storage.base import StorageBackend
from supernova.storage.writer import write_batch

logger = logging.getLogger(__name__)


async def run(
    source: DatasetSource,
    engine: EmbeddingEngine,
    storage: StorageBackend,
    chunker: Chunker,
    chunk_size: int = 10_000,
    num_workers: int = 8,
    flush_threshold: int = 100_000,
    output_dir: str = "/tmp/supernova",
    max_text_length: int | None = None,
    dense_column: str | None = "dense_embedding",
    sparse_column: str | None = None,
    multivector_column: str | None = None,
    rendered_text_column: str = "text",
    filename_prefix: str = "",
    expected_total_rows: int | None = None,
    row_group_size: int | None = None,
):
    logger.info(
        "Starting pipeline: source=%s engine=%s storage=%s chunk_size=%d num_workers=%d flush_threshold=%d",
        source.source_name,
        engine.model_name,
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

    async def flush(records):
        nonlocal batch_counter, total_records
        local_path = write_batch(
            records,
            output_dir,
            batch_counter,
            dense_column=dense_column,
            sparse_column=sparse_column,
            multivector_column=multivector_column,
            rendered_text_column=rendered_text_column,
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
            for chunk_id, records in source.get_chunks(
                chunker, chunk_size, max_text_length
            ):
                await work_queue.put((chunk_id, records))
        finally:
            # always send sentinels so workers exit even if the source raised;
            # otherwise drain_results hangs waiting on `finished_workers` and
            # the original exception gets masked.
            logger.info(
                "Chunker finished, sending stop signals to %d workers", num_workers
            )
            for _ in range(num_workers):
                await work_queue.put(None)

    # drain: pulls from result queue into buffer until all workers are done
    expected_chunks = None
    if expected_total_rows is not None and chunk_size > 0:
        expected_chunks = (expected_total_rows + chunk_size - 1) // chunk_size
    progress = tqdm(unit=" chunks", desc="Embedding", total=expected_chunks)
    embedded_records = 0

    async def drain_results():
        nonlocal embedded_records
        finished_workers = 0
        while finished_workers < num_workers:
            result = await result_queue.get()
            if result is None:  # worker finished sentinel
                finished_workers += 1
                continue
            await buffer.push(result)
            embedded_records += len(result.records)
            progress.update(1)
            postfix = {"records": f"{embedded_records:,}"}
            if expected_total_rows:
                postfix["pct"] = f"{100 * embedded_records / expected_total_rows:.1f}%"
            progress.set_postfix(**postfix)
        await buffer.drain()
        progress.close()

    worker_tasks = [
        asyncio.create_task(worker(i, work_queue, result_queue, engine))
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

    manifest = {
        "source": source.source_name,
        "dense_embedder": engine.dense_model_name,
        "sparse_embedder": engine.sparse_model_name,
        "multivector_embedder": engine.multivector_model_name,
        "dimensions": engine.dimensions,
        "dense_column": dense_column,
        "sparse_column": sparse_column,
        "multivector_column": multivector_column,
        "chunk_size": chunk_size,
        "chunking_strategy": chunker.__class__.__name__,
        "max_tokens": engine.max_tokens,
        "num_workers": num_workers,
        "flush_threshold": flush_threshold,
        "total_records": total_records,
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
