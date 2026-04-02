import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

from tqdm import tqdm

from vectorforge.sources.base import DatasetSource
from vectorforge.embedders.base import Embedder
from vectorforge.pipeline.buffer import ResultBuffer
from vectorforge.pipeline.worker import worker
from vectorforge.storage.base import StorageBackend
from vectorforge.storage.writer import write_batch

logger = logging.getLogger(__name__)


async def run(
    source: DatasetSource,
    embedder: Embedder,
    storage: StorageBackend,
    chunk_size: int = 10_000,
    num_workers: int = 8,
    flush_threshold: int = 100_000,
    output_dir: str = "/tmp/vectorforge",
):
    logger.info(
        "Starting pipeline: source=%s embedder=%s storage=%s chunk_size=%d num_workers=%d flush_threshold=%d",
        source.source_name, embedder.model_name, storage.destination, chunk_size, num_workers, flush_threshold,
    )
    start_time = time.time()
    total_records = 0

    await storage.ensure_ready()

    work_queue: asyncio.Queue = asyncio.Queue(maxsize=num_workers * 2)
    result_queue: asyncio.Queue = asyncio.Queue()

    batch_counter = 0

    async def flush(records):
        nonlocal batch_counter, total_records
        local_path = write_batch(records, output_dir, batch_counter)
        logger.info("Wrote batch %d (%d records) to %s", batch_counter, len(records), local_path)
        batch_counter += 1
        total_records += len(records)
        await storage.upload_file(local_path)
        os.remove(local_path)

    buffer = ResultBuffer(flush_fn=flush, flush_threshold=flush_threshold)

    # chunker: feeds work queue, then sends sentinels to shut down workers
    async def run_chunker():
        for chunk_id, records in source.get_chunks(embedder, chunk_size):
            await work_queue.put((chunk_id, records))
        logger.info("Chunker finished, sending stop signals to %d workers", num_workers)
        for _ in range(num_workers):
            await work_queue.put(None)

    # drain: pulls from result queue into buffer until all workers are done
    progress = tqdm(unit=" chunks", desc="Embedding")
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
            progress.set_postfix(records=f"{embedded_records:,}")
        await buffer.drain()
        progress.close()

    worker_tasks = [
        asyncio.create_task(worker(i, work_queue, result_queue, embedder))
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
        total_records, batch_counter, elapsed, total_records / elapsed if elapsed > 0 else 0,
    )

    manifest = {
        "source": source.source_name,
        "embedder": embedder.model_name,
        "dimensions": embedder.dimensions,
        "chunk_size": chunk_size,
        "max_tokens": embedder.max_tokens,
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
        "_manifest.json",
    )
    logger.info("Uploaded manifest to %s", storage.destination)