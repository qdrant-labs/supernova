"""Async orchestration for loading pre-embedded data into vector stores."""

import asyncio
import logging
import time

from tqdm import tqdm

from .datasource.base import DataReader
from .vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


def _slice_batch(records: list[dict], batch_size: int) -> list[list[dict]]:
    """Slice a large prefetched chunk into upsert-sized batches."""
    return [records[i:i + batch_size] for i in range(0, len(records), batch_size)]


async def run_loader(
    reader: DataReader,
    store: VectorStore,
    batch_size: int = 1000,
    prefetch_size: int | None = None,
    concurrency: int = 8,
) -> None:
    """Stream pre-embedded parquet data into a vector store.

    Reads large chunks (prefetch_size) from DuckDB to minimize remote I/O,
    then slices into upsert-sized batches and writes them concurrently.
    """
    if prefetch_size is None:
        prefetch_size = batch_size * 10

    logger.info(f"Loading into {store.name} (prefetch={prefetch_size:,}, batch={batch_size:,})")

    # Get dimensions and total count for setup
    dimension = reader.get_dimensions()
    total = reader.get_total_count()
    logger.info(f"Found {total:,} vectors (dim={dimension})")

    # Ensure collection/index exists
    await store.ensure_collection(dimension)

    # Defer indexing for fast bulk load
    logger.info("Deferring indexing for bulk load...")
    await store.defer_indexing()

    sem = asyncio.Semaphore(concurrency)
    loaded = 0
    t0 = time.perf_counter()
    errors = 0

    async def _upsert(batch: list[dict]):
        nonlocal loaded, errors
        async with sem:
            try:
                await store.upsert_batch(batch)
                loaded += len(batch)
            except Exception:
                errors += len(batch)
                logger.exception(f"Failed to upsert batch of {len(batch)} points")

    tasks: list[asyncio.Task] = []
    pbar = tqdm(total=total, desc="Loading", unit=" pts")

    try:
        # Read large chunks from DuckDB, slice into upsert batches
        for chunk in reader.read_batches(prefetch_size):
            for batch in _slice_batch(chunk, batch_size):
                task = asyncio.create_task(_upsert(batch))
                task.add_done_callback(lambda _, b=batch: pbar.update(len(b)))
                tasks.append(task)

            # Drain completed tasks between chunks to bound memory
            done = [t for t in tasks if t.done()]
            for t in done:
                tasks.remove(t)
                t.result()

        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        pbar.close()
        reader.close()

    elapsed = time.perf_counter() - t0
    rate = loaded / elapsed if elapsed > 0 else 0
    logger.info(
        f"Upload done: {loaded:,} loaded, {errors:,} errors "
        f"in {elapsed:.1f}s ({rate:,.0f} pts/s)"
    )

    # Re-enable indexing and wait for HNSW build
    await store.enable_indexing()
    t1 = time.perf_counter()
    await store.wait_for_indexing()
    index_elapsed = time.perf_counter() - t1
    logger.info(f"Indexing completed in {index_elapsed:.1f}s")

    await store.close()
