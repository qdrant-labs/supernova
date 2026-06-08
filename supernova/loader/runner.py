"""Async orchestration for loading pre-embedded data into vector stores."""

import asyncio
import logging
import time

from tqdm import tqdm

from .datasource.base import DataReader
from .vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


def _slice_batch(records: list[dict], batch_size: int) -> list[list[dict]]:
    """
    Slice a large prefetched chunk into upsert-sized batches.
    """
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


async def run_loader(
    reader: DataReader,
    store: VectorStore,
    batch_size: int = 1000,
    prefetch_size: int | None = None,
    concurrency: int = 8,
    manage_indexing: bool = True,
    target_wps: float = 0.0,
) -> None:
    """
    Stream pre-embedded parquet data into a vector store.

    Producer reads chunks from DuckDB and pushes upsert-sized batches into a
    bounded queue. ``concurrency`` worker tasks consume the queue and upsert
    in parallel. The bounded queue provides backpressure -- the producer
    blocks once the queue is full -- so memory is capped to roughly one
    in-flight chunk plus (queue capacity + concurrency) batches.

    ``target_wps`` (writes/sec, i.e. points/sec; 0 = unbounded) paces the
    producer: it sleeps to admit batches at the target point rate, mirroring
    ``storm``'s paced query mode. The bounded queue is the safety valve -- if the
    workers can't sustain the target the queue fills, the producer blocks on
    ``put``, and the achieved rate sags below target, which is the finding ("this
    cluster tops out below N wps"), not an error. Per worker, like storm's qps:
    fleet target = num_jobs × target_wps.

    When manage_indexing=False, skips collection creation and indexing
    lifecycle (for distributed workers where the master handles this).
    """
    if prefetch_size is None:
        prefetch_size = batch_size * 10

    pacing = f", target={target_wps:,.0f} pts/s" if target_wps > 0 else ""
    logger.info(
        f"Loading into {store.name} (prefetch={prefetch_size:,}, batch={batch_size:,}{pacing})"
    )

    dimensions = reader.get_dimensions()
    total = reader.get_total_count()
    logger.info(f"Found {total:,} records (dims={dimensions})")

    if manage_indexing:
        await store.ensure_collection(dimensions)
        logger.info("Deferring indexing for bulk load...")
        await store.defer_indexing()

    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    loaded = 0
    errors = 0
    pbar = tqdm(total=total, desc="Loading", unit=" pts")
    t0 = time.perf_counter()

    async def worker():
        nonlocal loaded, errors
        while True:
            batch = await queue.get()
            try:
                if batch is None:
                    return
                try:
                    await store.upsert_batch(batch)
                    loaded += len(batch)
                except Exception:
                    errors += len(batch)
                    logger.exception(f"Failed to upsert batch of {len(batch)} points")
                finally:
                    pbar.update(len(batch))
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

    # Points-based pacing clock: advance the virtual schedule by one batch's
    # worth of points each dispatch and sleep to it. Falling behind (delay <= 0)
    # admits the next batch immediately to catch up, so the average tracks target.
    per_point_s = 1.0 / target_wps if target_wps > 0 else 0.0
    next_dispatch = time.perf_counter()

    try:
        try:
            for chunk in reader.read_batches(prefetch_size):
                for batch in _slice_batch(chunk, batch_size):
                    if per_point_s:
                        next_dispatch += len(batch) * per_point_s
                        delay = next_dispatch - time.perf_counter()
                        if delay > 0:
                            await asyncio.sleep(delay)
                    await queue.put(batch)  # backpressure: blocks when queue is full
        finally:
            # always send sentinels so workers exit even if the producer raised;
            # without this, asyncio.gather would block on hung workers and the
            # exception would be masked by an unrelated cancellation later.
            for _ in range(concurrency):
                await queue.put(None)
        await asyncio.gather(*workers)
    finally:
        pbar.close()
        reader.close()

    elapsed = time.perf_counter() - t0
    rate = loaded / elapsed if elapsed > 0 else 0
    logger.info(
        f"Upload done: {loaded:,} loaded, {errors:,} errors "
        f"in {elapsed:.1f}s ({rate:,.0f} pts/s)"
    )

    if manage_indexing:
        await store.enable_indexing()
        t1 = time.perf_counter()
        await store.wait_for_indexing()
        index_elapsed = time.perf_counter() - t1
        logger.info(f"Indexing completed in {index_elapsed:.1f}s")

    await store.close()
