import asyncio
import os

from vectorforge.sources.base import DatasetSource
from vectorforge.embedders.base import Embedder
from vectorforge.pipeline.buffer import ResultBuffer
from vectorforge.pipeline.worker import worker
from vectorforge.storage.writer import write_chunk
from vectorforge.storage.s3 import upload_to_s3


async def run(
    source: DatasetSource,
    embedder: Embedder,
    s3_bucket: str,
    s3_prefix: str,
    chunk_size: int = 10_000,
    num_workers: int = 8,
    output_dir: str = "/tmp/vectorforge",
):
    work_queue: asyncio.Queue = asyncio.Queue(maxsize=num_workers * 2)
    result_queue: asyncio.Queue = asyncio.Queue()

    async def flush(chunk_result):
        local_path = write_chunk(chunk_result, output_dir)
        await upload_to_s3(local_path, s3_bucket, s3_prefix)
        # below is technically blocking, but upload_to_s3 is the real
        # bottleneck so it shouldn't add much overhead... ideally.
        os.remove(local_path)

    buffer = ResultBuffer(flush_fn=flush)

    # start workers
    worker_tasks = [
        asyncio.create_task(worker(i, work_queue, result_queue, embedder))
        for i in range(num_workers)
    ]

    # drain result queue into buffer concurrently
    async def drain_results(total_chunks):
        for _ in range(total_chunks):
            result = await result_queue.get()
            await buffer.push(result)
        await buffer.drain()

    # run chunker (fills work queue), track how many chunks we send
    chunk_count = 0

    async def run_chunker():
        nonlocal chunk_count
        for chunk_id, records in source.get_chunks(chunk_size):
            await work_queue.put((chunk_id, records))
            chunk_count += 1
        for _ in range(num_workers):
            await work_queue.put(None)  # Sentinels

    await run_chunker()
    await asyncio.gather(*worker_tasks)
    await drain_results(chunk_count)
