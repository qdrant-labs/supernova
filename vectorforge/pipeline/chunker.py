import asyncio

from vectorforge.sources.base import DatasetSource


async def chunker(source: DatasetSource, work_queue: asyncio.Queue, chunk_size: int):
    """Push chunks from the source onto the work queue."""
    for chunk_id, records in source.get_chunks(chunk_size):
        await work_queue.put((chunk_id, records))
