import asyncio

from vectorforge.embedders.engine import EmbeddingEngine
from vectorforge.models import ChunkResult, EmbeddedRecord


async def worker(
    _worker_id: int,
    work_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    engine: EmbeddingEngine,
):
    while True:
        item = await work_queue.get()
        if item is None:  # sentinel
            work_queue.task_done()
            await result_queue.put(None)  # signal drain that this worker is done
            break

        chunk_id, records = item
        texts = [r.text for r in records]

        result = await engine.embed(texts)

        embedded = [
            EmbeddedRecord(
                row_id=r.row_id,
                source_row_id=r.source_row_id,
                chunk_id=r.chunk_id,
                chunk_index=r.chunk_index,
                text=r.text,
                dense_embedding=result.dense[i] if result.dense else None,
                sparse_embedding=result.sparse[i] if result.sparse else None,
                columns=r.columns,
            )
            for i, r in enumerate(records)
        ]

        await result_queue.put(ChunkResult(chunk_id=chunk_id, records=embedded))
        work_queue.task_done()
