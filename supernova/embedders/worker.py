import asyncio

from supernova.embedders.engine import EmbeddingEngine
from supernova.models import ChunkResult, EmbeddedRecord


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
                text=r.text,
                dense_embedding=result.dense[i] if result.dense else None,
                sparse_embedding=result.sparse[i] if result.sparse else None,
                multivector_embedding=result.multivector[i]
                if result.multivector
                else None,
                columns=r.columns,
            )
            for i, r in enumerate(records)
        ]

        await result_queue.put(ChunkResult(chunk_id=chunk_id, records=embedded))
        work_queue.task_done()
