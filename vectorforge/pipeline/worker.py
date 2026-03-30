import asyncio

from vectorforge.embedders.base import Embedder
from vectorforge.models import ChunkResult, EmbeddedRecord


async def worker(
    _worker_id: int,
    work_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    embedder: Embedder,
):
    while True:
        item = await work_queue.get()
        if item is None:  # sentinel
            work_queue.task_done()
            await result_queue.put(None)  # signal drain that this worker is done
            break

        chunk_id, records = item
        texts = [r.text for r in records]

        embeddings = await embedder.embed(texts)

        embedded = [
            EmbeddedRecord(
                row_id=r.row_id,
                source_row_id=r.source_row_id,
                chunk_id=r.chunk_id,
                chunk_index=r.chunk_index,
                text=r.text,
                source=r.source,
                embedding=emb,
                model=embedder.model_name,
                payload=r.payload,
            )
            for r, emb in zip(records, embeddings)
        ]

        await result_queue.put(ChunkResult(chunk_id=chunk_id, records=embedded))
        work_queue.task_done()
