import asyncio

from nova_embed.embedders.engine import EmbeddingEngine
from nova_embed.models import ChunkResult, EmbeddedRecord


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

        # name -> row-aligned [embedding | None] for every configured entry
        result = await engine.embed([r.row for r in records])

        embedded = [
            EmbeddedRecord(
                row=r.row,
                embeddings={name: outputs[i] for name, outputs in result.items()},
            )
            for i, r in enumerate(records)
        ]

        await result_queue.put(ChunkResult(chunk_id=chunk_id, records=embedded))
        work_queue.task_done()
