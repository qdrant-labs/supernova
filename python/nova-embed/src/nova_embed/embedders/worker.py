import asyncio

from nova_embed.embedders.engine import EmbeddingEngine
from nova_embed.models import ChunkResult, EmbeddedRecord


async def worker(
    _worker_id: int,
    work_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    engine: EmbeddingEngine,
    drop_columns: frozenset[str] = frozenset(),
):
    """Embed chunks off the work queue.

    ``drop_columns`` are removed from each row HERE — after embedding (an
    input_column may be dropped, that's the primary use) but before the records
    enter the flush buffer, so a fat column (e.g. raw image bytes) doesn't sit
    in memory for up to flush_threshold records, and never reaches the parquet.
    """
    checked_drops = not drop_columns

    while True:
        item = await work_queue.get()
        if item is None:  # sentinel
            work_queue.task_done()
            await result_queue.put(None)  # signal drain that this worker is done
            break

        chunk_id, records = item

        if not checked_drops:
            # typo harassment: a drop that matches nothing is almost certainly
            # a misspelled column (or one already gone via exclude_columns)
            missing = sorted(drop_columns - records[0].row.keys())
            if missing:
                raise ValueError(
                    f"pipeline.drop_columns {missing} not found in rows. "
                    f"Available columns: {sorted(records[0].row)}. Check the "
                    f"spelling, and whether source.exclude_columns already "
                    f"removed it."
                )
            checked_drops = True

        # name -> row-aligned [embedding | None] for every configured entry
        result = await engine.embed([r.row for r in records])

        embedded = [
            EmbeddedRecord(
                row={k: v for k, v in r.row.items() if k not in drop_columns},
                embeddings={name: outputs[i] for name, outputs in result.items()},
            )
            for i, r in enumerate(records)
        ]

        await result_queue.put(ChunkResult(chunk_id=chunk_id, records=embedded))
        work_queue.task_done()
