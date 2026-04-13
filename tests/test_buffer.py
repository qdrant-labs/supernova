import pytest

from vectorforge.pipeline.buffer import ResultBuffer
from vectorforge.models import ChunkResult, EmbeddedRecord


def _make_chunk(chunk_id: int, num_records: int = 3) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        records=[
            EmbeddedRecord(
                row_id=chunk_id * 10 + i,
                source_row_id=chunk_id * 10 + i,
                chunk_id=chunk_id,
                chunk_index=0,
                text=f"text {i}",
                dense_embedding=[0.0],
            )
            for i in range(num_records)
        ],
    )


@pytest.mark.asyncio
async def test_buffer_flushes_at_threshold():
    flushed_batches = []

    async def flush_fn(records):
        flushed_batches.append(len(records))

    buffer = ResultBuffer(flush_fn=flush_fn, flush_threshold=5)

    await buffer.push(_make_chunk(0, num_records=3))
    assert flushed_batches == []

    await buffer.push(_make_chunk(1, num_records=3))
    assert flushed_batches == [6]


@pytest.mark.asyncio
async def test_buffer_preserves_order():
    flushed_batches = []

    async def flush_fn(records):
        flushed_batches.append([r.chunk_id for r in records])

    buffer = ResultBuffer(flush_fn=flush_fn, flush_threshold=100)

    await buffer.push(_make_chunk(2))
    await buffer.push(_make_chunk(0))
    await buffer.push(_make_chunk(1))

    await buffer.drain()
    assert len(flushed_batches) == 1
    assert flushed_batches[0] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.mark.asyncio
async def test_buffer_holds_gaps():
    flushed_batches = []

    async def flush_fn(records):
        flushed_batches.append(len(records))

    buffer = ResultBuffer(flush_fn=flush_fn, flush_threshold=100)

    await buffer.push(_make_chunk(1))
    await buffer.push(_make_chunk(3))
    await buffer.drain()
    assert flushed_batches == []


@pytest.mark.asyncio
async def test_buffer_drain_flushes_remaining():
    flushed_batches = []

    async def flush_fn(records):
        flushed_batches.append(len(records))

    buffer = ResultBuffer(flush_fn=flush_fn, flush_threshold=100)

    await buffer.push(_make_chunk(0))
    await buffer.push(_make_chunk(1))
    assert flushed_batches == []

    await buffer.drain()
    assert flushed_batches == [6]
