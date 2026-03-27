import pytest

from vectorforge.pipeline.buffer import ResultBuffer
from vectorforge.models import ChunkResult, EmbeddedRecord


def _make_chunk(chunk_id: int) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        records=[
            EmbeddedRecord(
                row_id=chunk_id * 10 + i,
                chunk_id=chunk_id,
                text=f"text {i}",
                source="test",
                embedding=[0.0],
                model="test-model",
            )
            for i in range(3)
        ],
    )


@pytest.mark.asyncio
async def test_buffer_flushes_in_order():
    flushed = []

    async def flush_fn(result: ChunkResult):
        flushed.append(result.chunk_id)

    buffer = ResultBuffer(flush_fn=flush_fn)

    # Push chunks out of order
    await buffer.push(_make_chunk(2))
    assert flushed == []

    await buffer.push(_make_chunk(0))
    assert flushed == [0]

    await buffer.push(_make_chunk(1))
    assert flushed == [0, 1, 2]


@pytest.mark.asyncio
async def test_buffer_drain():
    flushed = []

    async def flush_fn(result: ChunkResult):
        flushed.append(result.chunk_id)

    buffer = ResultBuffer(flush_fn=flush_fn)
    await buffer.push(_make_chunk(0))
    await buffer.push(_make_chunk(1))
    await buffer.drain()
    assert flushed == [0, 1]


@pytest.mark.asyncio
async def test_buffer_holds_gaps():
    flushed = []

    async def flush_fn(result: ChunkResult):
        flushed.append(result.chunk_id)

    buffer = ResultBuffer(flush_fn=flush_fn)
    await buffer.push(_make_chunk(1))
    await buffer.push(_make_chunk(3))
    await buffer.drain()
    # Should not flush anything — chunk 0 is missing
    assert flushed == []
