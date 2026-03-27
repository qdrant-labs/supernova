import heapq

from vectorforge.models import ChunkResult


class ResultBuffer:
    """
    Priority queue that holds ChunkResults and flushes them in order.
    Ensures chunk_0 is written before chunk_1 even if chunk_1 arrives first.
    """

    def __init__(self, flush_fn):
        self._heap: list[tuple[int, ChunkResult]] = []
        self._next_expected = 0
        self._flush_fn = flush_fn  # async fn(ChunkResult) -> None

    async def push(self, result: ChunkResult):
        heapq.heappush(self._heap, (result.chunk_id, result))
        await self._try_flush()

    async def _try_flush(self):
        while self._heap and self._heap[0][0] == self._next_expected:
            _, result = heapq.heappop(self._heap)
            await self._flush_fn(result)
            self._next_expected += 1

    async def drain(self):
        """
        Flush any remaining chunks at end of pipeline.
        """
        await self._try_flush()
