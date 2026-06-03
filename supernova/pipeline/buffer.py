import heapq
import logging

from supernova.models import ChunkResult, EmbeddedRecord

logger = logging.getLogger(__name__)


class ResultBuffer:
    """
    Priority queue that accumulates ChunkResults in order and flushes
    them as a batch once the record count crosses flush_threshold.

    Ordering guarantee: only flushes a contiguous run starting from
    chunk 0. If chunk 2 arrives before chunk 1, it waits.
    """

    def __init__(self, flush_fn: callable, flush_threshold: int = 100_000):
        self._heap: list[tuple[int, ChunkResult]] = []
        self._next_expected = 0
        self._flush_fn = flush_fn  # async fn(list[EmbeddedRecord]) -> None
        self._flush_threshold = flush_threshold
        self._pending: list[EmbeddedRecord] = []
        self._pending_count = 0

    async def push(self, result: ChunkResult):
        heapq.heappush(self._heap, (result.chunk_id, result))
        await self._try_collect()

    async def _try_collect(self):
        """
        Pop in-order chunks into the pending buffer, flush when threshold is hit.
        """
        while self._heap and self._heap[0][0] == self._next_expected:
            _, result = heapq.heappop(self._heap)
            self._pending.extend(result.records)
            self._pending_count += len(result.records)
            self._next_expected += 1

            if self._pending_count >= self._flush_threshold:
                logger.info(
                    "Buffer hit threshold (%d records), flushing", self._pending_count
                )
                await self._flush()

    async def _flush(self):
        if not self._pending:
            return
        await self._flush_fn(self._pending)
        self._pending = []
        self._pending_count = 0

    async def drain(self):
        """
        Flush any remaining records at end of pipeline.
        """
        await self._try_collect()
        if self._pending:
            logger.info("Draining remaining %d records", self._pending_count)
        await self._flush()
