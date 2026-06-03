"""Core contract for `nova storm`.

A load tester knows how to fire ONE query and report its latency/result. The
load *profile* (how many concurrent requests, for how long, how to ramp) is
driven by the runner — not the backend — so a backend stays a thin "issue one
query" adapter and the same runner drives any vector store.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """Outcome of a single query."""

    latency_s: float
    ok: bool
    returned_ids: list = field(default_factory=list)
    error: str | None = None


@dataclass
class LoadProfile:
    """Per-worker load shape.

    Replicated across the fleet (NOT sharded): every storm worker runs this same
    profile, so total offered load ≈ ``num_workers × concurrency``.
    """

    concurrency: int = 32
    duration_s: float = 60.0
    ramp_s: float = 0.0
    # TODO: open-loop mode (hold a target QPS regardless of latency) in addition
    # to the current closed-loop mode (hold a fixed in-flight concurrency).


class BaseLoadTester(ABC):
    """Issues queries against a target store. One instance per worker."""

    @abstractmethod
    async def setup(self) -> None:
        """Connect / warm up before the measured window opens."""

    @abstractmethod
    async def query(self, vector: list[float]) -> QueryResult:
        """Fire a single query and return its result + latency."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down connections."""