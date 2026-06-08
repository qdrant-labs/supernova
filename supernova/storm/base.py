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
    profile, so total offered load ≈ ``num_workers × concurrency`` (closed-loop)
    or ``num_workers × target_qps`` (paced).
    """

    concurrency: int = 32
    duration_s: float = 60.0
    ramp_s: float = 0.0
    # 0 = closed-loop: hold `concurrency` requests in flight, measure max
    # throughput. >0 = open-loop: pace one launch every 1/target_qps seconds and
    # let `concurrency` act as the in-flight ceiling. Per worker, like the rest.
    target_qps: float = 0.0
    # TODO: honor ramp_s (stagger task starts) in both modes.


class BaseLoadTester(ABC):
    """Issues queries against a target store. One instance per worker."""

    @abstractmethod
    async def setup(self) -> None:
        """Connect / warm up before the measured window opens."""

    @abstractmethod
    async def query(self, vector: list[float], query_filter=None) -> QueryResult:
        """Fire a single query and return its result + latency.

        ``query_filter`` is the backend-native object produced by
        :meth:`compile_filter` (``None`` = unfiltered), passed in each call so the
        run can compile it once and reuse it.
        """

    @abstractmethod
    async def close(self) -> None:
        """Tear down connections."""

    def compile_filter(self, spec: dict | None):
        """Translate the config's ``query.filter`` block into a native filter
        object, ONCE per run (not per query).

        The filter *shape* is vendor-specific by design — a Qdrant filter dict
        doesn't match Elastic's — so this is the seam where that raw vendor shape
        becomes a native object. Default: no filter support. A ``None`` spec
        returns ``None`` (unfiltered, fine); a populated spec raises, so a
        configured filter is never silently dropped and measured as if it ran
        unfiltered. Backends opt in by overriding.
        """
        if spec:
            raise NotImplementedError(
                f"{type(self).__name__} does not support query filters"
            )
        return None