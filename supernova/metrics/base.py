"""The metrics backend contract.

A ``MetricsBackend`` is *where measurements go*. Commands (embed/load/storm) and
anything they call emit through the ambient backend (``supernova.metrics``'s
module-level ``log``/``observe``/``event``); the backend decides whether that
means printing, batching INSERTs into Postgres, or doing nothing.

Two deliberate choices:

* **The base is all no-ops.** ``NullBackend`` is just this class. A custom
  backend subclasses it and overrides only the verbs it cares about — that is the
  whole "write ``my_backend.py`` and override functions" extension story.
* **Backends are fail-open.** A metrics hiccup must never crash the workload it
  observes. Buffer, retry, drop-on-overflow, swallow-and-warn — never raise into
  the caller. A six-hour embed job dying because Postgres blipped is unacceptable.
"""

import time
from contextlib import contextmanager


class MetricsBackend:
    def init(self) -> None:
        """One-time backend setup before any run: open connections, create or
        validate schema, fail fast on a bad config. Called once by the bootstrap
        before start(), so a bad DSN errors before the workload spins up."""

    def start(self, run_id: str, context: dict) -> None:
        """Open the run. ``run_id`` is unique per execution (so reruns don't
        collide); ``context`` is the ambient identity every emission inherits —
        at least ``node_id``, ``command``, and the resolved ``config``. A DB
        backend writes the ``runs`` row here. Called once by the bootstrap."""

    def finish(self, status: str = "ok") -> None:
        """Flush and close. Always called, including on exceptions."""

    def log(self, name: str, value: float, **tags) -> None:
        """A scalar time-series point: rolling QPS, writes/sec, batch size."""

    def observe(self, name: str, value: float, **tags) -> None:
        """One sample of a distribution: a single query's latency. Backends that
        compute percentiles/histograms aggregate these; simple ones treat it like
        ``log``."""

    def event(self, message: str, **tags) -> None:
        """A timestamped annotation — 'workload started', 'indexing enabled', an
        error. Renders as a Grafana annotation. Keep it low-volume."""

    def summary(self, values: dict) -> None:
        """The final result record for this run/node (p50/p95/p99, totals). One
        low-frequency write, distinct from the per-sample stream."""

    def flush(self) -> None:
        """Force buffered emissions out. Best-effort."""

    @contextmanager
    def timed(self, name: str, **tags):
        """``with metrics.timed("query_ms"): ...`` -> observe() the elapsed ms.
        Sugar over observe(); no need to override."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - t0) * 1000.0, **tags)