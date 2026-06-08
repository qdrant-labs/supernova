"""Closed-loop load generator + single-worker result summary.

Holds ``profile.concurrency`` requests in flight for ``profile.duration_s``,
recording every latency. Aggregating ACROSS workers is a separate step (see
``nova storm-dist``) and must merge latency *distributions* — raw samples or
histograms — never average per-worker percentiles.
"""

import asyncio
import time
from dataclasses import dataclass

from supernova import metrics
from supernova.storm.base import BaseLoadTester, LoadProfile, QueryResult


@dataclass
class StormResults:
    """One worker's raw measurements."""

    latencies_s: list[float]
    n_ok: int
    n_err: int
    wall_s: float

    def summary(self) -> dict:
        """p50/p95/p99 etc. for THIS worker. Fleet-wide stats must be computed by
        merging raw samples from every worker, not by averaging these."""
        import numpy as np

        lat_ms = np.array(self.latencies_s or [0.0]) * 1000.0
        total = self.n_ok + self.n_err
        return {
            "requests": total,
            "errors": self.n_err,
            "throughput_qps": round(total / self.wall_s, 1) if self.wall_s else 0.0,
            "p50_ms": round(float(np.percentile(lat_ms, 50)), 2),
            "p95_ms": round(float(np.percentile(lat_ms, 95)), 2),
            "p99_ms": round(float(np.percentile(lat_ms, 99)), 2),
            "max_ms": round(float(lat_ms.max()), 2),
        }


async def run_storm(
    tester: BaseLoadTester,
    vectors: list[list[float]],
    profile: LoadProfile,
    filter_spec: dict | None = None,
) -> StormResults:
    """
    Drive one worker's load profile and collect latencies.

    Two modes, chosen by ``profile.target_qps``:
      * ``0`` (default) — closed-loop: hold ``concurrency`` requests in flight,
        measuring the max throughput the cluster will give at that depth.
      * ``>0`` — open-loop paced: launch one query every ``1/target_qps`` seconds
        with ``concurrency`` as an in-flight ceiling, measuring latency AT a fixed
        offered rate.
    """
    await tester.setup()
    query_filter = tester.compile_filter(filter_spec)

    latencies: list[float] = []
    n_ok = n_err = 0
    stop_at = time.perf_counter() + profile.duration_s

    def record(r: QueryResult) -> None:
        nonlocal n_ok, n_err
        latencies.append(r.latency_s)
        metrics.observe("latency_ms", r.latency_s * 1000.0, ok=r.ok)
        if r.ok:
            n_ok += 1
        else:
            n_err += 1

    t0 = time.perf_counter()
    # TODO: honor profile.ramp_s (stagger task starts), and add a coordinated
    # wall-clock start across the fleet so all workers hammer simultaneously
    # (otherwise "fleet p99 at N QPS" was never actually measured at N QPS).
    if profile.target_qps and profile.target_qps > 0:
        await _run_paced(tester, vectors, query_filter, profile, stop_at, record)
    else:
        await _run_closed_loop(tester, vectors, query_filter, profile, stop_at, record)
    wall = time.perf_counter() - t0
    await tester.close()
    return StormResults(latencies_s=latencies, n_ok=n_ok, n_err=n_err, wall_s=wall)


async def _run_closed_loop(tester, vectors, query_filter, profile, stop_at, record):
    """Hold ``concurrency`` requests in flight until the window closes; each task
    fires the next query the instant its previous one returns."""
    n = len(vectors)
    idx = 0

    async def loop() -> None:
        nonlocal idx
        # asyncio is single-threaded, so `idx += 1` between awaits is race-free.
        while time.perf_counter() < stop_at:
            v = vectors[idx % n]
            idx += 1
            record(await tester.query(v, query_filter))

    tasks = [asyncio.create_task(loop()) for _ in range(profile.concurrency)]
    await asyncio.gather(*tasks)


async def _run_paced(tester, vectors, query_filter, profile, stop_at, record):
    """Open-loop: launch a query every ``1/target_qps`` seconds regardless of
    whether prior ones have returned — this is what avoids coordinated omission
    (a slow response can't delay the next launch and hide latency).

    ``concurrency`` caps in-flight requests as a safety valve. When the target is
    achievable in-flight stays low and pacing is smooth; when the cluster can't
    keep up the cap fills, ``acquire`` stalls the dispatcher, and the achieved QPS
    sags below target — which is itself the finding ("this cluster tops out below
    N QPS"), not an error.
    """
    n = len(vectors)
    interval = 1.0 / profile.target_qps
    sem = asyncio.Semaphore(profile.concurrency)
    inflight: set[asyncio.Task] = set()
    idx = 0

    async def one(v) -> None:
        try:
            record(await tester.query(v, query_filter))
        finally:
            sem.release()

    next_dispatch = time.perf_counter()
    while time.perf_counter() < stop_at:
        await sem.acquire()
        # The acquire above may have blocked; re-check before launching.
        if time.perf_counter() >= stop_at:
            sem.release()
            break
        v = vectors[idx % n]
        idx += 1
        task = asyncio.create_task(one(v))
        inflight.add(task)
        task.add_done_callback(inflight.discard)

        next_dispatch += interval
        delay = next_dispatch - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        # delay <= 0 means we're behind schedule; loop immediately to catch up.

    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)