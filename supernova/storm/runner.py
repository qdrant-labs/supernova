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
    profile: LoadProfile
) -> StormResults:
    """
    Drive one worker's load profile (closed-loop) and collect latencies.
    """
    await tester.setup()
    latencies: list[float] = []
    n_ok = n_err = 0
    n = len(vectors)
    idx = 0
    stop_at = time.perf_counter() + profile.duration_s

    def record(r: QueryResult) -> None:
        nonlocal n_ok, n_err
        latencies.append(r.latency_s)
        metrics.observe("latency_ms", r.latency_s * 1000.0, ok=r.ok)
        if r.ok:
            n_ok += 1
        else:
            n_err += 1

    async def loop() -> None:
        nonlocal idx
        # asyncio is single-threaded, so `idx += 1` between awaits is race-free.
        while time.perf_counter() < stop_at:
            v = vectors[idx % n]
            idx += 1
            record(await tester.query(v))

    t0 = time.perf_counter()
    # TODO: honor profile.ramp_s (stagger task starts), and add a coordinated
    # wall-clock start across the fleet so all workers hammer simultaneously
    # (otherwise "fleet p99 at N QPS" was never actually measured at N QPS).
    tasks = [asyncio.create_task(loop()) for _ in range(profile.concurrency)]
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    await tester.close()
    return StormResults(latencies_s=latencies, n_ok=n_ok, n_err=n_err, wall_s=wall)