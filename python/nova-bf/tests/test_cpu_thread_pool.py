"""`params.cpu_thread_count` — pyarrow's CPU pool, not its IO pool.

The bug this guards against was silent and cost ~30% of read time on a real
run: pyarrow takes its CPU-pool default from OMP_NUM_THREADS, GPU images
commonly pin that to 1, and nova-bf used to set only the IO pool. A decode
pool of one thread throttles every reader thread at once and presents exactly
like an IO bottleneck — the CPU sits idle, `read_wall_s` stays flat however you
tune `io_workers`/`io_thread_count`, and nothing warns.

So the property under test is not "the number is stored somewhere" but "a
hostile environment cannot leave the pool at 1".
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

pytest.importorskip("torch")

import nova_bf.compute as compute_mod
from nova_bf.compute import run_compute
from nova_bf.config import BruteForceConfig


def _cfg(tmp_path, **params):
    import numpy as np
    import pyarrow.parquet as pq

    cdir = tmp_path / "corpus"
    cdir.mkdir()
    rng = np.random.default_rng(0)
    for name, n in (("a", 8), ("b", 5)):
        pq.write_table(pa.table({
            "dense_embedding": pa.array(
                rng.standard_normal((n, 4)).astype("float32").tolist(),
                type=pa.list_(pa.float32())),
            "id": pa.array([f"{name}{i}" for i in range(n)]),
        }), str(cdir / f"{name}.parquet"))
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((3, 4)).astype("float32").tolist(),
            type=pa.list_(pa.float32())),
        "qid": pa.array(["0", "1", "2"]),
    }), str(tmp_path / "q.parquet"))

    return BruteForceConfig.model_validate({
        "corpus": {"path": str(cdir), "id_column": "id"},
        "queries": {"path": str(tmp_path / "q.parquet"), "id_column": "qid"},
        "output": {"path": str(tmp_path / "out")},
        "params": params,
        "searches": [{"name": "d", "vector_type": "dense", "metric": "cosine", "k": 3}],
    })


@pytest.fixture(autouse=True)
def _restore_pool():
    """Every test here mutates a PROCESS-GLOBAL pyarrow setting; leaving it
    changed would silently re-tune every later test in the session."""
    before = pa.cpu_count()
    yield
    pa.set_cpu_count(before)


def test_a_hostile_default_is_overridden(tmp_path):
    """The actual failure: the environment hands us a pool of 1 and nothing
    complains. A run must not inherit it."""
    pa.set_cpu_count(1)
    run_compute(_cfg(tmp_path))
    assert pa.cpu_count() > 1, (
        "run_compute left pyarrow's CPU pool at 1 — parquet decode would be "
        "single-threaded for every reader thread at once")
    # Against what this process may actually USE, not what the machine has:
    # under a container CPU limit those differ, and taking the machine's
    # number would oversubscribe the decode pool (see `_usable_cpu_count`).
    assert pa.cpu_count() == compute_mod._usable_cpu_count()


def test_explicit_value_is_honored(tmp_path):
    pa.set_cpu_count(1)
    run_compute(_cfg(tmp_path, cpu_thread_count=3))
    assert pa.cpu_count() == 3


def test_cli_override_beats_the_config(tmp_path):
    """`--cpu-thread-count` exists to sweep this knob without editing a config, the
    same way `--io-thread-count` does."""
    pa.set_cpu_count(1)
    run_compute(_cfg(tmp_path, cpu_thread_count=2), cpu_thread_count=5)
    assert pa.cpu_count() == 5


def test_zero_means_auto_not_zero(tmp_path):
    """0 is the config default and means "use every core" — NOT "set the pool
    to 0", which pyarrow would reject, and not "leave it alone", which is what
    `io_thread_count`'s 0 means. The two knobs read the same and differ here."""
    pa.set_cpu_count(1)
    run_compute(_cfg(tmp_path, cpu_thread_count=0))
    # Against what this process may actually USE, not what the machine has:
    # under a container CPU limit those differ, and taking the machine's
    # number would oversubscribe the decode pool (see `_usable_cpu_count`).
    assert pa.cpu_count() == compute_mod._usable_cpu_count()


def test_the_two_pools_are_not_the_same_pool(tmp_path):
    """Setting the IO pool must not move the CPU pool. They are different
    pools doing different jobs (fetch vs decode); conflating them is what made
    the original problem invisible for so long."""
    pa.set_cpu_count(1)
    run_compute(_cfg(tmp_path, io_thread_count=7, cpu_thread_count=3))
    assert pa.io_thread_count() == 7
    assert pa.cpu_count() == 3


# ---------------------------------------------------------------------------
# what "all the CPUs" means in a container
# ---------------------------------------------------------------------------


def test_usable_cpu_count_respects_affinity(monkeypatch):
    """`taskset`/cpuset pinning: the machine has more, this process may not
    use them."""
    monkeypatch.setattr(compute_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(compute_mod.os, "sched_getaffinity", lambda _p: set(range(4)))
    monkeypatch.setattr(compute_mod, "_cgroup_cpu_quota", lambda: None)
    assert compute_mod._usable_cpu_count() == 4


def test_usable_cpu_count_respects_a_cgroup_quota(monkeypatch):
    """A CPU-TIME quota (`docker --cpus`, a k8s limit) caps how much CPU the
    group may consume, not WHICH cpus — affinity still reports all 64, so the
    quota has to be read separately or a 4-CPU pod gets a 64-thread pool."""
    monkeypatch.setattr(compute_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(compute_mod.os, "sched_getaffinity", lambda _p: set(range(64)))
    monkeypatch.setattr(compute_mod, "_cgroup_cpu_quota", lambda: 4)
    assert compute_mod._usable_cpu_count() == 4


def test_usable_cpu_count_takes_the_tighter_of_the_two(monkeypatch):
    monkeypatch.setattr(compute_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(compute_mod.os, "sched_getaffinity", lambda _p: set(range(8)))
    monkeypatch.setattr(compute_mod, "_cgroup_cpu_quota", lambda: 4)
    assert compute_mod._usable_cpu_count() == 4
    monkeypatch.setattr(compute_mod, "_cgroup_cpu_quota", lambda: 32)
    assert compute_mod._usable_cpu_count() == 8


def test_usable_cpu_count_never_returns_zero(monkeypatch):
    """`pa.set_cpu_count(0)` is rejected outright, so a zero here would be
    worse than the hostile default this replaced.

    An EMPTY affinity set is what actually drives the intermediate to zero —
    `cpu_count() -> None` alone does not, because it is already coerced to 1
    before any clamping.
    """
    monkeypatch.setattr(compute_mod.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(compute_mod.os, "sched_getaffinity", lambda _p: set())
    monkeypatch.setattr(compute_mod, "_cgroup_cpu_quota", lambda: None)
    assert compute_mod._usable_cpu_count() == 1

    monkeypatch.setattr(compute_mod.os, "cpu_count", lambda: None)
    monkeypatch.setattr(compute_mod.os, "sched_getaffinity", lambda _p: set())
    assert compute_mod._usable_cpu_count() == 1


def test_usable_cpu_count_survives_probes_that_fail(monkeypatch):
    """Not Linux, or the cgroup files unreadable: fall back, never raise."""
    def boom(*_a, **_k):
        raise OSError("no")

    monkeypatch.setattr(compute_mod.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(compute_mod.os, "sched_getaffinity", boom)
    monkeypatch.setattr(compute_mod, "_cgroup_cpu_quota", lambda: None)
    assert compute_mod._usable_cpu_count() == 12


def test_cgroup_quota_parses_v2_and_reports_unlimited(monkeypatch, tmp_path):
    real_open = open

    def fake_open(path, *a, **k):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return real_open(tmp_path / "cpu.max", *a, **k)
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)

    (tmp_path / "cpu.max").write_text("400000 100000\n")
    assert compute_mod._cgroup_cpu_quota() == 4
    (tmp_path / "cpu.max").write_text("150000 100000\n")   # 1.5 cpus -> ceil
    assert compute_mod._cgroup_cpu_quota() == 2
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert compute_mod._cgroup_cpu_quota() is None
