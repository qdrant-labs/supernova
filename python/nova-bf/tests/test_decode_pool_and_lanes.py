"""The merge's decode-pool unclamp, and the lanes decision that replaced the probe.

THE DECODE POOL. pyarrow takes its CPU-pool default from OMP_NUM_THREADS, and a
GPU-only Ray/SkyPilot task gets OMP_NUM_THREADS=1, so the pool is ONE thread and
every reader thread in the reduce queues behind it. Measured on the production
32 x 4.04 GB dense partials, raising it alone took dense `reduce_s` 434.2 -> 218.7
and `io_wait_s` 263.6 -> 32.0. It presents as an IO bottleneck -- idle CPU, flat
throughput however you tune the GET concurrency -- so nothing about it is
self-announcing, which is why it needs a test rather than a comment.

THE LANES DECISION. `lanes_mode` used to be decided by a PREFETCH of rank 0's
whole `hit_ids` column before the reduce; it now comes from the first partial
the reduce fetches anyway. The property that makes that safe is that `lanes_mode`
picks an id REPRESENTATION and not a ranking, so either choice must produce
identical hits. `tests/parity` asserts this only under CUDA, so on a CPU-only
machine nothing checked it -- these tests do, via NOVA_BF_MERGE_FOLD=cpu.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nova_bf.merge as merge_mod
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.compute import _usable_cpu_count
from nova_bf.results import build_result_table

K = 4


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Both switches below are read from the ambient environment.

    `NOVA_BF_NO_GPU_ORDINALS` forces `_decide_lanes` to return False. With it
    set, `test_a_fully_dense_..._enables_lanes` FAILS and -- far worse --
    `test_the_decision_reads_past_the_first_slice` passes VACUOUSLY, because
    its `is False` becomes unconditional and the head-only-scan mutant it
    exists to catch sails through. A test that cannot fail is worse than no
    test, so pin the environment rather than inherit it.
    """
    monkeypatch.delenv("NOVA_BF_NO_GPU_ORDINALS", raising=False)
    monkeypatch.delenv("NOVA_BF_MERGE_FOLD", raising=False)
    # OMP_NUM_THREADS is the INPUT the unclamp tests exercise, so inheriting it
    # is not a nuisance but a correctness problem: `export OMP_NUM_THREADS=3`
    # in a shell profile (routine on an ML box, and set by many CI images)
    # makes `3 == pa.cpu_count() < usable` true and the helper widen a pool a
    # test asserted was untouched. Each test sets the value it means to test.
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)


@pytest.fixture(autouse=True)
def _restore_pool():
    """`set_cpu_count` is process-global; never leak it into another test."""
    before = pa.cpu_count()
    try:
        yield
    finally:
        pa.set_cpu_count(before)


def _cfg(root: str, cpu_thread_count: int = 0) -> BruteForceConfig:
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=f"{root}/corpus"),
        queries=QueriesConfig(path=f"{root}/queries.parquet"),
        output=OutputConfig(path=root),
        searches=[SearchSpec(name="test", k=K)],
    )
    cfg.params.cpu_thread_count = cpu_thread_count
    return cfg


# --------------------------------------------------------------------------
# _unclamp_decode_threads
# --------------------------------------------------------------------------

def test_a_pool_pinned_to_one_thread_is_raised(tmp_path, monkeypatch, caplog):
    """The whole point: a hostile launcher must not leave the pool at 1."""
    if _usable_cpu_count() < 2:
        pytest.skip("single-CPU runner: there is no wider pool to raise to")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    pa.set_cpu_count(1)
    with caplog.at_level("INFO"):
        merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))
    assert pa.cpu_count() > 1, "decode pool left pinned at one thread"
    assert pa.cpu_count() == _usable_cpu_count()
    assert "decode pool" in caplog.text


def test_a_pool_with_no_omp_behind_it_is_left_alone(tmp_path, monkeypatch):
    """With no OMP_NUM_THREADS at all there is no launcher clamp to undo.

    NOT "anything but 1 is someone's choice" -- that was the old rule, and a
    pool of 3 with OMP_NUM_THREADS=3 on a 24-CPU box IS a clamp under the
    current one and IS raised. What makes this pool untouchable is that no
    launcher variable explains it. The companion case (OMP set but not matching
    the pool) is `test_a_width_no_launcher_explains_is_left_alone`; this one
    pins the "no launcher involved" branch a developer hits on a laptop.
    """
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    target = max(2, min(3, _usable_cpu_count()))
    pa.set_cpu_count(target)
    merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))
    assert pa.cpu_count() == target


def test_cpu_thread_count_is_the_opt_out_and_can_lower_the_pool(tmp_path):
    """The real escape hatch, since pyarrow ignores ARROW_NUM_THREADS.

    It must be honoured whatever the pool currently says -- including DOWN, and
    including from an unclamped pool -- or it is not an opt-out at all.
    """
    pa.set_cpu_count(max(2, _usable_cpu_count()))
    merge_mod._unclamp_decode_threads(_cfg(str(tmp_path), cpu_thread_count=1))
    assert pa.cpu_count() == 1


def test_the_pool_never_exceeds_what_affinity_allows(tmp_path, monkeypatch):
    """A container gets its own share, not the host's core count.

    `os.cpu_count()` sees the host; the merge must not size a 96-thread pool
    inside a 4-CPU slice. This is review finding 17 (2026-09-01), which the
    first version of this helper re-introduced by calling `os.cpu_count()`.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(merge_mod.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(merge_mod.os, "sched_getaffinity", lambda _pid: set(range(4)))
    pa.set_cpu_count(1)
    merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))
    assert pa.cpu_count() == 4, "sized from the host, ignoring this process's affinity"


def test_a_failure_to_resize_never_fails_the_merge(tmp_path, monkeypatch, caplog):
    """Decoding slowly is a bad day; refusing to merge is a lost run.

    Restored in the test's own `finally`, and that is REQUIRED -- do not
    "simplify" this to monkeypatch.

    `_clean_env` above is autouse AND takes `monkeypatch`, which makes
    monkeypatch a dependency of an autouse fixture: it is set up first and so
    finalises LAST, after both autouse teardowns. Measured on this file's
    actual fixture set -- both autouse teardowns observe the stub still live.
    So a monkeypatched `pa.set_cpu_count` stub is still in place when
    `_restore_pool` calls it, turning this test into a teardown error and
    leaking a raising stub into the rest of the session.

    (Two earlier versions of this comment got the order wrong in opposite
    directions. It is not a fixed property of pytest -- it flips with whether
    an autouse fixture depends on monkeypatch, which `_clean_env` now does.
    Re-measure before trusting any claim here, including this one.)
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    real_set, real_count = pa.set_cpu_count, pa.cpu_count

    def boom(_n):
        raise RuntimeError("nope")

    pa.set_cpu_count, pa.cpu_count = boom, (lambda: 1)
    try:
        with caplog.at_level("WARNING"):
            merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))   # must not raise
    finally:
        pa.set_cpu_count, pa.cpu_count = real_set, real_count
    assert "could not set" in caplog.text


# --------------------------------------------------------------------------
# the lanes decision
# --------------------------------------------------------------------------

def _partial(qids, widths):
    """One partial table whose row i holds `widths[i]` hits."""
    ids = [[f"<urn:uuid:{i:08x}-{j:04x}>" for j in range(w)]
           for i, w in enumerate(widths)]
    scores = [[1.0 - j * 0.01 for j in range(w)] for w in widths]
    return build_result_table(list(qids), {}, ids, scores)


def test_the_decision_reads_past_the_first_slice(tmp_path):
    """The id WIDTH must be fixed across every row, so a prefix cannot decide
    it: a partial whose LATER ids are a different width cannot use lanes.

    The fixture spans TWO CHUNKS on purpose. The slice bound is
    `2_000_000 // k` rows, so at a test-sized k any modest fixture fits in one
    slice and a head-only scan would still see every row -- which is how an
    earlier version of this test passed against the very bug it named. Two
    chunks force the loop to continue past its first piece whatever the bound
    resolves to.
    """
    head = _partial([f"a{i}" for i in range(64)], [K] * 64)
    late = build_result_table(
        [f"b{i}" for i in range(64)], {},
        [[f"<urn:uuid:{i:08x}-{j:05x}>" for j in range(K)] for i in range(64)],
        [[1.0 - j * 0.01 for j in range(K)] for _ in range(64)])
    tbl = pa.concat_tables([head, late])
    assert len(tbl.column("hit_ids").chunks) == 2, "fixture is not multi-chunk"
    cfg = _cfg(str(tmp_path))
    assert merge_mod._decide_lanes(tbl, cfg.searches[0]) is False


def test_short_rows_still_enable_lanes(tmp_path):
    """Row LENGTH no longer matters: the device fold takes ragged rows
    (trimmed partials, queries with fewer than k hits), so a fixed-width
    partial with short or empty rows keeps its state in lanes."""
    dense = _partial([f"a{i}" for i in range(64)], [K] * 64)
    short = _partial([f"b{i}" for i in range(64)], [K] * 60 + [1, 0, 2, 0])
    tbl = pa.concat_tables([dense, short])
    cfg = _cfg(str(tmp_path))
    assert merge_mod._decide_lanes(tbl, cfg.searches[0]) is True


def test_a_partial_with_no_hits_keeps_arrow(tmp_path):
    """No ids at all means no width to pack; stay on Arrow."""
    tbl = _partial([f"q{i}" for i in range(16)], [0] * 16)
    assert merge_mod._decide_lanes(tbl, _cfg(str(tmp_path)).searches[0]) is False


def test_the_density_scan_stays_bounded_by_k(tmp_path, monkeypatch):
    """The bound itself, pinned: `_decide_lanes` must slice, not gulp.

    Asserting on the resulting slice sizes cannot catch this -- a test-sized
    fixture is smaller than the bound either way -- so spy on the argument.
    """
    seen: list[int] = []
    real = merge_mod._sliced
    monkeypatch.setattr(merge_mod, "_sliced",
                        lambda col, rows: (seen.append(rows), real(col, rows))[1])
    tbl = _partial([f"q{i}" for i in range(32)], [K] * 32)
    merge_mod._decide_lanes(tbl, _cfg(str(tmp_path)).searches[0])
    assert seen == [max(256, min(8192, 2_000_000 // K))]


def test_a_fully_dense_fixed_width_partial_enables_lanes(tmp_path):
    cfg = _cfg(str(tmp_path))
    tbl = _partial([f"q{i}" for i in range(64)], [K] * 64)
    assert merge_mod._decide_lanes(tbl, cfg.searches[0]) is True


def test_the_scan_is_bounded_per_call_on_a_single_row_group(tmp_path):
    """`_fixed_width` must not be handed the whole column at once.

    A partial is written with pyarrow's default row-group size, so a real one
    is ONE chunk: iterating chunks alone hands `_fixed_width` a 1e8-element
    child array whose internal `np.diff` allocates ~800 MB in the consumer
    thread. The deleted prefetch bounded itself to ~2M ids for this reason.
    """
    rows = 5000
    tbl = _partial([f"q{i}" for i in range(rows)], [K] * rows)
    col = tbl.column("hit_ids")
    assert len(col.chunks) == 1, "fixture is not one chunk"
    # A bound far below the fixture, so yielding the chunk whole is visible.
    sizes = [len(s) for s in merge_mod._sliced(col, 64)]
    assert sizes and max(sizes) <= 64, "a slice exceeded the bound"
    assert sum(sizes) == rows, "slicing dropped or duplicated rows"


@pytest.mark.parametrize("order", ["dense_first", "short_first"])
def test_lanes_and_arrow_agree_whichever_partial_decides(tmp_path, monkeypatch, order):
    """THE invariant the change rests on, and the one parity only checks on CUDA.

    `lanes_mode` is now set by whichever partial arrives first, which is a race
    between reader threads. That is only acceptable because the choice cannot
    change the answer. Fold the same two partials both ways and require
    identical ids and scores.
    """
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    qids = [f"q{i}" for i in range(32)]
    dense = _partial(qids, [K] * 32)
    short = _partial(qids, [K] * 30 + [1, 1])
    first, second = (dense, short) if order == "dense_first" else (short, dense)

    def fold(lanes: bool):
        state = None
        for tbl in (first, second):
            sc = tbl.column("hit_scores").combine_chunks()
            ids = tbl.column("hit_ids").combine_chunks()
            if state is None:
                state = merge_mod._topk_merge([sc], [ids], None, K, lanes_mode=lanes)
            else:
                state = merge_mod._topk_merge(
                    [state[1], sc], [state[0], ids], None, K, lanes_mode=lanes)
        out_ids, out_sc = state[0], state[1]
        return (merge_mod._as_ids_array(out_ids).to_pylist(), out_sc.to_pylist())

    assert fold(True) == fold(False), "the id representation changed the result"


def test_a_task_that_also_asked_for_cpus_is_still_unclamped(tmp_path, monkeypatch):
    """The clamp is not always 1, and keying on 1 silently missed the rest.

    Ray sets OMP_NUM_THREADS to the task's CPU allocation. `accelerators: A10G:1`
    alone gives 1, but a task that ALSO asks for `cpus: 4` gets 4 -- and a merge
    decoding 4-wide on a 32-core box is the same bug, two thirds as bad, with
    nothing in the log to say so. Detection keys on "the pool equals
    OMP_NUM_THREADS and is below what we can use", not on a magic number.
    """
    if _usable_cpu_count() < 5:
        pytest.skip("need >4 usable CPUs to tell a 4-thread clamp from the real width")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    pa.set_cpu_count(4)
    merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))
    assert pa.cpu_count() == _usable_cpu_count()


def test_a_width_no_launcher_explains_is_left_alone(tmp_path, monkeypatch):
    """The other half: only OMP-derived widths are ours to override.

    A pool width that does not match OMP_NUM_THREADS was chosen by somebody --
    directly, or by an embedder sharing this process -- and silently widening it
    would be exactly the overreach the cgroup case warns about.
    """
    if _usable_cpu_count() < 5:
        pytest.skip("need >4 usable CPUs for a width below the real one")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")   # does NOT match the pool
    pa.set_cpu_count(4)
    merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))
    assert pa.cpu_count() == 4


def test_an_unparseable_omp_leaves_a_narrow_pool_but_says_so(tmp_path, monkeypatch, caplog):
    """The one path where the merge runs throttled and could stay silent.

    Arrow is more permissive than `int()`: it clamps the pool on "1.0" and on
    the legal nested form "4,2", both of which land in the `except ValueError`
    branch. Declining to widen is right -- we cannot tell a clamp from a
    coincidence -- but doing it silently is not, because the merge then decodes
    at a fraction of the box with nothing in the log and a plausible-looking
    `cpu_thread_count` in the manifest. That is the exact signature of the
    original bug, which took a full day to find precisely because it was quiet.
    """
    if _usable_cpu_count() < 2:
        pytest.skip("need >1 usable CPU for the pool to count as narrow")
    monkeypatch.setenv("OMP_NUM_THREADS", "1.0")
    pa.set_cpu_count(1)
    with caplog.at_level("WARNING"):
        merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))
    assert pa.cpu_count() == 1, "declined to widen, as intended"
    assert "OMP_NUM_THREADS" in caplog.text and "cpu_thread_count" in caplog.text


def test_a_wide_pool_with_unparseable_omp_stays_quiet(tmp_path, monkeypatch, caplog):
    """The warning must not fire when there is nothing to warn about.

    "abc" and "" make Arrow ignore the variable entirely and use hardware
    concurrency, so the pool is already full width -- warning there would train
    operators to ignore the message.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "abc")
    pa.set_cpu_count(_usable_cpu_count())
    with caplog.at_level("WARNING"):
        merge_mod._unclamp_decode_threads(_cfg(str(tmp_path)))
    assert "OMP_NUM_THREADS" not in caplog.text
