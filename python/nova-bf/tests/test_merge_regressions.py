"""Regression tests for the partial-major reduce in `merge._reduce`.

Concurrency / resource / failure-mode invariants. Each test below started life
as a reproducer for a defect found in review; they are inverted here
to pin the fixed behaviour, so a regression fails loudly rather than silently
returning the old shape. Every test that could hang runs the merge on a watchdog
thread and fails on timeout rather than blocking the suite.
"""

from __future__ import annotations

import contextlib
import gc
import json
import threading
import time
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import nova_bf.io as io_mod
import nova_bf.merge as merge_mod
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.io import Store
from nova_bf.results import build_result_table, partial_dir, result_name

K = 4


def _cfg(root: str, name: str = "test") -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=f"{root}/corpus"),
        queries=QueriesConfig(path=f"{root}/queries.parquet"),
        output=OutputConfig(path=root),
        searches=[SearchSpec(name=name, k=K)],
    )


def _write_partials(cfg, pdir, n_partials: int, n_queries: int = 8) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    qids = [f"q{i}" for i in range(n_queries)]
    score = 1000.0
    for p in range(n_partials):
        ids, scores = [], []
        for q in qids:
            ids.append([f"{q}_p{p}_{i}" for i in range(K)])
            scores.append([score := score - 1.0 for _ in range(K)])
        payload = {"src": [f"payload-{q}" for q in qids]}
        pq.write_table(build_result_table(qids, payload, ids, scores),
                       str(pdir / f"rank{p:03d}.parquet"), row_group_size=4)


def _run_with_timeout(fn, timeout: float):
    """Run `fn` on a thread. Returns ('ok', value) | ('raised', exc) | ('hung', None)."""
    box: list = []

    def target():
        try:
            box.append(("ok", fn()))
        except BaseException as exc:                     # noqa: BLE001
            box.append(("raised", exc))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return ("hung", None)
    return box[0]


def _reader_threads(exclude: set) -> list[threading.Thread]:
    return [t for t in threading.enumerate()
            if t not in exclude and t.is_alive()
            and t is not threading.current_thread()
            and "_read" in t.name]


# ---------------------------------------------------------------------------
# `Store.root` is scheme-stripped -- a standing trap, deliberately documented.
# ---------------------------------------------------------------------------

def test_store_root_is_scheme_stripped_and_is_not_reusable_as_a_uri():
    """NOT a fixed bug -- a live footgun in `Store` itself, kept as documentation.

    `_fs_and_path` returns pyarrow's path, which for s3 is `bucket/key` with the
    scheme gone. So `Store(other.root)` silently downgrades an S3 store to a
    LocalFileSystem rooted at `$CWD/bucket/key`. Round-tripping is idempotent on
    LOCAL roots, which is why no test caught `_reduce` doing exactly this.
    Anything rebuilding a Store from another must pass `.uri`, never `.root`.
    """
    out = Store("s3://bucket/prefix")
    assert out.is_s3 and out.root == "bucket/prefix"

    downgraded = Store(out.root)
    assert downgraded.is_s3 is False
    assert isinstance(downgraded.fs, pafs.LocalFileSystem)
    assert downgraded.root.endswith("bucket/prefix") and downgraded.root.startswith("/")

    # ...whereas the uri round-trips faithfully.
    assert Store(out.uri).is_s3 is True


def test_merge_against_an_s3_style_root_reads_its_own_partials(tmp_path, monkeypatch):
    """`_reduce` must build its reader Store from `out.uri`, not `out.root`.

    With `out.root` the reduce looked for every partial on the LOCAL filesystem
    and every S3 merge died with a FileNotFoundError naming a cwd-relative path.
    This drives a filesystem whose paths are shaped like pyarrow's s3 ones
    (`bucket/key`) so the scheme strip would be fatal, and asserts the merge
    completes and produces the right answer.
    """
    fake_s3 = tmp_path / "s3root"
    fake_s3.mkdir()
    real = io_mod._fs_and_path

    def fake_fs_and_path(uri: str):
        if uri.startswith("s3://"):
            return (pafs.SubTreeFileSystem(str(fake_s3), pafs.LocalFileSystem()),
                    uri[len("s3://"):])
        return real(uri)

    monkeypatch.setattr(io_mod, "_fs_and_path", fake_fs_and_path)

    cfg = _cfg("s3://bucket/prefix")
    pdir = fake_s3 / "bucket" / "prefix" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=3)

    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=60)
    assert kind == "ok", f"s3-shaped merge {kind}: {res!r}"

    merged = fake_s3 / "bucket" / "prefix" / result_name(cfg, cfg.searches[0])
    t = pq.read_table(str(merged)).to_pydict()
    assert len(t["query_id"]) == 8
    # every query's top-K is the global one: partial 0 holds the highest scores
    assert all(len(h) == K for h in t["hit_ids"])


# ---------------------------------------------------------------------------
# A failing reader must not strand the others.
# ---------------------------------------------------------------------------

def _leaky_scenario(tmp_path, monkeypatch, n_partials=8, window=2, slow=1.0,
                    fail_rank="rank000"):
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=n_partials)

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: window)

    real_read = Store.read_columns

    def faulty(self, read_path, columns):
        if fail_rank in read_path:
            raise OSError(f"injected read failure on {read_path}")
        time.sleep(slow)                 # keep the other readers in flight
        return real_read(self, read_path, columns)

    monkeypatch.setattr(Store, "read_columns", faulty)
    return cfg


def test_failed_reduce_surfaces_the_error_and_strands_no_reader_thread(
        tmp_path, monkeypatch):
    """The consumer must DRAIN all `len(partials)` items, never break early.

    Every reader enqueues exactly one item (a table, or a `None` sentinel on
    failure) and holds a semaphore permit until the consumer releases it. A
    consumer that stopped at the first sentinel left the rest blocked forever --
    those holding a permit in `q.put` (queue full), the rest in
    `window.acquire()` (no permit ever returned). Observed before the fix: 6 of
    8 reader threads still alive after the failure. `daemon=True` only defers
    that to interpreter exit; inside a live process they are simply lost.
    """
    n_partials, slow = 8, 1.0
    cfg = _leaky_scenario(tmp_path, monkeypatch, n_partials=n_partials,
                          window=2, slow=slow)

    before = set(threading.enumerate())
    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)

    assert kind == "raised", f"merge {kind}"
    assert isinstance(res, OSError) and "injected read failure" in str(res)

    time.sleep(slow * 3)                 # far longer than any injected sleep
    leaked = _reader_threads(before)
    assert not leaked, (
        f"{len(leaked)} reader threads survived the failed reduce "
        f"({[t.name for t in leaked]}); the consumer stopped draining")


def test_repeated_failed_merges_do_not_accumulate_threads(tmp_path, monkeypatch):
    """A long-lived process -- one `run_merge` per search, a retry loop, a
    notebook, this very test session -- used to gain ~W wedged threads per
    failure, forever. Thread count must be flat across repeated failures."""
    counts = []
    for attempt in range(3):
        sub = tmp_path / f"try{attempt}"
        sub.mkdir()
        cfg = _leaky_scenario(sub, monkeypatch, n_partials=8, window=2, slow=1.0)
        kind, _ = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
        assert kind == "raised"
        time.sleep(1.5)
        counts.append(threading.active_count())
    assert counts[-1] <= counts[0], f"thread count grew across failures: {counts}"


def test_failed_reduce_pins_no_partial_table_in_memory(tmp_path, monkeypatch):
    """Tables already read when the failure lands must become collectable.

    Under the early-break consumer they were not: the wedged reader threads kept
    the `_read` closure -- and with it the Queue and every table sitting in it --
    reachable, so `gc.collect()` could not free them. At production shape that is
    `window_n` whole parsed partials lost per failed search.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=200)

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 3)

    refs: list[weakref.ref] = []
    real_read = Store.read_columns

    def faulty(self, read_path, columns):
        if "rank000" in read_path:
            raise OSError("injected")     # fails first
        time.sleep(0.3)                   # 1 and 2 are still in flight then
        tbl = real_read(self, read_path, columns)
        try:
            refs.append(weakref.ref(tbl))
        except TypeError:
            pytest.skip("pyarrow.Table is not weak-referenceable on this build")
        return tbl

    monkeypatch.setattr(Store, "read_columns", faulty)
    kind, _ = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
    assert kind == "raised"

    time.sleep(1.0)
    gc.collect()
    alive = [r for r in refs if r() is not None]
    assert refs, "no partial was read — the test proves nothing"
    assert not alive, (
        f"{len(alive)}/{len(refs)} partial tables still resident after the "
        "reduce raised — the abandoned queue is still reachable")


def test_leaked_permits_do_not_wedge_a_later_merge(tmp_path, monkeypatch):
    """`q` and `window` are `_reduce` locals, so even a badly-behaved reduce
    cannot strand permits anything else can see. A subsequent merge in the same
    process must still complete."""
    bad = tmp_path / "bad"
    bad.mkdir()
    cfg_bad = _leaky_scenario(bad, monkeypatch, n_partials=8, window=2, slow=1.0)
    assert _run_with_timeout(lambda: merge_mod.run_merge(cfg_bad), 120)[0] == "raised"

    monkeypatch.undo()                    # restore the real Store.read_columns
    good = tmp_path / "good"
    good.mkdir()
    cfg_good = _cfg(str(good / "out"))
    _write_partials(cfg_good, good / "out" / partial_dir(cfg_good, cfg_good.searches[0]),
                    n_partials=4)
    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg_good), timeout=60)
    assert kind == "ok", f"a later merge was wedged: {kind} {res!r}"


def test_consumer_side_failure_also_drains_and_joins_promptly(tmp_path, monkeypatch):
    """A raise from the CONSUMER must leave the reader threads as clean as a
    raise from a reader does, and must not stall on `join(timeout=30)`."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)
    qids = [f"q{i}" for i in range(8)]
    for p in range(3):
        order = qids if p != 1 else list(reversed(qids))   # partial 1 misaligned
        pq.write_table(
            build_result_table(order, {"src": [f"pay-{q}" for q in order]},
                               [[f"{q}_p{p}"] for q in order],
                               [[100.0 - p] for _ in order]),
            str(pdir / f"rank{p:03d}.parquet"))

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 1)
    real_read = Store.read_columns

    def slow(self, read_path, columns):
        if "rank002" in read_path:
            time.sleep(0.2)
        return real_read(self, read_path, columns)

    monkeypatch.setattr(Store, "read_columns", slow)

    before = set(threading.enumerate())
    t0 = time.perf_counter()
    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
    elapsed = time.perf_counter() - t0

    assert kind == "raised" and "not row-aligned" in str(res)
    assert elapsed < 5.0, f"took {elapsed:.1f}s to surface a consumer-side error"
    assert not _reader_threads(before), "consumer-side raise stranded readers"


# ---------------------------------------------------------------------------
# `head` must not pin the partial it came from.
# ---------------------------------------------------------------------------

def test_head_copies_query_id_and_payload_instead_of_slicing_the_partial(
        tmp_path, monkeypatch):
    """`head[bi]` must hold a COPY of query_id + payload only.

    It used to hold `tbl.slice(...)`. An Arrow slice is a view, so that pinned
    partial 0's whole table -- hit columns included -- for the entire reduce,
    long past the point its window permit was handed back at merge.py's
    `window.release()`. The window bounds `window x partial` and never
    accounted for the extra one.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=3, n_queries=200)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 1)

    built: list[tuple[str, ...]] = []
    real_table = merge_mod.pa.table

    def spy(mapping, *a, **kw):
        tbl = real_table(mapping, *a, **kw)
        built.append(tuple(tbl.schema.names))
        return tbl

    monkeypatch.setattr(merge_mod.pa, "table", spy)
    merge_mod.run_merge(cfg)

    # a projection carrying exactly query_id + payload was materialised...
    assert ("query_id", "src") in built, built
    # ...and nothing that `head` keeps carries hit columns. Scoped to tables
    # keyed by query_id: the spy patches the pyarrow MODULE, so it also sees
    # tables built inside other modules -- `build_ordinals` makes an {id, pos}
    # one on its CPU path -- which are not heads and are absent on a GPU box.
    heads = [s for s in built if "hit_ids" not in s and "query_id" in s]
    assert heads and all(set(s) == {"query_id", "src"} for s in heads), heads

    # The rationale, pinned: a slice really would have retained the parent.
    parent = pq.read_table(str(pdir / "rank000.parquet"))
    sl, parent_bytes = parent.slice(0, parent.num_rows), parent.nbytes
    del parent
    gc.collect()
    assert sl.nbytes == parent_bytes


# ---------------------------------------------------------------------------
# `ranged_get` concurrency must be divided by the window.
# ---------------------------------------------------------------------------

def test_ranged_get_pool_is_divided_by_the_window(tmp_path, monkeypatch):
    """`_ranged_download` builds a fresh ThreadPoolExecutor PER FILE, so W
    in-flight partials multiply it: 24 x 16 = 384 concurrent range GETs, with
    `params.io_workers` -- which capped exactly this in the deleted
    `_prefetch_all` -- no longer consulted. `_reduce` must divide the per-file
    pool by its own window."""
    cfg = _cfg(str(tmp_path / "out"))
    cfg.params.merge_ranged_reads = True
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=4, n_queries=500)

    monkeypatch.setattr(io_mod, "_RANGED_GET_MIN_BYTES", 1)
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 4096)
    window_n = 4
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: window_n)

    concurrencies: list[int] = []
    real_dl = Store._ranged_download

    def spy(self, read_path, size):
        concurrencies.append(self.ranged_get_concurrency)
        return real_dl(self, read_path, size)

    monkeypatch.setattr(Store, "_ranged_download", spy)
    merge_mod.run_merge(cfg)

    assert concurrencies, "ranged_get path never ran"
    # `max(1, ...)`, not `max(2, ...)`: a floor of 2 breached the pool above
    # window_n=12. This agreed with the code only because window_n is 4 here.
    expected = max(1, merge_mod._RANGED_GET_POOL // window_n)
    pool = merge_mod._RANGED_GET_POOL

    # THE FIRST PARTIAL IS DELIBERATELY AN EXCEPTION: it gets the whole pool,
    # because nothing can be folded until one partial has fully arrived and an
    # evenly divided pool makes every file take the same time whether it is
    # alone or not. Exactly one reader may do this.
    assert concurrencies.count(pool) == 1, (
        f"expected exactly one full-pool reader (the first): {concurrencies}")
    rest = [c for c in concurrencies if c != pool]
    assert set(rest) == {expected}, rest

    # STEADY STATE stays within the pool; the first partial's overshoot is
    # bounded by one file and ends as soon as it lands.
    assert window_n * expected <= pool


# ---------------------------------------------------------------------------
# Batch sizing must match the fold that actually happens.
# ---------------------------------------------------------------------------

def test_auto_batch_size_matches_the_two_way_fold():
    """The grid is `B x 2k`, not `B x W x k`: the reduce folds partial by
    partial, so `_topk_merge` sees the running state plus one partial however
    many partials exist. Sizing off `n_partials` was right for the old lockstep
    loop and is 32x too small at W=64 — 32x the fold calls, and one parquet row
    group per batch in the artifact people consume."""
    n_partials, k, n_rows = 64, 1000, 100_000
    rows = merge_mod._resolve_batch_rows(None, n_rows, k)
    slots = rows * 2 * k
    assert slots <= merge_mod._TARGET_CANDIDATE_SLOTS
    assert slots > merge_mod._TARGET_CANDIDATE_SLOTS * 0.9, (
        f"batch_rows={rows} uses only {slots/1e6:.2f} M of the "
        f"{merge_mod._TARGET_CANDIDATE_SLOTS/1e6:.0f} M target")
    # This used to assert the same call with a different partial count returned
    # the same rows. The fan-in is no longer a parameter, so that comparison is
    # now `rows == rows`; independence is structural rather than tested.


def test_fold_is_never_wider_than_two_and_does_not_fragment_the_output(
        tmp_path, monkeypatch):
    """Every `_topk_merge` call is the seed (one list) or a 2-way fold, and the
    output keeps one row group per batch — so the batch must be sized for the
    real grid, not W x k."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=4, n_queries=40)

    widths: list[int] = []
    real = merge_mod._topk_merge

    def spy(score_lists, id_lists, tie_lists, k, **kw):
        # **kw so the spy survives optional arguments; this test
        # pins the fold WIDTH, not `_topk_merge`'s signature.
        widths.append(len(score_lists))
        return real(score_lists, id_lists, tie_lists, k, **kw)

    monkeypatch.setattr(merge_mod, "_topk_merge", spy)
    merge_mod.run_merge(cfg)

    assert widths and set(widths) <= {1, 2}, f"fold widths seen: {sorted(set(widths))}"
    assert 1 in widths, "the single-partial seed must go THROUGH _topk_merge"
    md = pq.ParquetFile(f"{cfg.output.path}/{result_name(cfg, cfg.searches[0])}").metadata
    assert md.num_rows == 40
    # auto sizing covers all 40 rows in one batch -> one row group, not 20
    assert md.num_row_groups == 1, f"{md.num_row_groups} row groups for 40 rows"


# ---------------------------------------------------------------------------
# Row alignment across partials.
# ---------------------------------------------------------------------------

def test_row_misaligned_partials_are_rejected(tmp_path):
    """Partials are row-aligned by query, and the reduce reads payload from
    partial 0 only. A partial whose rows are in a different order therefore
    folds its hits into the WRONG queries and the output looks entirely normal.
    The old lockstep loop compared query_id across all W partials per batch;
    that check was dropped in the rewrite and must stay restored."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)

    qids = ["q0", "q1", "q2", "q3"]
    for p, order in enumerate([qids, list(reversed(qids))]):
        pq.write_table(
            build_result_table(order, {"src": [f"pay-{q}" for q in order]},
                               [[f"{q}_hit_p{p}"] for q in order],
                               [[100.0 + p] for _ in order]),
            str(pdir / f"rank{p:03d}.parquet"))

    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
    assert kind == "raised", f"misaligned partials merged without complaint ({kind})"
    assert "not row-aligned" in str(res), res


# ---------------------------------------------------------------------------
# Review, 2026-09-05: resource / lifetime / shutdown.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window_n", [1, 2, 3, 4, 8, 12, 13, 16, 24, 25, 64])
def test_range_gets_per_file_shrink_as_the_window_grows(window_n):
    """The per-file pool is DIVIDED by the window, because `_ranged_download`
    builds one PER FILE -- so the two cannot both be large.

    The old list ([1, 4, 12, 13, 16]) was exhaustive of the domain when the
    window came from a byte budget clamped to 16. The window is an operator
    knob with no ceiling now, so this covers past the pool size too.

    Above `_RANGED_GET_POOL` readers the division bottoms out at one GET each
    and the total tracks the window, which is the operator's choice to make:
    they asked for that many readers. What must not happen is per-file
    concurrency staying flat while the window grows, which is what a floor of
    2 did (32 outstanding GETs at window 16, against the 24 being divided).
    """
    per_file = max(1, merge_mod._RANGED_GET_POOL // window_n)
    if window_n <= merge_mod._RANGED_GET_POOL:
        assert window_n * per_file <= merge_mod._RANGED_GET_POOL, (
            f"{window_n} x {per_file} exceeds {merge_mod._RANGED_GET_POOL}")
    else:
        assert per_file == 1, "cannot give a reader less than one GET"


def test_zero_query_partials_are_refused_not_written_as_an_empty_file(tmp_path):
    """Zero queries means zero batches, so the writer was never created and
    `sink.close()` left a 0-byte file -- reported as success, unreadable
    afterwards ("Parquet file size is 0 bytes")."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=2, n_queries=0)
    with pytest.raises(RuntimeError, match="0 queries"):
        merge_mod.run_merge(cfg)


def test_a_failure_before_the_drain_does_not_strand_readers(tmp_path, monkeypatch):
    """`path`/`os.makedirs` used to sit between `t.start()` and the try that
    owns the drain, so anything raising there left every reader blocked on
    `window.acquire()` or `q.put()` with nobody to drain them -- each holding a
    parsed partial for the life of the process."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=200)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    def boom(*a, **kw):
        raise PermissionError("read-only output root")
    monkeypatch.setattr(merge_mod.os, "makedirs", boom)

    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(PermissionError):
        merge_mod.run_merge(cfg)
    time.sleep(0.5)
    leaked = [t for t in threading.enumerate()
              if t.ident not in before and t.is_alive() and "read" not in t.name.lower()
              or (t.ident not in before and t.is_alive())]
    assert not leaked, f"{len(leaked)} reader thread(s) stranded: {[t.name for t in leaked]}"


# ---------------------------------------------------------------------------
# Review round 2: defects introduced BY the round-1 fixes.
# ---------------------------------------------------------------------------

class _SliceBoom:
    """A table whose `.slice()` raises -- i.e. the fold fails BEFORE `sl` is
    bound, which is the one path where a `del sl` would explode."""

    def __init__(self, t):
        self._t = t

    def slice(self, *a, **kw):
        raise MemoryError("INJECTED before sl is bound")

    def __getattr__(self, n):
        return getattr(self._t, n)


def test_a_fold_failing_before_sl_is_bound_surfaces_the_real_error(tmp_path, monkeypatch):
    """`sl` is assigned INSIDE the try, so a failure on the first slice leaves
    it unbound. A `del sl` there raises UnboundLocalError out of the handler,
    abandoning the drain and stranding every reader -- re-creating the exact
    failure the drain was written to prevent, and hiding the real error."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=200)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    real = Store.read_columns
    monkeypatch.setattr(Store, "read_columns",
                        lambda self, p, c: _SliceBoom(real(self, p, c)))

    before = {t.ident for t in threading.enumerate()}
    t0 = time.monotonic()
    with pytest.raises(MemoryError, match="INJECTED"):
        merge_mod.run_merge(cfg)
    assert time.monotonic() - t0 < 25, "drain was abandoned; readers hit the join grace"
    time.sleep(0.3)
    leaked = [t for t in threading.enumerate()
              if t.ident not in before and t.is_alive()]
    assert not leaked, f"stranded readers: {[t.name for t in leaked]}"


def test_a_consumer_failure_keeps_its_location(tmp_path, monkeypatch):
    """The traceback is dropped so it cannot pin a multi-GB partial, but the
    location has to survive as text -- otherwise every consumer-side failure
    reports only the `raise errors[0]` line, and an unexpected error deep in the
    fold becomes unlocatable."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=3, n_queries=50)

    real = merge_mod._topk_merge

    def boom(*a, **kw):
        raise ValueError("INJECTED deep in the fold")
    monkeypatch.setattr(merge_mod, "_topk_merge", boom)

    with pytest.raises(ValueError, match="INJECTED") as ei:
        merge_mod.run_merge(cfg)
    notes = "\n".join(getattr(ei.value, "__notes__", []))
    assert "_fold" in notes or "_topk_merge" in notes, notes
    assert "merge.py" in notes, notes


def _no_leaked_readers(before, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        leaked = [t for t in threading.enumerate()
                  if t.ident not in before and t.is_alive()]
        if not leaked:
            return []
        time.sleep(0.1)
    return leaked


def test_an_interrupt_in_the_drain_still_releases_every_reader(tmp_path, monkeypatch):
    """Ctrl-C lands in `q.get()`, outside anything the loop catches. Only the
    consumer returns window permits, so an abandoned drain wedges every reader
    on `window.acquire()` for the life of the process -- each holding a parsed
    partial (a large allocation at production shape)."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=8, n_queries=150)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    real_q = merge_mod.Queue

    class Rude(real_q):
        n = 0

        def get(self, *a, **kw):
            item = super().get(*a, **kw)
            type(self).n += 1
            if type(self).n == 2:
                raise KeyboardInterrupt("user pressed ctrl-c")
            return item

    monkeypatch.setattr(merge_mod, "Queue", Rude)
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(KeyboardInterrupt):
        merge_mod.run_merge(cfg)
    assert not _no_leaked_readers(before), "readers stranded by the interrupt"


def test_a_thread_that_will_not_start_still_releases_the_ones_that_did(
        tmp_path, monkeypatch):
    """`t.start()` can raise -- a thread-capped container at W=64. It used to
    sit outside the try, so neither the drain nor the join ran and the readers
    already going were stranded with no diagnostic at all."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=150)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    real_start = threading.Thread.start
    state = {"n": 0}

    def flaky(self):
        state["n"] += 1
        if state["n"] == 4:
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", flaky)
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(RuntimeError, match="can't start new thread"):
        merge_mod.run_merge(cfg)
    assert not _no_leaked_readers(before), "readers stranded by the failed start"


def test_a_data_error_outranks_a_transient_read_error(tmp_path, monkeypatch):
    """`errors[0]` is first-append-wins and the two failures race: a reader can
    append its transient S3 error while the consumer is still inside the fold
    that is about to raise a row-misalignment. One is retriable and one is not,
    and reporting the wrong one sends the operator round the whole corpus again.

    The sleeps force the losing order deterministically -- the reader appends
    first, so a naive `errors[0]` yields the OSError.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=4, n_queries=100)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 4)

    real = Store.read_columns
    order: list[str] = []
    seen = {"n": 0}

    def flaky(self, path, cols):
        seen["n"] += 1
        if seen["n"] == 2:                 # a LATER reader, so partial 0 folds
            time.sleep(0.05)
            order.append("read")
            raise OSError("transient S3 slowdown")
        return real(self, path, cols)

    def slow_bad_fold(*a, **kw):
        time.sleep(0.30)                   # still folding when the reader fails
        order.append("fold")
        raise RuntimeError("partials are not row-aligned")

    monkeypatch.setattr(Store, "read_columns", flaky)
    monkeypatch.setattr(merge_mod, "_topk_merge", slow_bad_fold)

    with pytest.raises(RuntimeError, match="row-aligned"):
        merge_mod.run_merge(cfg)
    assert order[:2] == ["read", "fold"], f"ordering not forced: {order}"


def test_hit_id_output_type_does_not_follow_the_partials(tmp_path):
    """`_take_ids` gathers from the partials' own buffers, so the output type
    followed the input -- but a batch with NO hits falls back to an empty
    large_string. Mixing the two inside one merge aborts `ParquetWriter` mid
    write ("Table schema does not match schema used to create file"). It also
    keeps int64 character offsets, which a `string` output would not."""
    q, k = 4, 2
    # rows 2 and 3 have no hits at all -> the empty-batch fallback
    offs = pa.array([0, 2, 4, 4, 4], pa.int32())
    tbl = pa.table({
        "query_id": pa.array([f"q{i}" for i in range(q)]),
        "hit_ids": pa.ListArray.from_arrays(
            offs, pa.array(["a", "b", "c", "d"], pa.string())),   # NOT large_string
        "hit_scores": pa.ListArray.from_arrays(
            offs, pa.array([4.0, 3.0, 2.0, 1.0], pa.float32())),
    })
    cfg = _cfg(str(tmp_path / "out"))
    cfg.searches[0].k = k
    cfg.params.merge_batch_size = 2                      # forces a hits/no-hits split
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(pdir / "rank000.parquet"), compression="snappy")

    out = merge_mod.run_merge(cfg)
    got = pq.read_table(out[cfg.searches[0].name])
    assert got.schema.field("hit_ids").type == pa.list_(pa.large_string()), \
        got.schema.field("hit_ids").type
    assert got.column("hit_ids").to_pylist() == [["a", "b"], ["c", "d"], [], []]


def test_a_null_hit_scores_row_says_so(tmp_path):
    """`value_lengths()` on a null LIST entry yields NaN -> INT64_MIN, and the
    failure then surfaced from `np.repeat` as "repeats may not contain negative
    values", nowhere near the cause."""
    offs = pa.array([0, 2, 2, 4], pa.int32())
    null_row = pa.array([False, True, False])             # row 1 is a NULL list
    ids = pa.ListArray.from_arrays(
        offs, pa.array(["a", "b", "c", "d"], pa.large_string()), mask=null_row)
    sc = pa.ListArray.from_arrays(
        offs, pa.array([4.0, 3.0, 2.0, 1.0], pa.float32()), mask=null_row)
    tbl = pa.table({"query_id": pa.array(["q0", "q1", "q2"]),
                    "hit_ids": ids, "hit_scores": sc})
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(pdir / "rank000.parquet"), compression="snappy")

    with pytest.raises(RuntimeError, match="null"):
        merge_mod.run_merge(cfg)


# ---------------------------------------------------------------------------
# External review (GPT), 2026-09-06.
# ---------------------------------------------------------------------------

def _stamped(cfg, pdir, ranks, num_jobs, n_queries=8, tiebreak=None,
             run_key=True, names=None):
    """Partials carrying explicit run/rank metadata, one file per rank."""
    from nova_bf.results import (CONFIG_KEY, JOB_RANK_KEY, NUM_JOBS_KEY,
                                 RUN_KEY, TIEBREAK_KEY, config_identity,
                                 run_identity)
    pdir.mkdir(parents=True, exist_ok=True)
    spec = cfg.searches[0]
    csha = config_identity(cfg, spec)
    qids = [f"q{i}" for i in range(n_queries)]
    for r in ranks:
        tb = tiebreak or cfg.params.tiebreak
        rsha = run_identity(csha, "corpus", num_jobs, None, tb)
        ids = [[f"q_r{r}_{i}" for i in range(K)] for _ in qids]
        sc = [[float(K - i) for i in range(K)] for _ in qids]
        t = build_result_table(qids, {"src": [f"p-{q}" for q in qids]}, ids, sc)
        meta = {
            CONFIG_KEY: csha.encode(), NUM_JOBS_KEY: str(num_jobs).encode(),
            JOB_RANK_KEY: str(r).encode(), TIEBREAK_KEY: tb.encode(),
        }
        if run_key:
            meta[RUN_KEY] = rsha.encode()
        t = t.replace_schema_metadata(meta)
        pq.write_table(t, str(pdir / (names or {}).get(r, f"rank{r:03d}.parquet")))


@pytest.mark.parametrize("ranks,ok", [
    ([0, 1, 2, 3], True),
    ([0, 1, 2, 3, 4], False),      # a SUPERSET: rank 4 the run never declared
    ([0, 1, 2], False),            # the missing-rank case, already covered
])
def test_rank_set_must_be_exactly_zero_to_num_jobs(tmp_path, ranks, ok):
    """The docstring claims "the ranks present are exactly 0..num_jobs-1", but
    the check only looked for MISSING ranks -- so a directory holding an extra
    rank from a wider run passed, folding a corpus slice the run never declared
    into exact ground truth."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]), ranks, 4)
    if ok:
        merge_mod.run_merge(cfg)
    else:
        with pytest.raises(RuntimeError, match="ranks"):
            merge_mod.run_merge(cfg)


def test_searches_reduced_under_different_tiebreaks_are_refused(tmp_path):
    """Each search validates its own run fingerprint, but that fingerprint is
    per-(run, SEARCH) -- so two searches from different runs both pass. The
    tie-break rule IS run-global, and mixing rules puts hits decided by
    different rules in one artifact."""
    cfg = _cfg(str(tmp_path / "out"))
    cfg.searches.append(SearchSpec(name="second", k=K))
    root = tmp_path / "out"
    _stamped(cfg, root / partial_dir(cfg, cfg.searches[0]), [0, 1], 2,
             tiebreak="ordinal")
    _stamped(cfg, root / partial_dir(cfg, cfg.searches[1]), [0, 1], 2,
             tiebreak="id")
    with pytest.raises(RuntimeError, match="tie-break"):
        merge_mod.run_merge(cfg)


def test_float64_scores_are_refused_not_rounded(tmp_path):
    """The candidate grid is float32, so a float64 column is silently DOWNCAST
    into it -- scores differing below float32 resolution collapse into a tie and
    the rounded value is reported as the score."""
    q = 3
    offs = pa.array([0, 2, 4, 6], pa.int32())
    tbl = pa.table({
        "query_id": pa.array([f"q{i}" for i in range(q)]),
        "hit_ids": pa.ListArray.from_arrays(
            offs, pa.array([f"d{i}" for i in range(6)], pa.large_string())),
        "hit_scores": pa.ListArray.from_arrays(
            offs, pa.array([1.0000000002, 1.0000000001] * 3, pa.float64())),
    })
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(pdir / "rank000.parquet"))
    with pytest.raises(RuntimeError, match="float32"):
        merge_mod.run_merge(cfg)


def test_a_failed_write_leaves_no_result_under_the_canonical_name(
        tmp_path, monkeypatch):
    """A half-written parquet at the canonical name is worse than none: it
    opens, it looks like a result, and it is silently short. Anything finding
    results by filename rather than through the merge manifest consumes it."""
    cfg = _cfg(str(tmp_path / "out"))
    cfg.params.merge_batch_size = 2
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=2, n_queries=8)

    real = pq.ParquetWriter.write_table
    state = {"n": 0}

    def flaky(self, table, *a, **kw):
        state["n"] += 1
        if state["n"] == 2:                     # first batch lands, second dies
            raise OSError("object store went away mid-write")
        return real(self, table, *a, **kw)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", flaky)
    out_path = tmp_path / "out" / result_name(cfg, cfg.searches[0])
    with pytest.raises(OSError, match="went away"):
        merge_mod.run_merge(cfg)
    assert not out_path.exists(), "a truncated result was left behind"


def test_a_folded_partial_is_freed_before_its_window_permit_is_returned(
        tmp_path, monkeypatch):
    """`sl = tbl.slice(...)` is a zero-copy VIEW, so `del tbl` frees nothing
    while it is bound. Releasing the permit there admits the next partial on top
    of the previous one -- one whole partial beyond the window budget at
    production shape. It was cleared only on the fold-FAILURE path.

    At the release point: under one partial resident with the fix, nearly two
    without.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    qids = [f"q{i}" for i in range(4000)]
    for p in range(3):
        ids = [[f"<urn:uuid:{p:04d}{i:032d}>" for i in range(K)] for _ in qids]
        sc = [[float(K - i) for i in range(K)] for _ in qids]
        pq.write_table(build_result_table(qids, {"src": qids}, ids, sc),
                       str(pdir / f"rank{p:03d}.parquet"))
    one = pq.read_table(str(pdir / "rank000.parquet")).nbytes

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 1)
    samples: list[int] = []
    real_sem = merge_mod.Semaphore

    class Probe(real_sem):
        def release(self, *a, **kw):
            samples.append(pa.total_allocated_bytes())
            return super().release(*a, **kw)

    monkeypatch.setattr(merge_mod, "Semaphore", Probe)
    base = pa.total_allocated_bytes()
    merge_mod.run_merge(cfg)

    peak = (max(samples) - base) / one
    assert peak < 1.5, (
        f"{peak:.2f} partials resident when the window permit was returned; "
        "a folded partial is still being held by its slice")


def test_missing_ranks_are_caught_even_without_a_run_fingerprint(tmp_path):
    """The no-RUN_KEY path returned early, skipping the config AND rank checks
    -- so 2 partials of a 4-rank run merged clean, with half the corpus silently
    absent, while `num_jobs`/`job_rank` sat unread in their metadata."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]),
             [0, 1], 4, run_key=False)
    with pytest.raises(RuntimeError, match="declared 4 ranks"):
        merge_mod.run_merge(cfg)


def test_a_complete_run_without_a_run_fingerprint_still_merges(tmp_path):
    """The fall-through must not turn the legacy warning into a refusal:
    hours of legitimate GPU work should not be stranded by a missing stamp."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]),
             [0, 1, 2, 3], 4, run_key=False)
    merge_mod.run_merge(cfg)


def test_a_filename_disagreeing_with_its_rank_metadata_is_refused(tmp_path):
    """The rank set is otherwise a property of metadata alone -- the filename is
    used only for ordering, so a file whose name and stamp disagree means one of
    them was rewritten and the rank set cannot be trusted."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]),
             [0, 1, 2, 3], 4, names={3: "rank000.parquet", 0: "rank003.parquet"})
    with pytest.raises(RuntimeError, match="metadata disagree"):
        merge_mod.run_merge(cfg)


def test_a_non_integer_rank_names_the_search_and_the_file(tmp_path):
    """It used to exit as a bare `ValueError: invalid literal for int()`."""
    from nova_bf.results import JOB_RANK_KEY
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _stamped(cfg, pdir, [0, 1], 2)
    f = pdir / "rank001.parquet"
    t = pq.read_table(str(f))
    md = dict(t.schema.metadata)
    md[JOB_RANK_KEY] = b"not-an-int"
    pq.write_table(t.replace_schema_metadata(md), str(f))
    with pytest.raises(RuntimeError, match="not an integer"):
        merge_mod.run_merge(cfg)


def test_the_artifact_records_how_many_ranks_produced_it(tmp_path):
    """Otherwise the run's shape lives only in the merge manifest, which a later
    merge into the same output path overwrites -- leaving a finished file that
    cannot describe itself."""
    from nova_bf.results import NUM_JOBS_KEY
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]), [0, 1, 2], 3)
    out = merge_mod.run_merge(cfg)[cfg.searches[0].name]
    md = pq.ParquetFile(out).schema_arrow.metadata or {}
    assert md.get(NUM_JOBS_KEY) == b"3", md


@pytest.mark.parametrize("col", ["hit_scores", "hit_ids", "hit_tie"])
def test_a_null_inside_a_hit_list_is_refused(tmp_path, col):
    """Nulls in the values CHILD, not the list rows. Each corrupts silently and
    differently: a null score becomes NaN and is dropped as though it were
    padding (the candidate vanishes); a null hit_tie casts to INT64_MIN, the
    BEST possible tiebreak, so it beats every real hit; a null id is shipped as
    a `None` hit id."""
    offs = pa.array([0, 3], pa.int32())
    ids = ["a", None, "c"] if col == "hit_ids" else ["a", "b", "c"]
    sc = [3.0, None, 1.0] if col == "hit_scores" else [3.0, 2.0, 1.0]
    cols = {
        "query_id": pa.array(["q0"]),
        "hit_ids": pa.ListArray.from_arrays(offs, pa.array(ids, pa.large_string())),
        "hit_scores": pa.ListArray.from_arrays(offs, pa.array(sc, pa.float32())),
    }
    if col == "hit_tie":
        cols["hit_tie"] = pa.ListArray.from_arrays(
            offs, pa.array([5, None, 7], pa.int64()))
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), str(pdir / "rank000.parquet"))
    with pytest.raises(RuntimeError, match=f"{col} has 1 null value"):
        merge_mod.run_merge(cfg)


# ---------------------------------------------------------------------------
# The device fold must agree with the host path, candidate for candidate.
# ---------------------------------------------------------------------------

def _rand_lists(rng, b, k, dup_ids=0, tie_frac=0.0):
    """(scores, ids) ListArrays with controllable exact ties and duplicate ids."""
    ids = [f"<urn:uuid:{i:08x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(b * k)]
    if dup_ids:
        for i in range(dup_ids):
            ids[-(i + 1)] = ids[i]            # exact duplicate ids across rows
    sc = rng.random(b * k).astype(np.float32)
    if tie_frac:
        n_t = int(b * k * tie_frac)
        sc[:n_t] = np.float32(0.5)            # exact float32 ties
    off = pa.array(np.arange(b + 1, dtype=np.int32) * k)
    return (pa.ListArray.from_arrays(off, pa.array(sc, pa.float32())),
            pa.ListArray.from_arrays(off, pa.array(ids, pa.large_string())))


@pytest.mark.parametrize("n_inputs,b,k,dup,tie", [
    (2, 40, 10, 0, 0.0),      # the ordinary 2-way fold
    (2, 40, 10, 0, 0.35),     # heavy exact score ties -> ids decide
    (2, 25, 8, 12, 0.5),      # duplicate ids AND ties: input order must decide
    (1, 30, 12, 0, 0.0),      # the seed fold (n=1, width == k)
    (4, 20, 6, 0, 0.25),      # multi-way
])
def test_the_device_fold_matches_the_host_fold_exactly(
    monkeypatch, n_inputs, b, k, dup, tie
):
    """Same winners, same order, same scores — this is ground truth.

    Both paths are driven over the SAME inputs and compared element-wise. The
    device path skips the host candidate grid entirely, so a disagreement here
    is a wrong top-K, not a performance difference.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    lists = [_rand_lists(rng, b, k, dup_ids=dup, tie_frac=tie)
             for _ in range(n_inputs)]
    scores = [s for s, _ in lists]
    ids = [i for _, i in lists]

    # `NOVA_BF_MERGE_FOLD=cpu` is the documented way to run the torch fold
    # without a GPU; it makes `_fold_device()` hand back a CPU device, which
    # is all the device path needs.
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")

    # PROVE the device path ran. It DECLINES (returns None) on variable-width
    # ids or an all-empty batch, and a decline routes the batch to the host path --
    # so without this spy both sides of the comparison could be the same code
    # and the test would pass while proving nothing.
    took = []
    real = merge_mod._device_fold
    monkeypatch.setattr(merge_mod, "_device_fold",
                        lambda *a, **kw: (lambda r: (took.append(r is not None), r)[1])(real(*a, **kw)))
    dev_ids, dev_sc, dev_tie = merge_mod._topk_merge(scores, ids, None, k)
    assert took == [True], f"the device fold did not run: {took}"

    # Force the host path for the same inputs by refusing the dense fold.
    monkeypatch.setattr(merge_mod, "_device_fold", lambda *a, **kw: None)
    host_ids, host_sc, host_tie = merge_mod._topk_merge(scores, ids, None, k)

    assert dev_ids.to_pylist() == host_ids.to_pylist(), "winning ids differ"
    assert dev_sc.to_pylist() == host_sc.to_pylist(), "winning scores differ"
    assert dev_tie is None and host_tie is None


def _spy_device_fold(monkeypatch):
    """Record whether each `_device_fold` call took the batch."""
    took = []
    real = merge_mod._device_fold

    def spy(*a, **kw):
        r = real(*a, **kw)
        took.append(r is not None)
        return r

    monkeypatch.setattr(merge_mod, "_device_fold", spy)
    return took


def _ragged_input(rng, b, k, tag, *, empty_frac=0.2, ties=False, pool=None):
    """One partial with ragged rows: 0..k hits per row, some rows empty."""
    lens = rng.integers(0, k + 1, size=b).astype(np.int32)
    lens[rng.random(b) < empty_frac] = 0
    tot = int(lens.sum())
    off = pa.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    if ties:
        sc = rng.choice(np.array([0.25, 0.5, 0.75], np.float32), size=tot)
    else:
        sc = rng.random(tot).astype(np.float32)
    # Each row's hits sorted descending, as compute writes them.
    sc = np.concatenate([np.sort(sc[o:o + n])[::-1] for o, n in
                         zip(np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)]
                        or [np.empty(0, np.float32)]).astype(np.float32)
    if pool is not None:
        ids = [pool[i] for i in rng.integers(0, len(pool), size=tot)]
    else:
        ids = [f"<urn:uuid:{tag:04x}{i:04x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(tot)]
    return (pa.ListArray.from_arrays(off, pa.array(sc, pa.float32())),
            pa.ListArray.from_arrays(off, pa.array(ids, pa.large_string())))


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# The device fold on the CPU everywhere, and on a real GPU where there is one.
FOLD_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not _cuda_ok(), reason="needs CUDA"))]


@pytest.mark.parametrize("fold_dev", FOLD_DEVICES)
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ties,dups", [(False, False), (True, False), (True, True)])
def test_the_device_fold_takes_ragged_rows_and_matches_numpy(
        monkeypatch, seed, ties, dups, fold_dev):
    """Short rows (trimmed partials, queries with < k hits) take the device
    fold and give exactly the NumPy fold's answer: same ids, same scores, same
    order, same per-row counts -- including heavy exact ties and ids that
    repeat across and within inputs, where only the tie order decides."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(1000 + seed + 100 * (2 * ties + dups))
    b = int(rng.integers(1, 40))
    k = int(rng.integers(1, 12))
    n = int(rng.integers(2, 7))     # >= 2 inputs, so rows can exceed k
    pool = ([f"<urn:uuid:{i:08x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(7)]
            if dups else None)
    ins = [_ragged_input(rng, b, k, w, ties=ties, pool=pool) for w in range(n)]
    if sum(len(s.flatten()) for s, _ in ins) == 0:
        pytest.skip("drew an all-empty batch")
    scores, ids = [s for s, _ in ins], [i for _, i in ins]

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold_dev)
    took = _spy_device_fold(monkeypatch)
    d_ids, d_sc, _ = merge_mod._topk_merge(scores, ids, None, k)
    assert took == [True], "the device fold declined ragged rows"

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h_ids, h_sc, _ = merge_mod._topk_merge(scores, ids, None, k)
    assert d_sc.to_pylist() == h_sc.to_pylist()
    assert d_ids.to_pylist() == h_ids.to_pylist()


@pytest.mark.parametrize("fold_dev", FOLD_DEVICES)
@pytest.mark.parametrize("widths,k", [((3, 5, 0, 2), 5), ((4, 4), 4), ((1,), 3), ((6, 6, 6), 6)])
def test_uniform_width_inputs_take_the_reshape_layout_and_match_numpy(
        monkeypatch, widths, k, fold_dev):
    """Every row of an input holding the same count (full-width partials, or
    any constant width) builds the grid by reshape + concatenation instead of
    a scatter. Same answer as NumPy, including widths that differ per input,
    an all-empty input in the middle, and widths below and at k."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(sum(widths) * 31 + k)
    b = 17
    ins = []
    for w, width in enumerate(widths):
        lens = np.full(b, width, dtype=np.int32)
        off = pa.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
        tot = int(lens.sum())
        sc = rng.choice(np.array([0.1, 0.2, 0.3], np.float32), size=tot)
        ins.append((pa.ListArray.from_arrays(off, pa.array(sc, pa.float32())),
                    pa.ListArray.from_arrays(off, pa.array(
                        [f"<urn:uuid:{w:04x}{i:04x}-aaaa-bbbb-cccc-{i:012x}>"
                         for i in range(tot)], pa.large_string()))))
    scores, ids = [s for s, _ in ins], [i for _, i in ins]
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold_dev)
    took = _spy_device_fold(monkeypatch)
    d_ids, d_sc, _ = merge_mod._topk_merge(scores, ids, None, k)
    assert took == [True]
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h_ids, h_sc, _ = merge_mod._topk_merge(scores, ids, None, k)
    assert d_sc.to_pylist() == h_sc.to_pylist()
    assert d_ids.to_pylist() == h_ids.to_pylist()


def _with_ties(rng, sc, *, dup_frac=0.0):
    """A `hit_tie` column for `sc`: compute's file*1e8 + row layout, with an
    optional share of values repeated (only ever equal within a test, but the
    fold must still resolve them exactly as the host does)."""
    n = len(sc.flatten())
    tie = rng.integers(0, 50, n) * 100_000_000 + rng.integers(0, 10_000_000, n)
    if dup_frac and n:
        m = rng.random(n) < dup_frac
        tie[m] = tie[rng.integers(0, n, int(m.sum()))] if n else tie[m]
    return pa.ListArray.from_arrays(sc.offsets, pa.array(tie.astype(np.int64), pa.int64()))


def _var_width_ids(sc, tag):
    """IDs of differing byte widths: no lanes, so ties-mode gathers Arrow."""
    n = len(sc.flatten())
    return pa.ListArray.from_arrays(sc.offsets, pa.array(
        [f"d{tag}-{i}" for i in range(n)], pa.large_string()))


@pytest.mark.parametrize("fold_dev", FOLD_DEVICES)
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ids_kind", ["fixed", "variable"])
@pytest.mark.parametrize("ties,dup_frac", [(False, 0.0), (True, 0.0), (True, 0.3)])
def test_the_device_fold_takes_hit_tie_and_matches_numpy(
        monkeypatch, seed, ids_kind, ties, dup_frac, fold_dev):
    """`tiebreak: ordinal`: the device fold ranks the `hit_tie` VALUES and
    gives exactly the NumPy fold's ids, scores, ties, order and counts --
    ragged rows, heavy score ties, repeated tie values, and ids of any width
    (fixed-width or not: ties do not need the ids to pack)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(5000 + seed + 97 * (2 * ties + (dup_frac > 0)))
    b = int(rng.integers(1, 40))
    k = int(rng.integers(1, 12))
    n = int(rng.integers(2, 7))
    ins = [_ragged_input(rng, b, k, w, ties=ties) for w in range(n)]
    if sum(len(s.flatten()) for s, _ in ins) == 0:
        pytest.skip("drew an all-empty batch")
    scores = [s for s, _ in ins]
    ids = ([i for _, i in ins] if ids_kind == "fixed"
           else [_var_width_ids(s, w) for w, s in enumerate(scores)])
    tl = [_with_ties(rng, s, dup_frac=dup_frac) for s in scores]

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold_dev)
    took = _spy_device_fold(monkeypatch)
    d_ids, d_sc, d_tie = merge_mod._topk_merge(scores, ids, tl, k)
    assert took == [True], "the device fold declined a hit_tie batch"

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h_ids, h_sc, h_tie = merge_mod._topk_merge(scores, ids, tl, k)
    assert d_sc.to_pylist() == h_sc.to_pylist()
    assert d_ids.to_pylist() == h_ids.to_pylist()
    assert d_tie.to_pylist() == h_tie.to_pylist()


@pytest.mark.parametrize("seed", range(6))
def test_the_host_torch_tie_path_still_matches_numpy(monkeypatch, seed):
    """With ties on the device fold, nothing else exercises the host-grid
    torch path for `hit_tie` (`_fold_torch`, ties ranked by `_rank_dense`) --
    which still runs whenever the device fold declines. Force it."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(700 + seed)
    b, k, n = int(rng.integers(1, 30)), int(rng.integers(1, 9)), int(rng.integers(2, 5))
    ins = [_ragged_input(rng, b, k, w, ties=True) for w in range(n)]
    scores, ids = [s for s, _ in ins], [i for _, i in ins]
    tl = [_with_ties(rng, s, dup_frac=0.3) for s in scores]
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    monkeypatch.setattr(merge_mod, "_device_fold", lambda *a, **kw: None)
    merge_mod._reset_fold_used()
    t_ids, t_sc, t_tie = merge_mod._topk_merge(scores, ids, tl, k)
    assert merge_mod._FOLD_USED == {"torch:cpu"}, merge_mod._FOLD_USED
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h_ids, h_sc, h_tie = merge_mod._topk_merge(scores, ids, tl, k)
    assert t_sc.to_pylist() == h_sc.to_pylist()
    assert t_ids.to_pylist() == h_ids.to_pylist()
    assert t_tie.to_pylist() == h_tie.to_pylist()


def test_an_oom_assembling_device_winners_falls_back_to_the_host(monkeypatch):
    """The winners' lane gather in `_assemble` runs on the device; an OOM
    there must take the host path (inputs are untouched), not abort."""
    pytest.importorskip("torch")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(3)
    b, k = 12, 5
    ins = [_ragged_input(rng, b, k, w) for w in range(3)]
    tl = [_with_ties(rng, s) for s, _ in ins]
    real = merge_mod._assemble
    calls = []

    def flaky(*a, **kw):
        calls.append(kw.get("lanes_win") is not None or kw.get("lanes_fallback") is not None)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(*a, **kw)

    monkeypatch.setattr(merge_mod, "_assemble", flaky)
    merge_mod._reset_fold_used()
    got = merge_mod._topk_merge([s for s, _ in ins], [i for _, i in ins], tl, k,
                                lanes_mode=True)
    assert len(calls) == 2, "the host path did not run after the OOM"
    assert merge_mod._DEVICE_FOLDS[0] == 0, "a failed device fold was counted"
    monkeypatch.setattr(merge_mod, "_assemble", real)
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    want = merge_mod._topk_merge([s for s, _ in ins], [i for _, i in ins], tl, k)
    assert merge_mod._as_ids_array(got[0]).to_pylist() == want[0].to_pylist()
    assert got[1].to_pylist() == want[1].to_pylist()
    assert got[2].to_pylist() == want[2].to_pylist()


def test_a_tie_grid_too_wide_to_rank_declines(monkeypatch):
    """`_rank_dense` packs ranks into 32 bits over the PADDED grid; a ragged
    batch whose grid exceeds that must decline, not raise (NumPy can still
    do it). Simulated by shrinking the limit check's arithmetic input."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    sc, ids = _ragged_input(rng, 6, 4, 0, empty_frac=0.0)
    tl = [_with_ties(rng, sc)]
    real_stack = np.stack

    def wide_stack(arrs, *a, **kw):
        out = real_stack(arrs, *a, **kw)
        return out * 0 + 0x7FFFFFFF if out.ndim == 2 and out.dtype == np.int64 else out

    monkeypatch.setattr(merge_mod.np, "stack", wide_stack)
    assert merge_mod._device_fold([sc], [ids], 4, torch.device("cpu"), tie_lists=tl) is None


def test_hit_tie_fold_matches_a_naive_reference(monkeypatch):
    """Independent of both folds: per row, sort by (score desc, tie asc)."""
    pytest.importorskip("torch")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(91)
    b, k = 30, 6
    ins = [_ragged_input(rng, b, k, w, ties=True) for w in range(4)]
    tl = [_with_ties(rng, s) for s, _ in ins]
    took = _spy_device_fold(monkeypatch)
    out_ids, out_sc, out_tie = merge_mod._topk_merge(
        [s for s, _ in ins], [i for _, i in ins], tl, k)
    assert took == [True]
    for r in range(b):
        cand = []
        for (s, i), t in zip(ins, tl):
            cand += list(zip(s[r].as_py(), t[r].as_py(), i[r].as_py()))
        cand.sort(key=lambda c: (-c[0], c[1]))
        want = cand[:k]
        assert out_sc[r].as_py() == [c[0] for c in want], r
        assert out_tie[r].as_py() == [c[1] for c in want], r
        assert out_ids[r].as_py() == [c[2] for c in want], r


@pytest.mark.parametrize("fold_dev", FOLD_DEVICES)
def test_a_hit_tie_lane_state_keeps_the_device_fold(monkeypatch, fold_dev):
    """The ordinal `_reduce` shape: lanes on, the seed fold STARTS the lane
    state (`want_lanes`), later folds carry it with ragged groups, and the
    winning ties travel with their ids. Equals the NumPy chain."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(57)
    b, k = 25, 8
    groups = []
    for g in range(5):
        grp = [_ragged_input(rng, b, k // 3, 10 * g + w) for w in range(3)]
        groups.append([(s, i, _with_ties(rng, s)) for s, i in grp])

    def chain(lanes_mode):
        state = None
        lazy = []
        for grp in groups:
            scs, idss, tis = (list(x) for x in zip(*grp))
            if state is not None:
                lazy.append(isinstance(state[0], merge_mod._LazyIds))
                scs, idss, tis = [state[1]] + scs, [state[0]] + idss, [state[2]] + tis
            state = merge_mod._topk_merge(scs, idss, tis, k, lanes_mode=lanes_mode)
        return state, lazy

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold_dev)
    took = _spy_device_fold(monkeypatch)
    d, lazy = chain(True)
    assert took and all(took), f"a fold declined: {took}"
    assert lazy and all(lazy), f"the state did not stay in lanes: {lazy}"

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h, _ = chain(False)
    assert merge_mod._as_ids_array(d[0]).to_pylist() == h[0].to_pylist()
    assert d[1].to_pylist() == h[1].to_pylist()
    assert d[2].to_pylist() == h[2].to_pylist()


def test_a_lane_state_meeting_variable_width_ids_declines_with_ties(monkeypatch):
    """A lane state can only fold with ids of its own width; a partial whose
    ids are not one fixed width must send the batch to the host path (which
    materialises the state), not mix representations."""
    pytest.importorskip("torch")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(12)
    b, k = 10, 4
    s0, i0 = _dense_pair(rng, b, k, 0)
    t0 = _with_ties(rng, s0)
    st = merge_mod._topk_merge([s0], [i0], [t0], k, lanes_mode=True)
    assert isinstance(st[0], merge_mod._LazyIds)
    s1, _ = _ragged_input(rng, b, k, 1)
    i1, t1 = _var_width_ids(s1, 1), _with_ties(rng, s1)
    took = _spy_device_fold(monkeypatch)
    d = merge_mod._topk_merge([st[1], s1], [st[0], i1], [st[2], t1], k, lanes_mode=True)
    assert took == [False]
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h = merge_mod._topk_merge([s0, s1], [i0, i1], [t0, t1], k)
    assert merge_mod._as_ids_array(d[0]).to_pylist() == h[0].to_pylist()
    assert d[1].to_pylist() == h[1].to_pylist()
    assert d[2].to_pylist() == h[2].to_pylist()


def test_the_device_fold_matches_a_naive_reference_on_ragged_rows(monkeypatch):
    """An independent check that does not share code with either fold: per
    row, sort every candidate by (score desc, id asc) and keep k."""
    pytest.importorskip("torch")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(77)
    b, k = 30, 6
    ins = [_ragged_input(rng, b, k, w, ties=True) for w in range(4)]
    took = _spy_device_fold(monkeypatch)
    out_ids, out_sc, _ = merge_mod._topk_merge(
        [s for s, _ in ins], [i for _, i in ins], None, k)
    assert took == [True]
    for r in range(b):
        cand = []
        for s, i in ins:
            cand += list(zip(s[r].as_py(), i[r].as_py()))
        cand.sort(key=lambda t: (-t[0], t[1]))
        want = cand[:k]
        assert out_sc[r].as_py() == [c[0] for c in want], r
        assert out_ids[r].as_py() == [c[1] for c in want], r


@pytest.mark.parametrize("fold_dev", FOLD_DEVICES)
def test_a_ragged_lane_state_keeps_the_device_fold(monkeypatch, fold_dev):
    """The `_reduce` shape with trimmed partials (lanes are on whenever the ids
    are fixed width): a lane state whose rows are SHORT (an earlier fold of
    short partials), plus a group of short Arrow partials. Every fold must take the device path, and the chain must equal
    the NumPy fold of the same chain."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(31)
    b, k = 25, 8
    groups = [[_ragged_input(rng, b, k // 3, 10 * g + w) for w in range(3)]
              for g in range(5)]

    ragged_states = []

    def chain(lanes_mode):
        state = None
        for grp in groups:
            scs = [s for s, _ in grp]
            idss = [i for _, i in grp]
            if state is not None:
                if isinstance(state[0], merge_mod._LazyIds):
                    ragged_states.append(state[0].dense_k is None)
                scs, idss = [state[1]] + scs, [state[0]] + idss
            state = merge_mod._topk_merge(scs, idss, None, k, lanes_mode=lanes_mode)
        return state

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", fold_dev)
    took = _spy_device_fold(monkeypatch)
    d = chain(True)
    assert took and all(took), f"a fold declined: {took}"
    assert isinstance(d[0], merge_mod._LazyIds)
    assert sum(ragged_states) >= 2, (
        f"fixture should fold ragged lane states: {ragged_states}")

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h = chain(False)
    assert merge_mod._as_ids_array(d[0]).to_pylist() == h[0].to_pylist()
    assert d[1].to_pylist() == h[1].to_pylist()


def test_the_device_fold_takes_an_empty_arrow_input_beside_a_lane_state(monkeypatch):
    """A partial whose rows are ALL empty (a trimmed rank that contributed
    nothing to this batch) has no id width of its own; it must not make the
    device fold decline, nor shift any other input's ids."""
    pytest.importorskip("torch")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(8)
    b, k = 10, 4
    s0, i0 = _dense_pair(rng, b, k, 0)
    lazy_ids, lazy_sc, _ = merge_mod._topk_merge([s0], [i0], None, k, lanes_mode=True)
    off0 = pa.array(np.zeros(b + 1, dtype=np.int32))
    s_empty = pa.ListArray.from_arrays(off0, pa.array([], pa.float32()))
    i_empty = pa.ListArray.from_arrays(off0, pa.array([], pa.large_string()))
    s2, i2 = _ragged_input(rng, b, k, 2)

    took = _spy_device_fold(monkeypatch)
    d = merge_mod._topk_merge([lazy_sc, s_empty, s2], [lazy_ids, i_empty, i2],
                              None, k, lanes_mode=True)
    assert took == [True]
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    h = merge_mod._topk_merge([s0, s_empty, s2], [i0, i_empty, i2], None, k)
    assert merge_mod._as_ids_array(d[0]).to_pylist() == h[0].to_pylist()
    assert d[1].to_pylist() == h[1].to_pylist()


def test_the_device_fold_declines_an_all_empty_batch(monkeypatch):
    """No candidates at all: nothing to rank, so decline and let the host
    path return the empty rows."""
    torch = pytest.importorskip("torch")
    b, k = 6, 3
    off0 = pa.array(np.zeros(b + 1, dtype=np.int32))
    sc = pa.ListArray.from_arrays(off0, pa.array([], pa.float32()))
    ids = pa.ListArray.from_arrays(off0, pa.array([], pa.large_string()))
    assert merge_mod._device_fold([sc, sc], [ids, ids], k,
                                        torch.device("cpu")) is None
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    out_ids, out_sc, _ = merge_mod._topk_merge([sc, sc], [ids, ids], None, k)
    assert out_ids.to_pylist() == [[]] * b and out_sc.to_pylist() == [[]] * b


def test_an_empty_input_of_another_string_type_does_not_crash_the_device_fold(monkeypatch):
    """An all-empty `string` input beside `large_string` ones: it adds no ids,
    so the winners' gather must leave it out rather than chunk two string
    types together (ArrowTypeError). The old dense-only path declined this."""
    pytest.importorskip("torch")
    F = pa.list_(pa.float32())
    s1 = pa.array([[0.9, 0.1], [0.5]], F)
    i1 = pa.array([["aa", "bb"], ["cc"]], pa.list_(pa.large_string()))
    s2 = pa.array([[], []], F)
    i2 = pa.array([[], []], pa.list_(pa.string()))
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    took = _spy_device_fold(monkeypatch)
    ids, sc, _ = merge_mod._topk_merge([s1, s2], [i1, i2], None, 2)
    assert took == [True]
    assert ids.to_pylist() == [["aa", "bb"], ["cc"]]
    assert sc.to_pylist() == [[pytest.approx(0.9), pytest.approx(0.1)], [0.5]]


def test_the_device_fold_declines_more_candidates_than_its_ordinals_hold(monkeypatch):
    """Past the 32-bit ordinal (and int32 permutation) range the device fold
    must hand the batch to the host path, not wrap."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    sc, ids = _ragged_input(rng, 8, 4, 0, empty_frac=0.0)
    n = len(sc.flatten())
    monkeypatch.setattr(merge_mod, "_DEVICE_FOLD_MAX_CANDIDATES", n - 1)
    assert merge_mod._device_fold([sc], [ids], 4, torch.device("cpu")) is None
    monkeypatch.setattr(merge_mod, "_DEVICE_FOLD_MAX_CANDIDATES", n)
    assert merge_mod._device_fold([sc], [ids], 4, torch.device("cpu")) is not None


def test_the_device_fold_declines_variable_width_ids(monkeypatch):
    """Without fixed-width ids there is no lane ranking, so no device grid."""
    torch = pytest.importorskip("torch")
    b, k = 10, 4
    off = pa.array(np.arange(b + 1, dtype=np.int32) * k)
    sc = pa.ListArray.from_arrays(
        off, pa.array(np.arange(b * k, dtype=np.float32), pa.float32()))
    ids = pa.ListArray.from_arrays(
        off, pa.array([f"id{i}" for i in range(b * k)], pa.large_string()))
    assert merge_mod._device_fold([sc], [ids], k, torch.device("cpu")) is None


@pytest.mark.parametrize("special,where,why", [
    (float("-inf"), 3,  "-inf is how both paths mark PADDING, and a real "
                        "score can be -inf too; both must drop it the same way "
                        "(the valid mask keeps only scores > -inf)"),
    (float("inf"),  5,  "+inf is a VALID hit and must survive the valid mask"),
    (float("nan"),  7,  "NaN must sort BELOW every real candidate, matching the "
                        "NumPy semantics `_fold_packed` forces with SENTINEL_KEY"),
])
def test_the_device_fold_handles_special_scores_like_the_host(
    monkeypatch, special, where, why
):
    """Exactly the values where a dense fast path quietly disagrees."""
    torch = pytest.importorskip("torch")
    b, k, n = 16, 6, 2
    rng = np.random.default_rng(5)
    scores, ids = [], []
    for w in range(n):
        sc = rng.random(b * k).astype(np.float32)
        sc[where::(b * k // 3 or 1)] = special      # sprinkle it across rows
        off = pa.array(np.arange(b + 1, dtype=np.int32) * k)
        scores.append(pa.ListArray.from_arrays(off, pa.array(sc, pa.float32())))
        ids.append(pa.ListArray.from_arrays(off, pa.array(
            [f"<urn:uuid:{w:04x}{i:04x}-aaaa-bbbb-cccc-{i:012x}>"
             for i in range(b * k)], pa.large_string())))

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    took = []
    real = merge_mod._device_fold
    monkeypatch.setattr(merge_mod, "_device_fold",
                        lambda *a, **kw: (lambda r: (took.append(r is not None), r)[1])(real(*a, **kw)))
    d_ids, d_sc, _ = merge_mod._topk_merge(scores, ids, None, k)
    assert took == [True], "device fold did not run"

    monkeypatch.setattr(merge_mod, "_device_fold", lambda *a, **kw: None)
    h_ids, h_sc, _ = merge_mod._topk_merge(scores, ids, None, k)

    dv, hv = d_sc.to_pylist(), h_sc.to_pylist()
    assert [len(r) for r in dv] == [len(r) for r in hv], f"hit COUNTS differ: {why}"
    # NaN != NaN, so compare bit patterns rather than values.
    assert ([np.asarray(r, dtype=np.float32).tobytes() for r in dv]
            == [np.asarray(r, dtype=np.float32).tobytes() for r in hv]), why
    assert d_ids.to_pylist() == h_ids.to_pylist(), why


def test_the_device_fold_agrees_when_ids_repeat_inside_one_input(monkeypatch):
    """Duplicate ids WITHIN a single input, not just across inputs.

    Joint ranking gives equal ids equal lane values, so their relative order is
    decided by the stable sort over position. The device path builds that
    position mapping with a permute rather than a scatter, which is precisely
    where an off-by-one would hide.
    """
    torch = pytest.importorskip("torch")
    b, k = 20, 8
    rng = np.random.default_rng(3)
    pool = [f"<urn:uuid:{i:08x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(6)]
    ids_flat = [pool[i % len(pool)] for i in range(b * k)]      # heavy repeats
    sc = np.full(b * k, 0.25, dtype=np.float32)                 # everything ties
    sc[::3] = rng.random(len(sc[::3])).astype(np.float32)
    off = pa.array(np.arange(b + 1, dtype=np.int32) * k)
    s_arr = pa.ListArray.from_arrays(off, pa.array(sc, pa.float32()))
    i_arr = pa.ListArray.from_arrays(off, pa.array(ids_flat, pa.large_string()))

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    d = merge_mod._topk_merge([s_arr, s_arr], [i_arr, i_arr], None, k)
    monkeypatch.setattr(merge_mod, "_device_fold", lambda *a, **kw: None)
    h = merge_mod._topk_merge([s_arr, s_arr], [i_arr, i_arr], None, k)
    assert d[0].to_pylist() == h[0].to_pylist()
    assert d[1].to_pylist() == h[1].to_pylist()


def test_the_device_fold_handles_an_empty_batch(monkeypatch):
    """A tail batch can have zero rows; `b == 0` must not reach torch at all."""
    torch = pytest.importorskip("torch")
    k = 4
    off = pa.array(np.array([0], dtype=np.int32))
    s_arr = pa.ListArray.from_arrays(off, pa.array([], pa.float32()))
    i_arr = pa.ListArray.from_arrays(off, pa.array([], pa.large_string()))
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    ids, sc, tie = merge_mod._topk_merge([s_arr], [i_arr], None, k)
    assert len(ids) == 0 and len(sc) == 0 and tie is None


def _dense_pair(rng, b, k, seed):
    off = pa.array(np.arange(b + 1, dtype=np.int32) * k)
    sc = pa.array(rng.random(b * k).astype(np.float32), pa.float32())
    ids = pa.array([f"<urn:uuid:{seed:04x}{i:04x}-aaaa-bbbb-cccc-{i:012x}>"
                    for i in range(b * k)], pa.large_string())
    return pa.ListArray.from_arrays(off, sc), pa.ListArray.from_arrays(off, ids)


def test_the_device_fold_still_runs_once_the_state_carries_lanes(monkeypatch):
    """THE SHAPE OF EVERY FOLD AFTER THE SEED: a lane-carrying state plus an
    Arrow partial. The device path must TAKE it, not decline.

    A guard for "mixed lane/Arrow inputs" -- commented "never happens today" --
    declined exactly this, so the fast path ran once per batch and the other
    ten folds took the host scatter path after building lanes they discarded.
    Nothing caught it: the output stayed byte-identical and every test either
    used an Arrow state or checked only the seed. The merge just ran 2x slower.

    Asserting the RESULT is not enough here; the decline is invisible in the
    answer. This asserts the path.
    """
    pytest.importorskip("torch")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(4)
    b, k = 14, 6

    s0, i0 = _dense_pair(rng, b, k, 0)
    lazy_ids, lazy_sc, _ = merge_mod._topk_merge([s0], [i0], None, k, lanes_mode=True)
    assert isinstance(lazy_ids, merge_mod._LazyIds), "seed must produce lazy state"

    took = []
    real = merge_mod._device_fold
    monkeypatch.setattr(merge_mod, "_device_fold",
                        lambda *a, **kw: (lambda r: (took.append(r is not None), r)[1])(real(*a, **kw)))
    s1, i1 = _dense_pair(rng, b, k, 1)
    merge_mod._topk_merge([lazy_sc, s1], [lazy_ids, i1], None, k, lanes_mode=True)
    assert took == [True], (
        "the device fold DECLINED a lazy state + Arrow partial — that is every "
        "fold after the seed, so the fast path would be effectively dead")


def test_lazy_state_survives_a_batch_that_declines_the_device_fold(monkeypatch):
    """The state carries LANES; the general path speaks Arrow. A batch that
    declines must convert, not crash.

    The device fold used to decline whenever a row was shorter than k (it now
    declines on variable-width ids; this test forces the decline), and real
    partials contain queries that matched fewer than k documents. Caught only
    on real data — every synthetic fixture had full rows, so the whole local
    suite passed while the production merge died with
    `'_LazyIds' object has no attribute 'flatten'`.
    """
    pytest.importorskip("torch")
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(9)
    b, k = 12, 5

    s0, i0 = _dense_pair(rng, b, k, 0)
    lazy_ids, lazy_sc, _ = merge_mod._topk_merge([s0], [i0], None, k, lanes_mode=True)
    assert isinstance(lazy_ids, merge_mod._LazyIds)

    lens = np.full(b, k, dtype=np.int32)
    lens[4] = k - 3                                    # one ragged row
    off2 = pa.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    tot = int(lens.sum())
    s1 = pa.ListArray.from_arrays(
        off2, pa.array(rng.random(tot).astype(np.float32), pa.float32()))
    i1 = pa.ListArray.from_arrays(off2, pa.array(
        [f"<urn:uuid:ffff{i:04x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(tot)],
        pa.large_string()))

    # Ragged rows no longer decline, so force the decline this test is about.
    monkeypatch.setattr(merge_mod, "_device_fold", lambda *a, **kw: None)
    out_ids, out_sc, _ = merge_mod._topk_merge(
        [lazy_sc, s1], [lazy_ids, i1], None, k, lanes_mode=True)
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    ref_ids, ref_sc, _ = merge_mod._topk_merge([s0, s1], [i0, i1], None, k)
    assert merge_mod._as_ids_array(out_ids).to_pylist() == ref_ids.to_pylist()
    assert out_sc.to_pylist() == ref_sc.to_pylist()


def _ragged_lists(lens, k, seed=0):
    rng = np.random.default_rng(seed)
    off = pa.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    tot = int(np.sum(lens))
    sc = pa.ListArray.from_arrays(
        off, pa.array(rng.random(tot).astype(np.float32), pa.float32()))
    ids = pa.ListArray.from_arrays(off, pa.array(
        [f"<urn:uuid:{i:08x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(tot)],
        pa.large_string()))
    return sc, ids


@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_misaligned_id_rows_are_refused_on_every_fold_path(monkeypatch, backend):
    """`hit_ids` split differently from `hit_scores` must RAISE, not mispair.

    The guard lived inside the general path's scatter loop, so the device fast
    path returned before it ran: scores [[3,2],[1,.5]] with ids [[u1],[u2,u3,u4]]
    produced [[u1,u2],[u3,u4]] -- row 0's second hit carrying row 1's id. Wrong
    ground truth, silently, and only on the fast path.
    """
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", backend)
    k = 2
    sc, _ = _ragged_lists(np.array([2, 2], dtype=np.int32), k)
    off_bad = pa.array(np.array([0, 1, 4], dtype=np.int32))
    ids_bad = pa.ListArray.from_arrays(off_bad, pa.array(
        [f"<urn:uuid:{i:08x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(4)],
        pa.large_string()))
    with pytest.raises(RuntimeError, match="split differently|wrong"):
        merge_mod._topk_merge([sc], [ids_bad], None, k)


@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_a_null_score_inside_a_list_is_refused_on_every_fold_path(monkeypatch, backend):
    """A null score became NaN and was silently dropped by the `> -inf` mask on
    the device path, so the query lost a hit instead of the merge refusing."""
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", backend)
    k = 2
    off = pa.array(np.array([0, 2, 4], dtype=np.int32))
    sc = pa.ListArray.from_arrays(
        off, pa.array([0.9, None, 0.7, 0.6], pa.float32()))
    ids = pa.ListArray.from_arrays(off, pa.array(
        [f"<urn:uuid:{i:08x}-aaaa-bbbb-cccc-{i:012x}>" for i in range(4)],
        pa.large_string()))
    with pytest.raises(RuntimeError, match="null"):
        merge_mod._topk_merge([sc], [ids], None, k)


# ---------------------------------------------------------------------------
# The two merge-manifest layouts must not both describe the same parquet.
# ---------------------------------------------------------------------------

def _two_search_cfg(root: str) -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=f"{root}/corpus"),
        queries=QueriesConfig(path=f"{root}/queries.parquet"),
        output=OutputConfig(path=root),
        searches=[SearchSpec(name="alpha", k=K), SearchSpec(name="beta", k=K)],
    )


def _manifest_paths(cfg, root):
    from nova_bf import manifest as run_manifest

    whole = root / run_manifest.manifest_name(cfg, "merge")
    per = {s.name: root / run_manifest.manifest_name(cfg, "merge", search=s.name)
           for s in cfg.searches}
    return whole, per


def test_a_manifest_that_vanishes_mid_delete_is_not_reported_as_a_failure(
        tmp_path, monkeypatch, caplog):
    """Under `--jobs` every child deletes the same whole-run manifest.

    All but one lose the race between `get_file_info` and `delete_file` and see
    `FileNotFoundError`. The broad `except` logged "it may describe a run that is
    no longer on disk" -- alarming, and false: the file was correctly removed.
    """
    root = tmp_path / "out"
    root.mkdir(parents=True)
    victim = root / "_bf_manifest_gone_merge.json"
    victim.write_text("{}")
    out = Store(str(root))

    class RacingFS:
        # pyarrow filesystem methods are read-only, so wrap rather than patch.
        def __init__(self, inner):
            self._inner = inner

        def get_file_info(self, path):
            return self._inner.get_file_info(path)

        def delete_file(self, path):
            self._inner.delete_file(path)
            raise FileNotFoundError(path)      # as if a sibling got there first

    out.fs = RacingFS(out.fs)

    with caplog.at_level("WARNING"):
        merge_mod._drop_manifests(out, ["_bf_manifest_gone_merge.json"], "why")
    assert not victim.exists()
    assert caplog.records == []


# ---------------------------------------------------------------------------
# Round 3: the probe, the failed-write cleanup, and manifest coverage.
# ---------------------------------------------------------------------------

def test_the_lanes_probe_sees_every_row_not_just_the_first_batch(
        tmp_path, monkeypatch, caplog):
    """The fixed-width check is over EVERY row, so a partial prefix cannot
    decide it: rows 0..8999 with one id width and 500 later rows with another
    must keep the state on Arrow (lane state would pay a materialise per fold
    for a device fold that then declines every batch).

    (This used to be a DENSITY probe: lanes needed full-k rows. The device
    fold now takes ragged rows, so only the id width gates lanes.)
    """
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")   # the probe needs a device
    root = tmp_path / "out"
    cfg = _cfg(str(root))
    pdir = root / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)

    # `probe_rows` is `max(256, min(8192, 2_000_000 // k))` = 8192 at k=4, so
    # 9500 rows split [8192, 1308] and the 500 odd-width rows (9000..9499) sit
    # entirely in the SECOND slice. DO NOT SHRINK `n_head` below 8192 -- at
    # 2000 rows this becomes one slice, a head-only scan reaches the same
    # answer, and the test silently stops guarding the thing it is named for.
    # The unit-level version, with a multi-chunk fixture that does not depend
    # on the sizing, is in tests/test_decode_pool_and_lanes.py.
    n_head, n_late = 9000, 500
    qids = [f"q{i}" for i in range(n_head + n_late)]
    for p in range(2):
        ids, scores = [], []
        for i in range(len(qids)):
            fmt = "{:08x}-{:04x}{}" if i < n_head else "{:08x}-{:05x}{}"
            ids.append([f"<urn:uuid:{fmt.format(i, j, p)}>" for j in range(K)])
            scores.append([1.0 - j * 0.01 - p * 0.001 for j in range(K)])
        pq.write_table(build_result_table(qids, {}, ids, scores),
                       str(pdir / f"rank{p:03d}.parquet"))

    with caplog.at_level("INFO"):
        merge_mod.run_merge(cfg)
    log = caplog.text
    assert "not one fixed width" in log, (
        "the probe called a partial fixed-width whose later ids are not")
    assert "merge state ids: arrow" in log, log[-2000:]


@pytest.mark.parametrize("with_tie", [False, True])
def test_a_ragged_run_merge_takes_the_device_fold_every_time(
        tmp_path, monkeypatch, caplog, with_tie):
    """End to end: partials whose rows are short and uneven (the shape a
    trimmed or filtered search writes) keep lanes on and take the device
    fold on EVERY fold call, and the result equals the NumPy merge's --
    with ties by id, and with ties from a `hit_tie` column (ordinal)."""
    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "cpu")
    rng = np.random.default_rng(23)
    n_q, n_parts = 300, 5

    def write(root):
        cfg = _cfg(str(root))
        pdir = root / partial_dir(cfg, cfg.searches[0])
        pdir.mkdir(parents=True)
        r = np.random.default_rng(23)
        qids = [f"q{i}" for i in range(n_q)]
        for p in range(n_parts):
            ids, scores, ties = [], [], []
            for i in range(n_q):
                w = int(r.integers(0, K + 1))
                sc = np.sort(r.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=w))[::-1]
                scores.append([float(x) for x in sc])
                ids.append([f"<urn:uuid:{p:02x}{i:06x}-{j:04x}>" for j in range(w)])
                ties.append([int(x) for x in r.integers(0, 10**9, size=w)])
            pq.write_table(build_result_table(qids, {}, ids, scores,
                                              hit_tie=ties if with_tie else None),
                           str(pdir / f"rank{p:03d}.parquet"))
        return cfg

    def merged(root):
        from nova_bf import manifest as run_manifest

        cfg = write(root)
        paths = merge_mod.run_merge(cfg)
        name = cfg.searches[0].name
        tbl = pq.read_table(paths[name])
        man = [json.loads((root / run_manifest.manifest_name(
            cfg, "merge", search=name)).read_text())]
        return tbl, man

    with caplog.at_level("INFO"):
        dev_tbl, dev_man = merged(tmp_path / "dev")
    assert "merge state ids: device lanes" in caplog.text, caplog.text[-2000:]
    entries = [e for m in dev_man for e in m.get("searches", [])]
    assert entries, dev_man
    for e in entries:
        assert e["merge_fold_calls"] > 0
        assert e["merge_fold_device_calls"] == e["merge_fold_calls"], e

    monkeypatch.setenv("NOVA_BF_MERGE_FOLD", "numpy")
    host_tbl, host_man = merged(tmp_path / "host")
    for e in (e for m in host_man for e in m.get("searches", [])):
        assert e["merge_fold_device_calls"] == 0, e
    # The merged output carries no `hit_tie` (it is a merge-time key only).
    for col in ("query_id", "hit_ids", "hit_scores"):
        assert dev_tbl.column(col).to_pylist() == host_tbl.column(col).to_pylist(), col
    for e in entries:
        assert e["tiebreak_source"] == ("hit_tie" if with_tie else "hit_ids"), e


@pytest.mark.parametrize("scheme", ["local", "s3"])
def test_a_failed_write_removes_the_truncated_output_on_every_store(
        tmp_path, monkeypatch, scheme):
    """S3 is NOT special here, though this module long believed it was.

    Measured against a real S3 API (MinIO) with pyarrow's own
    `open_output_stream`: writing a partial payload and calling close() COMMITS
    it -- at 1 MB (single PutObject) and at 12 MB (multipart), over a previous
    object as readily as onto a fresh key, and even when the stream is merely
    abandoned. `sink.close()` in the merge's `finally` IS the commit and it runs
    unconditionally, so a failed merge leaves a truncated parquet under the
    canonical name on every store. Skipping the delete for S3 left it there, and
    the message told the operator the previous object was intact when it had in
    fact just been overwritten.
    """
    if scheme == "s3":
        fake_s3 = tmp_path / "s3root"
        fake_s3.mkdir()
        real_fs = io_mod._fs_and_path

        def fake_fs_and_path(uri: str):
            if uri.startswith("s3://"):
                return (pafs.SubTreeFileSystem(str(fake_s3),
                                               pafs.LocalFileSystem()),
                        uri[len("s3://"):])
            return real_fs(uri)

        monkeypatch.setattr(io_mod, "_fs_and_path", fake_fs_and_path)
        cfg = _cfg("s3://bucket/prefix")
        root = fake_s3 / "bucket" / "prefix"
    else:
        cfg = _cfg(str(tmp_path / "out"))
        root = tmp_path / "out"

    assert Store(cfg.output.path).is_s3 is (scheme == "s3")

    cfg.params.merge_batch_size = 2
    _write_partials(cfg, root / partial_dir(cfg, cfg.searches[0]),
                    n_partials=2, n_queries=8)
    out_path = root / result_name(cfg, cfg.searches[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"PREVIOUS GOOD RESULT")

    real = pq.ParquetWriter.write_table
    state = {"n": 0}

    def flaky(self, table, *a, **kw):
        state["n"] += 1
        if state["n"] == 2:                   # first batch lands, second dies
            raise OSError("object store went away mid-write")
        return real(self, table, *a, **kw)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", flaky)
    with pytest.raises(OSError, match="went away"):
        merge_mod.run_merge(cfg)
    assert not out_path.exists(), (
        f"{scheme}: a truncated result was left under the canonical output name")



@contextlib.contextmanager
def caplog_at(level):
    """A logging capture that works inside `pytest.raises`."""
    import logging

    class Sink(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

        @property
        def text(self):
            return "\n".join(r.getMessage() % () if not r.args else
                              r.getMessage() for r in self.records)

    sink = Sink()
    sink.setLevel(level)
    log = logging.getLogger("nova_bf.merge")
    log.addHandler(sink)
    try:
        yield sink
    finally:
        log.removeHandler(sink)


# ---------------------------------------------------------------------------
# Round 4: whether the COMMIT ran decides what survives, not whether an object
# was there before.
# ---------------------------------------------------------------------------

class _CloseFailsFS(pafs.FileSystemHandler):
    """A real pyarrow filesystem whose OUTPUT STREAM fails to close.

    A Python proxy will not do: the merge hands `out.fs` to `pq.ParquetFile`,
    which requires a genuine `FileSystem`. `PyFileSystem` over a handler is the
    only way to intercept `open_output_stream` and still be one.
    """

    def __init__(self, root):
        self._fs = pafs.SubTreeFileSystem(str(root), pafs.LocalFileSystem())

    def get_type_name(self):
        return "closefails"

    def __eq__(self, other):
        return isinstance(other, _CloseFailsFS) and other._fs == self._fs

    def get_file_info(self, paths):
        return self._fs.get_file_info(paths)

    def get_file_info_selector(self, selector):
        return self._fs.get_file_info(selector)

    def create_dir(self, path, recursive=True):
        self._fs.create_dir(path, recursive=recursive)

    def delete_dir(self, path):
        self._fs.delete_dir(path)

    def delete_dir_contents(self, path, missing_dir_ok=False):
        self._fs.delete_dir_contents(path, missing_dir_ok=missing_dir_ok)

    def delete_root_dir_contents(self):
        self._fs.delete_dir_contents("", accept_root_dir=True)

    def delete_file(self, path):
        self._fs.delete_file(path)

    def move(self, src, dest):
        self._fs.move(src, dest)

    def copy_file(self, src, dest):
        self._fs.copy_file(src, dest)

    def open_input_stream(self, path):
        return self._fs.open_input_stream(path)

    def open_input_file(self, path):
        return self._fs.open_input_file(path)

    def open_append_stream(self, path, metadata=None):
        return self._fs.open_append_stream(path, metadata=metadata)

    def normalize_path(self, path):
        return path

    def open_output_stream(self, path, metadata=None):
        import io

        class NoCommit(io.BytesIO):
            # NOTHING REACHES THE STORE. Writes go to a buffer that is thrown
            # away, because on real S3 an upload whose close() fails commits
            # nothing -- measured, the previous object survives at its original
            # size. Writing through to the local stand-in instead would
            # truncate it at open, and the test could then no longer tell
            # "correctly left the previous result alone" from "left a
            # truncated write behind".
            def close(self):
                raise OSError("object store went away during commit")

        return pa.PythonFile(NoCommit(), mode="w")


def test_an_s3_merge_that_fails_to_commit_keeps_the_previous_result(
        tmp_path, monkeypatch):
    """`sink.close()` raising on S3 means NOTHING was uploaded -- do not delete.

    Measured against a real S3 API (MinIO), writing a partial payload and then
    taking the store away so the commit fails:

        1 MB  over a previous 2 KB object -> close() RAISED -> previous, 2 KB
        12 MB over a previous 2 KB object -> close() RAISED -> previous, 2 KB
        1 MB  onto a fresh key            -> close() RAISED -> still NotFound

    An object-store outage fails the body AND the commit together, which is the
    commonest merge failure at the 153 GB shape. Deleting there destroys a
    finished ground-truth artifact this run never overwrote -- and if the delete
    itself then fails, tells the operator to remove it by hand.
    """
    fake_s3 = tmp_path / "s3root"
    fake_s3.mkdir()
    handler = _CloseFailsFS(fake_s3)
    real_fs = io_mod._fs_and_path

    def fake_fs_and_path(uri: str):
        if uri.startswith("s3://"):
            return pafs.PyFileSystem(handler), uri[len("s3://"):]
        return real_fs(uri)

    monkeypatch.setattr(io_mod, "_fs_and_path", fake_fs_and_path)
    cfg = _cfg("s3://bucket/prefix")
    root = fake_s3 / "bucket" / "prefix"
    _write_partials(cfg, root / partial_dir(cfg, cfg.searches[0]), n_partials=2)

    out_path = root / result_name(cfg, cfg.searches[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"PREVIOUS GOOD RESULT" * 100)

    with pytest.raises(OSError, match="went away"), caplog_at("ERROR") as log:
        merge_mod.run_merge(cfg)

    assert out_path.exists(), (
        "the merge deleted the output key although the commit never ran")
    assert out_path.read_bytes() == b"PREVIOUS GOOD RESULT" * 100, (
        "the previous, valid result was not left byte-for-byte intact")
    assert "before committing" in log.text
    assert "removed the incomplete" not in log.text


def test_a_footer_failure_is_not_an_uncommitted_upload(tmp_path, monkeypatch):
    """`writer.close()` and `sink.close()` are different events.

    Folding both into one `close_err` made a FOOTER failure look like an
    uncommitted upload. Measured against MinIO with the footer write
    interrupted: the sink still committed -- a 2982-byte non-parquet over a
    previous 2048-byte result -- and the merge declined to delete it AND told
    the operator the object there was the previous one, intact. Its mtime is
    fresh, so the check the message asks for confirms the lie.
    """
    fake_s3 = tmp_path / "s3root"
    fake_s3.mkdir()
    real_fs = io_mod._fs_and_path

    def fake_fs_and_path(uri: str):
        if uri.startswith("s3://"):
            return (pafs.SubTreeFileSystem(str(fake_s3), pafs.LocalFileSystem()),
                    uri[len("s3://"):])
        return real_fs(uri)

    monkeypatch.setattr(io_mod, "_fs_and_path", fake_fs_and_path)
    cfg = _cfg("s3://bucket/prefix")
    root = fake_s3 / "bucket" / "prefix"
    _write_partials(cfg, root / partial_dir(cfg, cfg.searches[0]), n_partials=2)
    out_path = root / result_name(cfg, cfg.searches[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"PREVIOUS GOOD RESULT" * 100)

    def bad_footer(self, *a, **kw):
        raise KeyboardInterrupt("interrupted writing the footer")

    monkeypatch.setattr(pq.ParquetWriter, "close", bad_footer)

    with pytest.raises(BaseException), caplog_at("ERROR") as log:
        merge_mod.run_merge(cfg)

    assert not out_path.exists(), (
        "a footerless object was committed and left under the canonical name")
    assert "removed the incomplete" in log.text





def test_a_failed_merge_removes_its_search_manifest_with_its_output(tmp_path):
    """A manifest for a parquet that is not there reads as complete.

    With one manifest per search, an EARLIER merge's record for this search is
    still on the prefix and it names the file the failure path just deleted.
    The two have to go together.
    """
    from nova_bf import manifest as run_manifest

    root = tmp_path / "out"
    cfg = _cfg(str(root))
    cfg.params.merge_batch_size = 2
    _write_partials(cfg, root / partial_dir(cfg, cfg.searches[0]),
                    n_partials=2, n_queries=8)
    merge_mod.run_merge(cfg)                       # a good run, with a manifest
    man = root / run_manifest.manifest_name(cfg, "merge", search="test")
    out_path = root / result_name(cfg, cfg.searches[0])
    assert man.exists() and out_path.exists()

    real = pq.ParquetWriter.write_table
    state = {"n": 0}

    def flaky(self, table, *a, **kw):
        state["n"] += 1
        if state["n"] == 2:                        # first batch lands, second dies
            raise OSError("object store went away mid-write")
        return real(self, table, *a, **kw)

    import unittest.mock as _mock
    with _mock.patch.object(pq.ParquetWriter, "write_table", flaky):
        with pytest.raises(OSError, match="went away"):
            merge_mod.run_merge(cfg)

    assert not out_path.exists(), "a truncated result was left behind"
    assert not man.exists(), (
        "the manifest outlived the output it describes, so the prefix now "
        "reads as a completed merge")
