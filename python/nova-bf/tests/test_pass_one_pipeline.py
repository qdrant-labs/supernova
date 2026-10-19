"""Pass-one pipelining (`params.pipeline_pass_one`) is a scheduling change only.

Slice N+1's pass one (phase A, `_twopass_launch`) is launched before slice N's
bookkeeping (phase B, `_twopass_prepare` and the fold). These tests pin:

  * OUTPUT: byte-identical to the unpipelined run in slice AND tiles mode --
    tiles mode is not byte-identical to one-pass, so this also pins that the
    pipelined run takes the very same per-slice decisions;
  * DECISIONS: the per-slice two-pass decision counters match the unpipelined
    run's exactly, through every fallback (pass-one OOM / failure in a
    launch, mid-run disable, crossover bail, certification refusal, a hint
    flipping between launch and decision);
  * ORDER: A(N+1) is issued before B(N), and B(N+1) reads thresholds that
    already include B(N)'s fold;
  * ACCOUNTING: every launch is consumed exactly once and the manifest's
    pass-one counter identity holds.

CPU through the two-pass CPU opt-in, at the fixture scale of
`test_twopass_tiles.py`; the CUDA side stream is covered by the last test,
which skips without a GPU.
"""

from __future__ import annotations

import contextlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf import compute
from nova_bf import twopass
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)

DIM, K, NQ = 64, 5, 48
TILE = 16

# Every per-slice DECISION the two-pass books, and every per-slice pass-one
# counter. Pipelining must never change these: a launch's own counter writes
# are held back and booked where the inline pass one would have made them.
DECISION_KEYS = (
    "slices_twopass", "slices_plain", "rows_full", "rows_live", "rows_padded",
    "gemms", "slices_tiles", "tile_pairs_full", "tile_pairs_live",
    "tile_rows_rescored", "tiles_crossover_bails", "tiles_fallbacks",
    "certify_inconclusive", "audit_slices", "audit_violations",
    "groups_seen", "groups_refused", "verifications",
    "slices_guard_refused", "last_refusal_reason", "slices_pass_one_oom",
    "tiles_bound_oom", "eps_evaluations", "eps_cache_hits",
)

# The ONLY counters a pipelined run may differ by, besides its own
# `pipeline_*` bookkeeping: a launch the decision then did not want (the hint
# flipped to plain, or the two-pass was disabled after it was issued) really
# ran a kernel, so it keeps its `slices_fused` / `slices_unfused` increment
# and books a `slices_discarded` -- the only extra pass-one work pipelining
# does. `_assert_identical` pins that invariant instead of equality.
LAUNCH_KEYS = ("slices_fused", "slices_unfused", "slices_discarded")
PIPELINE_KEYS = ("pipeline_launched", "pipeline_used", "pipeline_discarded",
                 "pipeline_retried", "pipeline_pass_one")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for k in ("NOVA_BF_TWOPASS_RESCORE", "NOVA_BF_NO_TWOPASS",
              "NOVA_BF_TWOPASS_NO_VERIFY", "NOVA_BF_NO_TILE_KERNEL",
              "NOVA_BF_TWOPASS_TILE_THRESHOLD", "NOVA_BF_PIPELINE_PASS_ONE",
              "NOVA_BF_TWOPASS_FORCE_ENGAGE"):
        monkeypatch.delenv(k, raising=False)
    twopass.reset()
    yield
    twopass.reset()


@pytest.fixture
def small(monkeypatch):
    """Fixture scale, as in `test_twopass_tiles.py`. Not correctness knobs."""
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 8)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 8)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 16)
    monkeypatch.setattr(twopass, "CPU_TILE", TILE)
    monkeypatch.setattr(twopass, "TILE_CROSSOVER", 1.0)
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_DEVICE", "cpu")


def _write(path, vectors, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(),
                                        type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture
def ds(tmp_path):
    """Several files (each its own batch group), a 37-row file smaller than
    any slice, partial last tiles at every batch size used, and exact
    duplicates of earlier rows (real ties) in a later file."""
    rng = np.random.default_rng(29)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    sizes = [200, 200, 37, 200, 144, 200, 200]
    g = 0
    first = None
    for fi, n in enumerate(sizes):
        vecs = rng.standard_normal((n, DIM)).astype(np.float16).astype(np.float32)
        if fi == 0:
            first = vecs.copy()
        if fi == 3:
            vecs[:40] = first[:40]
        _write(cdir / f"f{fi}.parquet", vecs,
               id=[f"c{g + r:06d}" for r in range(n)],
               lang=["eng" if (g + r) % 3 else "fra" for r in range(n)])
        g += n
    qpath = tmp_path / "queries.parquet"
    _write(qpath, rng.standard_normal((NQ, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(NQ)],
           qset=["a"] * (NQ // 2) + ["b"] * (NQ - NQ // 2))
    return {"cdir": str(cdir), "qpath": str(qpath)}


def _cfg(ds, out, metric, mode, pipeline, batch=40, filtered=True):
    searches = [
        SearchSpec(name="all", metric=metric, k=K),
        SearchSpec(name="half", metric=metric, k=K,
                   rows={"column": "qset", "isin": ["b"]}),
    ]
    if filtered:
        # A filtered member in the SAME score group as the unfiltered ones.
        searches.append(SearchSpec(
            name="eng", metric=metric, k=K,
            filter=Filter(must=[FilterCondition(field="lang", match="eng")])))
    params = dict(io_workers=1, dense_batch_size=batch,
                  pipeline_pass_one=pipeline)
    if mode == "off":
        params["two_pass"] = "off"
    else:
        params["two_pass_rescore"] = mode
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid",
                              payload_fields=["qset"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(**params),
        searches=searches,
    )


def _run(ds, out, metric, mode, pipeline, **kw):
    out.mkdir(parents=True, exist_ok=True)
    twopass.reset()
    paths = run_compute(_cfg(ds, out, metric, mode, pipeline, **kw),
                        num_jobs=1, job_rank=0)
    return {n: pq.read_table(p) for n, p in paths.items()}, twopass.stats()


def _assert_identical(on, off, st_on, st_off, label=""):
    for name in off:
        assert on[name].equals(off[name]), f"{label} {name}: output differs"
    for key in DECISION_KEYS:
        assert st_on[key] == st_off[key], (
            f"{label} {key}: {st_on[key]} pipelined vs {st_off[key]} "
            f"unpipelined -- a decision changed")
    # ... and every other counter, so a new one is covered by default.
    for key in st_off.keys() - set(LAUNCH_KEYS) - set(PIPELINE_KEYS):
        assert st_on[key] == st_off[key], (
            f"{label} {key}: {st_on[key]!r} pipelined vs {st_off[key]!r}")
    # The extra pass ones are exactly the discarded launches that ran one.
    extra = {k: st_on[k] - st_off[k] for k in LAUNCH_KEYS}
    assert min(extra.values()) >= 0, (label, extra)
    assert (extra["slices_fused"] + extra["slices_unfused"]
            == extra["slices_discarded"]), (label, extra)
    assert extra["slices_discarded"] <= (st_on["pipeline_discarded"]
                                         - st_on["pipeline_retried"]), (
        label, extra, st_on)
    _assert_accounting(st_on, st_off)
    assert st_off["pipeline_launched"] == 0, st_off


def _identity_gap(st):
    return (st["slices_fused"] + st["slices_unfused"]
            - st["slices_twopass"] - st["slices_discarded"])


def _assert_accounting(st, st_off=None):
    assert st["pipeline_launched"] == (st["pipeline_used"]
                                       + st["pipeline_discarded"]), st
    assert st["pipeline_retried"] <= st["pipeline_discarded"], st
    # The manifest's pass-one identity, in both runs. (A tiles-mode slice
    # whose pass one a guard refused or ran out of memory used to re-score
    # its all-live tiles and book a `slices_twopass` no kernel paid for; it
    # is now scored plain, as in slice mode.)
    assert _identity_gap(st) == 0, st
    if st_off is not None:
        assert _identity_gap(st_off) == 0, st_off


# --- 1. byte identity, end to end ---------------------------------------------


@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("batch", [40, 72])
@pytest.mark.parametrize("filtered", [True, False])
def test_pipelined_is_byte_identical_to_unpipelined(ds, tmp_path, small, mode,
                                                    metric, batch, filtered):
    off, st_off = _run(ds, tmp_path / "off", metric, mode, False,
                       batch=batch, filtered=filtered)
    on, st_on = _run(ds, tmp_path / "on", metric, mode, True,
                     batch=batch, filtered=filtered)
    assert st_on["slices_twopass"] > 0 and st_on["rows_live"] < st_on["rows_full"]
    if mode == "tiles":
        assert st_on["slices_tiles"] > 0, st_on
    assert st_on["pipeline_used"] > 0, (
        "no pipelined launch was ever used, so this proved nothing", st_on)
    assert st_on["pipeline_pass_one"] is True
    assert st_off["pipeline_pass_one"] is False
    _assert_identical(on, off, st_on, st_off, f"{mode}/{metric}/{batch}")


class _FlipHint(dict):
    """`_TP_LIVE_HINT` whose stored value cycles engage, engage, plain.

    Writes happen only in phase B, at the same points pipelined or not, so
    both runs see the same hint sequence -- but phase A reads it one slice
    early, so the gate disagrees between launch and decision both ways: a
    launch the decision does not want (discarded), and a decision that
    wants a pass one nothing launched (run inline)."""

    def __init__(self):
        super().__init__()
        self.writes = 0

    def __setitem__(self, key, value):
        self.writes += 1
        super().__setitem__(key, 2.0 if self.writes % 3 == 0 else 0.0)


@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_a_hint_that_flips_between_launch_and_decision(ds, tmp_path, small,
                                                       monkeypatch, mode):
    monkeypatch.setattr(compute, "_TP_LIVE_HINT", _FlipHint())
    off, st_off = _run(ds, tmp_path / "off", "cosine", mode, False)
    monkeypatch.setattr(compute, "_TP_LIVE_HINT", _FlipHint())
    on, st_on = _run(ds, tmp_path / "on", "cosine", mode, True)
    assert st_on["pipeline_used"] > 0, st_on
    # The decision said plain to a pass one launched on the stale hint.
    assert st_on["pipeline_discarded"] > st_on["pipeline_retried"], st_on
    # ... and engaged where the stale hint had launched nothing (inline).
    assert st_on["slices_twopass"] > st_on["pipeline_used"], st_on
    _assert_identical(on, off, st_on, st_off, mode)


# --- 2. fallbacks ---------------------------------------------------------------


def _fail_in_launch(monkeypatch, exc_factory):
    """Make pass one fail inside phase A only; inline pass ones still work."""
    state = {"in_launch": False, "fired": 0}
    real_launch = compute._twopass_launch
    real_rowmax = twopass.approx_rowmax

    def launch(*a, **kw):
        state["in_launch"] = True
        try:
            return real_launch(*a, **kw)
        finally:
            state["in_launch"] = False

    def rowmax(*a, **kw):
        if state["in_launch"]:
            state["fired"] += 1
            raise exc_factory()
        return real_rowmax(*a, **kw)

    monkeypatch.setattr(compute, "_twopass_launch", launch)
    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)
    return state


@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("kind", ["oom", "error"])
def test_a_failed_launch_reruns_pass_one_inline(ds, tmp_path, small,
                                                monkeypatch, mode, kind):
    """A pass one that cannot allocate (the pipelined run holds one more
    slice) or raises in phase A is retried inline by phase B -- where the
    unpipelined run ran it -- so every decision, and the output, is the
    unpipelined one."""
    off, st_off = _run(ds, tmp_path / "off", "dot", mode, False)
    exc = ((lambda: twopass.PassOneUnavailable("CUDA out of memory"))
           if kind == "oom" else (lambda: RuntimeError("launch boom")))
    state = _fail_in_launch(monkeypatch, exc)
    on, st_on = _run(ds, tmp_path / "on", "dot", mode, True)
    assert state["fired"] > 0
    assert st_on["pipeline_used"] == 0, st_on
    assert st_on["pipeline_retried"] == st_on["pipeline_launched"] > 0, st_on
    # The launch's OOM (or its `slices_unfused` before an error) is not
    # booked: the inline retry books its own, exactly as unpipelined.
    assert st_on["slices_pass_one_oom"] == st_off["slices_pass_one_oom"] == 0
    for key in LAUNCH_KEYS:
        assert st_on[key] == st_off[key], (key, st_on, st_off)
    _assert_identical(on, off, st_on, st_off, f"{mode}/{kind}")


# The real functions, so a helper installed once per run wraps them and not
# the previous run's wrapper.
_REAL = {
    "_upper_bounds": twopass._upper_bounds,
    "norm_guard": twopass.norm_guard,
    "approx_rowmax": twopass.approx_rowmax,
    "_twopass_launch": compute._twopass_launch,
    "_tp_discard": compute._tp_discard,
}


def _row_key(row) -> tuple:
    return tuple(float(v) for v in np.asarray(row, dtype=np.float32)[:8])


def _slice_keys(ds, starts):
    """Identity keys of the corpus slices starting at `starts`, a list of
    `(file index, row)`: their first row's leading values, as `_spy_slice`
    reads them off the slice tensor."""
    keys = set()
    for fi, r in starts:
        t = pq.read_table(f"{ds['cdir']}/f{fi}.parquet")
        keys.add(_row_key(t.column("dense_embedding")[r].as_py()))
    return keys


def _spy_slice(monkeypatch):
    """`cur["key"]`: the slice whose pass one (`_upper_bounds`, behind both
    `upper_bounds` and `upper_bounds_begin`) is running. Only meaningful
    before the generator's yield -- the guards and the row-max launch."""
    cur = {"key": None}
    real = _REAL["_upper_bounds"]

    def spy(Q, Cb, *a, **kw):
        cur["key"] = _row_key(Cb[0].float().numpy()) if Cb.shape[0] else None
        return (yield from real(Q, Cb, *a, **kw))

    monkeypatch.setattr(twopass, "_upper_bounds", spy)
    return cur


def _track_launches(monkeypatch):
    """Every launch record, and the ids of those discarded (booked)."""
    seen = {"recs": [], "discarded": set()}
    real_launch = compute._twopass_launch    # possibly another test wrapper
    real_discard = _REAL["_tp_discard"]

    def launch(*a, **kw):
        out = real_launch(*a, **kw)
        for rec in (out or {}).values():
            seen["recs"].append((rec, dict(rec.deltas)))
        return out

    def discard(rec, retried=False, count=True):
        if count and rec.cert_key is not None:
            seen["discarded"].add(id(rec))
        return real_discard(rec, retried=retried, count=count)

    monkeypatch.setattr(compute, "_twopass_launch", launch)
    monkeypatch.setattr(compute, "_tp_discard", discard)
    return seen


# Slice starts (file, row) at the fixture's batch of 40: full-width slices.
_SOME_SLICES = [(0, 80), (1, 40), (1, 120), (3, 80), (5, 40), (5, 160),
                (6, 120)]


@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("flip", [False, True])
def test_a_guard_refused_launch_books_its_refusal_only_if_used(
    ds, tmp_path, small, monkeypatch, mode, flip
):
    """A launch a guard refuses (decided on the device in phase A, read back
    in phase B) books `slices_guard_refused` / `last_refusal_reason` only if
    used (steady hint), as inline booked them; discarded on the gate
    (flipping hint), the unpipelined run never ran that pass one, so neither
    they nor its kernel's `slices_fused` / `slices_discarded` are booked."""
    refuse = _slice_keys(ds, _SOME_SLICES)

    def install():
        cur = _spy_slice(monkeypatch)
        real = _REAL["norm_guard"]

        def guard(cn):
            got = real(cn)
            return got | True if cur["key"] in refuse else got

        monkeypatch.setattr(twopass, "norm_guard", guard)
        if flip:
            monkeypatch.setattr(compute, "_TP_LIVE_HINT", _FlipHint())

    install()
    off, st_off = _run(ds, tmp_path / "off", "cosine", mode, False)
    install()
    seen = _track_launches(monkeypatch)
    on, st_on = _run(ds, tmp_path / "on", "cosine", mode, True)
    # Phase A no longer knows (it reads nothing back): the verdict is read
    # when phase B consumes or discards the launch.
    assert not any(d.get("slices_guard_refused") for _, d in seen["recs"])
    refused = [(rec, d) for rec, d in seen["recs"]
               if rec.refusal is not None]
    assert all(r.refusal == "corpus norm" for r, _ in refused)
    used = [r for r, _ in refused if id(r) not in seen["discarded"]]
    dropped = [r for r, _ in refused if id(r) in seen["discarded"]]
    # The flipping hint discards the refused launches; a steady one uses them.
    assert (dropped if flip else used), (len(used), len(dropped), st_on)
    assert st_off["slices_guard_refused"] > 0
    assert st_off["last_refusal_reason"] == "corpus norm"
    _assert_identical(on, off, st_on, st_off, mode)
    # The extra `slices_discarded` are exactly the discarded launches that
    # ran a kernel the guards did NOT refuse: a refused launch, which phase A
    # can no longer skip, books neither its kernel nor a discard.
    ran = lambda d: d.get("slices_fused", 0) + d.get("slices_unfused", 0)  # noqa: E731
    kept = [r for r, d in seen["recs"]
            if id(r) in seen["discarded"] and r.refusal is None and ran(d)
            and not d.get("slices_pass_one_oom")]
    # (Booked when the verdict is read -- deferred to the next sync -- and
    # all of them by the time the counters are read.)
    assert (st_on["slices_discarded"] - st_off["slices_discarded"]
            == len(kept)), (len(kept), st_on, st_off)
    if flip:
        assert all(ran(d) for r, d in refused), "refused launches ran a kernel"


@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_a_persistent_pass_one_oom_fails_both_the_launch_and_the_retry(
    ds, tmp_path, small, monkeypatch, mode
):
    """Some slices' pass one can never allocate, launched or inline: the
    launch's OOM is not booked, the inline retry's is -- once, as
    unpipelined -- and every decision is the unpipelined one."""
    oom = _slice_keys(ds, _SOME_SLICES)
    fired = {"n": 0}

    def install():
        cur = _spy_slice(monkeypatch)
        real = _REAL["approx_rowmax"]

        def rowmax(*a, **kw):
            if cur["key"] in oom:
                fired["n"] += 1
                raise twopass.PassOneUnavailable("CUDA out of memory")
            return real(*a, **kw)

        monkeypatch.setattr(twopass, "approx_rowmax", rowmax)

    install()
    off, st_off = _run(ds, tmp_path / "off", "dot", mode, False)
    n_off, fired["n"] = fired["n"], 0
    install()
    seen = _track_launches(monkeypatch)
    on, st_on = _run(ds, tmp_path / "on", "dot", mode, True)
    oomed = [r for r, d in seen["recs"] if d.get("slices_pass_one_oom")]
    assert oomed and all(id(r) in seen["discarded"] for r in oomed)
    # Each OOM'd launch was retried inline and failed again.
    assert fired["n"] == n_off + len(oomed), (fired, n_off, len(oomed))
    assert st_on["pipeline_retried"] >= len(oomed)
    assert st_off["slices_pass_one_oom"] > 0
    _assert_identical(on, off, st_on, st_off, mode)


def _cuda_like_fused(monkeypatch, fail, cur):
    """Simulate CUDA's fused pass one on CPU: `fused_rowmax` reduces the
    float32 product itself (the kernel's arithmetic), and only a fused result
    is trusted (`accumulator_is_ours`, as on CUDA), so once the kernel is off
    every slice goes all live -- the fused state shows in the decisions. The
    kernel fails, like a real non-OOM failure (`fuse_disable`), on the
    slices in `fail`; `state` counts where."""
    state = {"in_launch": False, "launch_fail": 0, "inline_fail": 0}
    real_launch = _REAL["_twopass_launch"]

    def launch(*a, **kw):
        state["in_launch"] = True
        try:
            return real_launch(*a, **kw)
        finally:
            state["in_launch"] = False

    def fused(Qh, Ch, cs, config=None, probe=False, with_tiles=False):
        if twopass.fused_off() is not None:
            return None
        if cur["key"] in fail:
            state["launch_fail" if state["in_launch"] else "inline_fail"] += 1
            twopass.fuse_disable("test: fused kernel failed")
            return None
        g = (Qh.float() @ Ch.float().T) * cs[None, :]
        row_max = g.amax(dim=1)
        if with_tiles:
            return row_max, twopass.tile_amax(g, TILE).T.contiguous(), TILE
        return row_max

    monkeypatch.setattr(compute, "_twopass_launch", launch)
    monkeypatch.setattr(twopass, "fused_rowmax", fused)
    monkeypatch.setattr(twopass, "accumulator_is_ours",
                        lambda dev, used_fused: bool(used_fused))
    return state


@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("flip", [False, True])
# (0, 40) would be the certification slice: the kernel never runs at all.
@pytest.mark.parametrize("at", [(0, 80), (0, 120), (0, 160), (1, 80), (3, 40)])
def test_a_fused_failure_in_the_next_launch_does_not_change_this_slice(
    ds, tmp_path, small, monkeypatch, mode, flip, at
):
    """A(N+1) hits a non-OOM fused-kernel failure before B(N) runs. The
    unpipelined run disables the kernel only at slice N+1's own pass one, so
    slice N -- if B(N) runs a pass one inline (certification, or a launch the
    stale hint did not make) -- must still get the fused kernel, and slice
    N+1 must see it off. Decisions and output equal unpipelined."""
    fail = _slice_keys(ds, [at])

    def install():
        cur = _spy_slice(monkeypatch)
        if flip:
            monkeypatch.setattr(compute, "_TP_LIVE_HINT", _FlipHint())
        return _cuda_like_fused(monkeypatch, fail, cur)

    st0 = install()
    off, st_off = _run(ds, tmp_path / "off", "dot", mode, False)
    st1 = install()
    on, st_on = _run(ds, tmp_path / "on", "dot", mode, True)
    assert st_off["slices_fused"] > 0
    assert (st0["inline_fail"] > 0) is (st_off["slices_unfused"] > 0), st0
    assert twopass.fuse_pending() is None, "a deferred failure leaked"
    _assert_identical(on, off, st_on, st_off, f"{mode}/{at}/{flip}")


def test_a_fused_failure_in_a_launch_is_deferred_to_its_decision(
    ds, tmp_path, small, monkeypatch
):
    """The deferral itself, observed: during A(N+1) the kernel failed, but
    the run-level kernel state is untouched until B(N+1) consumes it -- and
    the launches made meanwhile are re-run inline."""
    fail = _slice_keys(ds, [(0, 120)])
    cur = _spy_slice(monkeypatch)
    state = _cuda_like_fused(monkeypatch, fail, cur)
    real_prepare = compute._twopass_prepare
    during = []

    def prepare(tp_groups, sl, *a, **kw):
        during.append((twopass._FUSE_OFF, twopass.fuse_pending()))
        return real_prepare(tp_groups, sl, *a, **kw)

    monkeypatch.setattr(compute, "_twopass_prepare", prepare)
    _, st = _run(ds, tmp_path / "on", "dot", "slice", True)
    assert state["launch_fail"] == 1 and state["inline_fail"] == 0, state
    # Some decision ran with the failure pending but the kernel still on.
    assert (None, "test: fused kernel failed") in during, during
    assert twopass._FUSE_OFF == "test: fused kernel failed"
    assert st["pipeline_used"] > 0, st


@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_a_mid_run_disable_discards_the_launch_already_issued(
    ds, tmp_path, small, monkeypatch, mode
):
    """B(N) disables the two-pass after A(N+1) launched: B(N+1) scores one
    pass, and the orphaned launch is booked as discarded."""
    def install():
        real = compute._twopass_prepare
        calls = [0]

        def prepare(*a, **kw):
            calls[0] += 1
            if calls[0] == 6:
                twopass.disable("test: mid-run disable")
            return real(*a, **kw)

        monkeypatch.setattr(compute, "_twopass_prepare", prepare)

    install()
    off, st_off = _run(ds, tmp_path / "off", "cosine", mode, False)
    install()
    on, st_on = _run(ds, tmp_path / "on", "cosine", mode, True)
    assert st_on["unavailable"] == "test: mid-run disable"
    assert st_on["pipeline_used"] > 0 and st_on["pipeline_discarded"] > 0, st_on
    _assert_identical(on, off, st_on, st_off, mode)


def test_a_tiles_crossover_bail_is_identical(ds, tmp_path, small, monkeypatch):
    monkeypatch.setattr(twopass, "TILE_CROSSOVER", 0.0)
    off, st_off = _run(ds, tmp_path / "off", "cosine", "tiles", False)
    on, st_on = _run(ds, tmp_path / "on", "cosine", "tiles", True)
    assert st_on["tiles_crossover_bails"] > 0 and st_on["pipeline_used"] > 0
    _assert_identical(on, off, st_on, st_off)


def test_a_certification_refusal_launches_nothing(ds, tmp_path, small,
                                                  monkeypatch):
    monkeypatch.setattr(compute, "_certify_two_pass",
                        lambda *a, **kw: "test: refused")
    off, st_off = _run(ds, tmp_path / "off", "cosine", "slice", False)
    on, st_on = _run(ds, tmp_path / "on", "cosine", "slice", True)
    assert st_on["slices_twopass"] == 0 and st_on["pipeline_launched"] == 0
    _assert_identical(on, off, st_on, st_off)


def test_an_exception_in_phase_b_releases_the_pending_launch(ds, tmp_path,
                                                             small, monkeypatch):
    """B(N) raising while A(N+1) is pending: the error surfaces unchanged and
    both launches are released (no generator left suspended)."""
    pendings = []
    real_begin = twopass.upper_bounds_begin

    def begin(*a, **kw):
        p = real_begin(*a, **kw)
        pendings.append(p)
        return p

    real = compute._twopass_prepare
    calls = [0]

    def prepare(*a, **kw):
        calls[0] += 1
        if calls[0] == 8:
            raise RuntimeError("phase B boom")
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "upper_bounds_begin", begin)
    monkeypatch.setattr(compute, "_twopass_prepare", prepare)
    with pytest.raises(RuntimeError, match="phase B boom"):
        _run(ds, tmp_path / "on", "cosine", "slice", True)
    assert pendings and all(not p.launched for p in pendings)


def test_a_failing_release_never_replaces_the_exception_in_flight(
    ds, tmp_path, small, monkeypatch
):
    """B(N) raising while launches are pending, and every release of them
    raising too (a sticky CUDA error surfaces at the first device call): the
    run reports B's error, not the release's, and both unwind paths --
    `process_slice`'s (this slice's launches) and the driver's (the next
    slice's) -- still attempt every release."""
    pendings = []
    real_begin = twopass.upper_bounds_begin

    def begin(*a, **kw):
        p = real_begin(*a, **kw)
        pendings.append(p)
        return p

    attempted = []

    def sticky(pending):
        attempted.append(pending)
        raise RuntimeError("CUDA error: an illegal memory access (sticky)")

    real = compute._twopass_prepare
    calls = [0]

    def prepare(*a, **kw):
        calls[0] += 1
        if calls[0] == 8:
            raise RuntimeError("phase B boom")
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "upper_bounds_begin", begin)
    monkeypatch.setattr(twopass, "upper_bounds_discard", sticky)
    monkeypatch.setattr(compute, "_twopass_prepare", prepare)
    try:
        with pytest.raises(RuntimeError, match="phase B boom"):
            _run(ds, tmp_path / "on", "cosine", "slice", True)
        # The ones nothing consumed: B(N)'s own (process_slice) and A(N+1)'s
        # (the driver). Every one of them was attempted despite the raises.
        left = [p for p in pendings if p.launched]
        assert len(left) >= 2, left
        assert {id(p) for p in left} <= {id(p) for p in attempted}
    finally:
        for p in pendings:
            if p._gen is not None:
                p._gen.close()


def test_the_driver_releases_every_pending_launch_when_one_release_fails(
    monkeypatch,
):
    """`_run_dense_pipelined` alone: B(0) raises with A(1) pending; the first
    release raises, the second is still made, and B's error surfaces."""

    class Rec:
        def __init__(self, bad):
            self.bad, self.released = bad, False
            self.pending = self.handle = object()
            self.cert_key = None
            self.fuse_failure = None

    recs = [Rec(True), Rec(False)]

    def release(pending):
        rec = next(r for r in recs if r.handle is pending)
        rec.released = True
        if rec.bad:
            raise RuntimeError("sticky")

    monkeypatch.setattr(twopass, "upper_bounds_discard", release)

    class Batch:
        def transfer(self, r0, r1, device):
            return r0

    def launch(sl):
        return None if sl == 0 else {"a": recs[0], "b": recs[1]}

    def process(r0, r1, sl, launched):
        raise RuntimeError("phase B boom")

    with pytest.raises(RuntimeError, match="phase B boom"):
        compute._run_dense_pipelined(Batch(), [(0, 4), (4, 8)], "cpu",
                                     launch, process)
    assert all(r.released for r in recs)


def test_the_side_stream_never_aliases_the_copy_stream(monkeypatch):
    """torch's pooled streams are handed out round-robin, so streams created
    independently can alias. The pass-one side stream and the dense copy
    stream are one per-device pair, created together, re-drawn on a
    collision, and reused by every batch (the copy stream is no longer
    created per batch, which would sooner or later land on the side
    stream)."""

    class Stream:
        # Pool of 3, and the draw after the first repeats it: the collision.
        draws = iter([7, 7, 8, 9, 7, 8, 9, 7, 8])

        def __init__(self, device=None):
            self.cuda_stream = next(Stream.draws)

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(compute, "_DEVICE_STREAMS", {})
    side = compute._pass_one_stream("cuda:0")
    copies = {id(compute._dense_copy_stream("cuda:0")) for _ in range(40)}
    copy = compute._dense_copy_stream("cuda:0")
    assert side is compute._pass_one_stream("cuda:0")
    assert copies == {id(copy)}, "one copy stream per device, not per batch"
    assert copy.cuda_stream != side.cuda_stream
    other = compute._dense_copy_stream("cuda:1")
    assert other is not copy and compute._pass_one_stream("cuda:1") is not side
    assert compute._pass_one_stream("cpu") is None


# --- 3. order -------------------------------------------------------------------


class _Order:
    """Record launches, decisions and folds of one end-to-end run."""

    def __init__(self, monkeypatch):
        self.events = []
        self.folded = {}       # id(threshold tensor) -> its value after the fold
        self.stale = []
        self.slice_ids = {}
        real_launch = compute._twopass_launch
        real_prepare = compute._twopass_prepare
        real_merge = compute._merge_topk

        self._keep = []    # pin every slice so no id() is ever recycled
        real_transfer = compute.DenseCorpusBatch.transfer

        def transfer(batch, r0, r1, device):
            sl = real_transfer(batch, r0, r1, device)
            # (batch, position in that batch): N and N+1 of ONE batch group.
            self._keep.append((batch, sl))
            nb = sum(1 for b, _ in self._keep if b is batch) - 1
            self.slice_ids[id(sl)] = (id(batch), nb)
            return sl

        def sid(sl):
            return self.slice_ids[id(sl)]

        monkeypatch.setattr(compute.DenseCorpusBatch, "transfer", transfer)

        def launch(tp_groups, sl, device, step):
            out = real_launch(tp_groups, sl, device, step)
            if out:
                self.events.append(("A", sid(sl)))
            return out

        def prepare(tp_groups, sl, spec_thr, spec_qsel, device, **kw):
            self.events.append(("B", sid(sl)))
            for g in tp_groups.values():
                for m in g["members"]:
                    # The decision must read the member's LIVE running
                    # threshold (the very tensor the fold updates in place),
                    # holding the latest fold's value -- not a copy taken
                    # earlier, e.g. at launch time.
                    t = spec_thr[m]
                    want = self.folded.get(id(t))
                    if want is None:
                        if self.folded:
                            self.stale.append((sid(sl), m, "not live"))
                    elif not torch.equal(t, want):
                        self.stale.append((sid(sl), m, "pre-fold value"))
            return real_prepare(tp_groups, sl, spec_thr, spec_qsel, device, **kw)

        def merge(top_key, top_enc, parts, k, thr=None):
            got = real_merge(top_key, top_enc, parts, k, thr=thr)
            if thr is not None:
                self.events.append(("fold", None))
                self.folded[id(thr)] = thr.clone()
            return got

        monkeypatch.setattr(compute, "_twopass_launch", launch)
        monkeypatch.setattr(compute, "_twopass_prepare", prepare)
        monkeypatch.setattr(compute, "_merge_topk", merge)


@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_launch_precedes_the_previous_decision_and_decisions_see_the_fold(
    ds, tmp_path, small, monkeypatch, mode
):
    order = _Order(monkeypatch)
    _run(ds, tmp_path / "on", "cosine", mode, True, filtered=False)
    ev = order.events
    pos = {e: i for i, e in enumerate(ev) if e[0] != "fold"}
    launched = [s for kind, s in ev if kind == "A"]
    assert len(launched) > 3, ev
    checked = 0
    for b, n in launched:
        prev = ("B", (b, n - 1))
        if prev in pos:
            # A(N+1) issued before B(N) ...
            assert pos[("A", (b, n))] < pos[prev], ((b, n), ev)
            # ... and B(N+1), when it ran, strictly after B(N).
            if ("B", (b, n)) in pos:
                assert pos[prev] < pos[("B", (b, n))]
            checked += 1
    assert checked > 3, ev
    # Between consecutive decisions B(N), B(N+1) of one batch group there is
    # a fold, which B(N+1)'s thresholds already reflect (`order.stale`).
    decided = [(i, s) for i, (kind, s) in enumerate(ev) if kind == "B"]
    for (i, (b, n)), (j, (b2, n2)) in zip(decided, decided[1:]):
        if b2 == b and n2 == n + 1:
            assert any(kind == "fold" for kind, _ in ev[i:j]), ((b, n), ev[i:j])
    assert not order.stale, f"decisions read pre-fold thresholds: {order.stale}"


def test_the_drivers_issue_a_of_the_next_slice_before_b(monkeypatch):
    """The plain driver, isolated: transfer(N+1), A(N+1), then B(N)."""
    calls = []

    class Batch:
        def transfer(self, r0, r1, device):
            calls.append(("T", r0))
            return r0

    def launch(sl):
        calls.append(("A", sl))
        return {"rec": sl}

    def process(r0, r1, sl, launched):
        assert launched == {"rec": sl}
        calls.append(("B", sl))

    compute._run_dense_pipelined(Batch(), [(0, 4), (4, 8), (8, 10)], "cpu",
                                 launch, process)
    assert calls == [("T", 0), ("A", 0), ("T", 4), ("A", 4), ("B", 0),
                     ("T", 8), ("A", 8), ("B", 4), ("B", 8)]


@pytest.mark.parametrize("inline", ["1", "0"])
def test_the_prefetch_driver_order_and_device_backlog(monkeypatch, inline):
    """The CUDA ring driver against a fake GPU that only progresses when the
    host waits: A(N+1) precedes B(N), staging keeps one in flight, and the
    device backlog is bounded at FOUR corpus slices (two scored, one with its
    pass one queued, one prefetched) -- one deeper than unpipelined."""
    monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", inline)
    calls = []
    submitted = completed = allocated = peak = 0
    stream = object()

    class Event:
        def record(self, got):
            assert got is stream
            self.through = submitted

        def synchronize(self):
            nonlocal completed
            completed = max(completed, self.through)

    class Slice:
        def __init__(self, r0):
            self.r0 = r0

        def wait_on(self, got):
            assert got is stream
            calls.append(("W", self.r0))

    class Batch:
        def prefetch(self, r0, r1, device):
            self.stage_prefetch(self.prepare_prefetch(r0, r1, device))

        def prepare_prefetch(self, r0, r1, device):
            nonlocal allocated, peak
            allocated += 1
            peak = max(peak, allocated - completed)
            assert allocated - completed <= 4, "unbounded device backlog"
            calls.append(("P", r0))
            return r0

        def stage_prefetch(self, plan):
            pass

        def transfer(self, r0, r1, device):
            calls.append(("T", r0))
            return Slice(r0)

    def launch(sl):
        calls.append(("A", sl.r0))
        return None

    def process(r0, r1, sl, launched):
        nonlocal submitted
        submitted += 1
        calls.append(("B", r0))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    try:
        compute._run_dense_prefetch_pipelined(
            Batch(), [(0, 1), (1, 2), (2, 3)], "cuda", launch, process)
        assert calls == [
            ("P", 0), ("T", 0), ("W", 0), ("A", 0), ("P", 1),
            ("T", 1), ("W", 1), ("A", 1), ("P", 2), ("B", 0),
            ("T", 2), ("W", 2), ("A", 2), ("B", 1),
            ("B", 2),
        ], calls
        for count in (64, 1, 2, 9):
            compute._run_dense_prefetch_pipelined(
                Batch(), [(i, i + 1) for i in range(count)], "cuda",
                launch, process)
            assert submitted == completed == allocated
        assert peak == 4
    finally:
        compute._close_dense_staging()


# --- 4. config, env and manifest ------------------------------------------------


def test_the_flag_defaults_on_and_the_env_overrides_it(monkeypatch):
    from nova_bf import config

    assert ParamsConfig().pipeline_pass_one is config.PIPELINE_PASS_ONE_DEFAULT is True
    twopass.set_pipeline_pass_one(None)
    assert twopass.pipeline_pass_one() is True
    twopass.set_pipeline_pass_one(False)
    assert twopass.pipeline_pass_one() is False
    monkeypatch.setenv("NOVA_BF_PIPELINE_PASS_ONE", "1")
    assert twopass.pipeline_pass_one() is True
    twopass.set_pipeline_pass_one(True)
    monkeypatch.setenv("NOVA_BF_PIPELINE_PASS_ONE", "no")
    assert twopass.pipeline_pass_one() is False
    monkeypatch.setenv("NOVA_BF_PIPELINE_PASS_ONE", "maybe")
    with pytest.raises(ValueError, match="NOVA_BF_PIPELINE_PASS_ONE"):
        twopass.pipeline_pass_one()
    twopass.set_pipeline_pass_one(False)
    twopass.reset()
    monkeypatch.delenv("NOVA_BF_PIPELINE_PASS_ONE")
    assert twopass.pipeline_pass_one() is True, "reset() must clear the run's flag"


@pytest.mark.parametrize("pipeline", [False, True])
def test_the_manifest_records_the_policy_and_consistent_counters(
    ds, tmp_path, small, monkeypatch, pipeline
):
    _run(ds, tmp_path / "m", "cosine", "tiles", pipeline)
    doc = json.loads(next((tmp_path / "m").rglob("*manifest*/*.json")).read_text())
    assert doc["params"]["pipeline_pass_one"] is pipeline
    tp = doc["params"]["kernels"]["twopass"]
    assert tp["pipeline_pass_one"] is pipeline
    assert tp["counters_consistent"] is True, tp
    assert tp["pipeline_launched"] == tp["pipeline_used"] + tp["pipeline_discarded"]
    assert (tp["pipeline_used"] > 0) is pipeline, tp


def test_the_env_override_reaches_the_run_and_the_manifest(ds, tmp_path, small,
                                                          monkeypatch):
    monkeypatch.setenv("NOVA_BF_PIPELINE_PASS_ONE", "1")
    _, st = _run(ds, tmp_path / "e", "dot", "slice", False)
    assert st["pipeline_used"] > 0 and st["pipeline_pass_one"] is True
    doc = json.loads(next((tmp_path / "e").rglob("*manifest*/*.json")).read_text())
    assert doc["params"]["pipeline_pass_one"] is True


# --- 5. the device -------------------------------------------------------------


# Device-shape fixture: d=128, 1024-row slices, k=10. The query height is per
# mode. Slice mode re-scores the live rows with ONE padded GEMM whose height
# comes from the production ladder (`twopass.pad_candidates`: PAD_SMALL's
# 2048/4096/5120, then PAD_FLOOR's 7168 and up, each kept only if it is below
# the full height, and a height is a candidate only if it holds every live
# row). At 512 queries every rung is >= the full height, so the only
# candidate is the full height itself and every slice is scored plain ("no
# useful query-row reduction") -- whatever the live fraction. At 8192 rows
# (the production MIN_QUERY_ROWS) all four rungs 2048..7168 are below the
# full height, but these Gaussian vectors keep ~60-65% of rows live, so only
# 5120 and 7168 are reached (on CPU: four slices at 7168, three at 5120). No
# ladder is patched, so these are the production heights. Tiles mode needs
# no padded height and keeps the small, fast 512-row shape.
DEVICE_NQ = {"slice": 8192, "tiles": 512}


def _device_shape_data(tmp_path, nq):
    rng = np.random.default_rng(3)
    cdir = tmp_path / "c"
    cdir.mkdir()
    for fi in range(6):
        n = 3000 if fi != 2 else 700
        v = rng.standard_normal((n, 128)).astype(np.float16).astype(np.float32)
        pq.write_table(pa.table({
            "dense_embedding": pa.array(v.tolist(), type=pa.list_(pa.float32())),
            "id": [f"c{fi}_{r}" for r in range(n)]}), str(cdir / f"f{fi}.parquet"))
    qp = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((nq, 128)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32())),
        "qid": [f"q{i}" for i in range(nq)]}), str(qp))
    return cdir, qp


def _device_shape_runs(tmp_path, monkeypatch, mode, metric, one_pass=False):
    """Unpipelined then pipelined run over `_device_shape_data`, and with
    `one_pass` also the two-pass-off run (returned last)."""
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 256)
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
    cdir, qp = _device_shape_data(tmp_path, DEVICE_NQ[mode])

    def run(pipeline, out):
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qp), id_column="qid"),
            output=OutputConfig(path=str(tmp_path / out)),
            params=ParamsConfig(dense_batch_size=1024, pipeline_pass_one=pipeline,
                                **({"two_pass": "off"} if out == "base"
                                   else {"two_pass_rescore": mode})),
            searches=[SearchSpec(name=metric, metric=metric, k=10)])
        (tmp_path / out).mkdir()
        twopass.reset()
        got = {n: pq.read_table(p) for n, p in run_compute(cfg).items()}
        return got, twopass.stats()

    off, st_off = run(False, "off")
    on, st_on = run(True, "on")
    if not one_pass:
        return off, st_off, on, st_on
    base, _ = run(False, "base")
    return off, st_off, on, st_on, base


def _assert_slice_mode_engaged(st):
    """Slice mode really ran a padded exact GEMM (0 < M < n_full) on some
    slice: an all-pruned slice books M = 0 and no GEMM, so `rows_padded <
    rows_full` alone would pass on a run that never re-scored a row.

    The stats are totals, not per slice; `gemms > 0` means some slice had
    M > 0, and M < n_full for EVERY booked slice because `_twopass_prepare`
    scores a slice whose padded height reaches n_full plain (`slices_plain`)
    and never books it as two-pass."""
    assert st["slices_twopass"] > 0 and st["gemms"] > 0, st
    assert 0 < st["rows_padded"] < st["rows_full"], st


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("pinned", ["2", "0"])
def test_pipelined_is_byte_identical_on_cuda(tmp_path, monkeypatch, mode,
                                             metric, pinned):
    """The side-stream launch on hardware: fused pass one on the pass-one
    stream, consumed on the compute stream, through the pinned ring driver
    (`NOVA_BF_PINNED=2`) and the pageable one (`0`)."""
    monkeypatch.setenv("NOVA_BF_PINNED", pinned)
    off, st_off, on, st_on, base = _device_shape_runs(
        tmp_path, monkeypatch, mode, metric, one_pass=True)
    assert st_on["slices_twopass"] > 0 and st_on["pipeline_used"] > 0, st_on
    assert st_on["slices_fused"] > 0, st_on
    assert st_on["audit_violations"] == 0, st_on
    _assert_identical(on, off, st_on, st_off, f"cuda/{mode}/{metric}")
    if mode == "slice":
        # A real row reduction, and through `_verify_shape`'s padded heights
        # the output is the one-pass output, byte for byte.
        _assert_slice_mode_engaged(st_on)
        for name in base:
            assert on[name].equals(base[name]), f"cuda/slice/{metric} {name}"


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_the_device_shape_engages_slice_mode(tmp_path, monkeypatch, metric):
    """The CPU half of the CUDA test's slice-mode precondition: the engage
    decision (pass-one bound, live rows, padded-height ladder) is the same
    code on either device, so with the production pad ladder (NOT the
    `small` fixture's) the CUDA test's data must engage slice mode here too,
    stay byte-identical pipelined, and equal one-pass. What only the GPU run
    can show: that `_verify_shape` accepts a padded height there too (cuBLAS
    keeping a 5120/7168-row GEMM bit-identical to the 8192-row one; if not,
    those slices go plain and the CUDA test's precondition fails), and the
    side-stream launch itself."""
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_DEVICE", "cpu")
    off, st_off, on, st_on, base = _device_shape_runs(
        tmp_path, monkeypatch, "slice", metric, one_pass=True)
    assert st_on["pipeline_used"] > 0, st_on
    _assert_slice_mode_engaged(st_on)
    _assert_identical(on, off, st_on, st_off, f"cpu/slice/{metric}")
    for name in base:
        assert on[name].equals(base[name]), f"cpu/slice/{metric} {name}"


# --- 6. device-side guards, end to end ------------------------------------------
#
# Pass one's per-slice guards are evaluated on the device and read back only
# after the launch (`twopass._upper_bounds`). End to end that must change
# nothing: every guard-refused slice prunes nothing, the results equal the
# one-pass run's, and the counters equal the unpipelined run's -- including a
# refused launch that phase A could no longer skip (it now runs a kernel whose
# counters are taken back) and phase B discards.


@pytest.fixture
def ds_degenerate(tmp_path):
    """The `ds` shape with degenerate corpus rows that trip each slice guard:
    a zero row (corpus norm), an fp16-overflowing row (corpus overflow) and
    a tiny-norm row (the prunability cut: all rows refused)."""
    rng = np.random.default_rng(31)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    sizes = [200, 200, 37, 200, 144, 200, 200]
    trip = {1: (50, 0.0), 3: (130, 1e5), 5: (10, 1e-4), 6: (170, 0.0)}
    g = 0
    for fi, n in enumerate(sizes):
        vecs = rng.standard_normal((n, DIM)).astype(np.float16).astype(np.float32)
        if fi in trip:
            r, f = trip[fi]
            vecs[r] *= np.float32(f)
        _write(cdir / f"f{fi}.parquet", vecs,
               id=[f"c{g + r:06d}" for r in range(n)],
               lang=["eng" if (g + r) % 3 else "fra" for r in range(n)])
        g += n
    qpath = tmp_path / "queries.parquet"
    _write(qpath, rng.standard_normal((NQ, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(NQ)],
           qset=["a"] * (NQ // 2) + ["b"] * (NQ - NQ // 2))
    return {"cdir": str(cdir), "qpath": str(qpath)}


_DEGENERATE_REASONS = {"corpus norm", "corpus overflow", "all rows refused"}


@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_guard_refused_slices_prune_nothing_and_match_one_pass(
    ds_degenerate, tmp_path, small, monkeypatch, metric, mode
):
    from twopass_compare import assert_same, norm_product, tolerance

    base, _ = _run(ds_degenerate, tmp_path / "base", metric, "off", False)
    off, st_off = _run(ds_degenerate, tmp_path / "off", metric, mode, False)
    on, st_on = _run(ds_degenerate, tmp_path / "on", metric, mode, True)
    assert st_off["slices_twopass"] > 0, st_off
    assert st_off["slices_guard_refused"] > 0, st_off
    assert st_off["last_refusal_reason"] in _DEGENERATE_REASONS, st_off
    assert st_on["pipeline_used"] > 0, st_on
    _assert_identical(on, off, st_on, st_off, f"{mode}/{metric}")
    tol = tolerance(DIM, 1.0 if metric == "cosine" else
                    norm_product(ds_degenerate["cdir"], ds_degenerate["qpath"]))
    for name in base:
        assert_same(mode, off[name], base[name], f"{mode}/{metric}/{name}",
                    abs_tol=tol)
        assert_same(mode, on[name], base[name], f"{mode}/{metric}/{name}",
                    abs_tol=tol)


@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_each_degenerate_slice_is_refused_with_its_own_reason(
    ds_degenerate, tmp_path, small, monkeypatch, mode
):
    """Every reason the fixture plants is reported (and only those), by the
    slice that planted it, pipelined or not -- the device verdict names the
    same guard the host used to."""
    real = twopass._probe_refusal
    for pipeline in (False, True):
        reasons = []

        def spy(vals):
            got = real(vals)
            if got is not None:
                reasons.append(got)
            return got

        monkeypatch.setattr(twopass, "_probe_refusal", spy)
        _, st = _run(ds_degenerate, tmp_path / f"r{pipeline}", "cosine",
                     mode, pipeline)
        assert set(reasons) == _DEGENERATE_REASONS, (pipeline, reasons)
        assert st["slices_guard_refused"] >= 3, st


def _install_read_counter(monkeypatch, state):
    """Count device->host reads made on the thread in `state["active"]`
    (a thread ident, or None): scalar/array conversions, `torch.equal`,
    `.to("cpu")`, and the data-dependent-shape ops (`nonzero`,
    `masked_select`, boolean-mask indexing) -- what twopass's own
    `_ReadCounter` counts (tests/test_twopass_sync_free.py)."""
    import threading

    def on():
        return state["active"] == threading.get_ident()

    def wrap(name, fn):
        def f(*a, **kw):
            if on():
                state["reads"].append(name)
            return fn(*a, **kw)
        return f

    for name in ("__bool__", "__float__", "__int__", "__index__", "item",
                 "tolist", "numpy", "cpu", "nonzero", "masked_select"):
        monkeypatch.setattr(torch.Tensor, name,
                            wrap(name, getattr(torch.Tensor, name)))
    for name in ("equal", "nonzero", "masked_select"):
        monkeypatch.setattr(torch, name, wrap(name, getattr(torch, name)))
    real_getitem = torch.Tensor.__getitem__

    def getitem(t, idx):
        items = idx if isinstance(idx, tuple) else (idx,)
        if on() and any(isinstance(i, torch.Tensor) and i.dtype is torch.bool
                        for i in items):
            state["reads"].append("t[mask]")
        return real_getitem(t, idx)

    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    real_to = torch.Tensor.to

    def to(t, *a, **kw):
        dev = kw.get("device", a[0] if a else None)
        if on() and str(dev).startswith("cpu"):
            state["reads"].append("to(cpu)")
        return real_to(t, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "to", to)


@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_phase_a_reads_nothing_back(ds_degenerate, tmp_path, small,
                                    monkeypatch, metric, mode):
    """Every device->host read made inside `_twopass_launch` (phase A), on
    the thread that runs it, over a whole pipelined run with refusals. On a
    GPU each would block the host on the compute stream."""
    import threading

    state = {"active": None, "reads": []}
    _install_read_counter(monkeypatch, state)
    real_launch = compute._twopass_launch
    launched = [0]

    def launch(*a, **kw):
        state["active"] = threading.get_ident()
        try:
            out = real_launch(*a, **kw)
        finally:
            state["active"] = None
        launched[0] += len(out or {})
        return out

    monkeypatch.setattr(compute, "_twopass_launch", launch)
    _, st = _run(ds_degenerate, tmp_path / "on", metric, mode, True)
    assert launched[0] > 0 and st["pipeline_used"] > 0, st
    assert st["slices_guard_refused"] > 0, st
    assert state["reads"] == [], state["reads"]


@pytest.mark.parametrize("pipeline", [False, True])
def test_tiles_refusals_keep_the_manifest_counter_identity(
    ds_degenerate, tmp_path, small, monkeypatch, pipeline
):
    """REGRESSION. A tiles-mode slice whose pass one a guard refused is all
    live with no kernel; it used to re-score its tiles anyway (whenever
    TILE_CROSSOVER >= 1) and book a `slices_twopass` no pass one paid for,
    so `counters_consistent` read False on any tiles run with a refusal. It
    is now scored plain, as slice mode scores it."""
    _run(ds_degenerate, tmp_path / "m", "cosine", "tiles", pipeline)
    doc = json.loads(next((tmp_path / "m").rglob("*manifest*/*.json")).read_text())
    tp = doc["params"]["kernels"]["twopass"]
    assert tp["slices_guard_refused"] > 0, tp
    assert tp["slices_tiles"] > 0, tp
    assert tp["counters_consistent"] is True, tp
    assert _identity_gap(tp) == 0, tp


# --- 7. a refused slice's speculative launch leaves no trace --------------------


def _has_zero_row(Ch) -> bool:
    return bool((Ch.float().abs().sum(dim=1) == 0).any())


def _rowmax_on_refused(monkeypatch, action):
    """Wrap approx_rowmax: on a slice holding a zero corpus row (refused,
    "corpus norm"), run `action()` -- raise, or disable the fused kernel."""
    real = _REAL["approx_rowmax"]
    fired = {"n": 0}

    def rowmax(Qh, Ch, *a, **kw):
        if _has_zero_row(Ch):
            fired["n"] += 1
            action()
        return real(Qh, Ch, *a, **kw)

    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)
    return fired


def _zero_rows_ds(tmp_path):
    rng = np.random.default_rng(37)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    g = 0
    for fi, n in enumerate([200, 200, 37, 200, 144, 200, 200]):
        vecs = rng.standard_normal((n, DIM)).astype(np.float16).astype(np.float32)
        if fi in (1, 3, 6):
            vecs[(50, 90, 170)[(1, 3, 6).index(fi)]] = 0.0
        _write(cdir / f"f{fi}.parquet", vecs,
               id=[f"c{g + r:06d}" for r in range(n)],
               lang=["eng" if (g + r) % 3 else "fra" for r in range(n)])
        g += n
    qpath = tmp_path / "queries.parquet"
    _write(qpath, rng.standard_normal((NQ, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(NQ)],
           qset=["a"] * (NQ // 2) + ["b"] * (NQ - NQ // 2))
    return {"cdir": str(cdir), "qpath": str(qpath)}


@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("kind", ["oom", "error"])
def test_a_launch_failure_on_a_refused_slice_changes_nothing(
    tmp_path, small, monkeypatch, mode, kind
):
    """A pass one that would fail (OOM, or any error) on a slice the guards
    refuse: unpipelined it is never launched; pipelined the speculative launch
    fails, and that failure must not surface -- no OOM, no exception, no
    hint change, the same refusal."""
    ds = _zero_rows_ds(tmp_path)

    def boom():
        if kind == "oom":
            twopass._STATS["slices_unfused"] += 1     # as the real path does
            raise twopass.PassOneUnavailable("CUDA out of memory")
        raise RuntimeError("cublas boom")

    fired = _rowmax_on_refused(monkeypatch, boom)
    off, st_off = _run(ds, tmp_path / "off", "cosine", mode, False)
    assert fired["n"] == 0, "unpipelined launched pass one on a refused slice"
    on, st_on = _run(ds, tmp_path / "on", "cosine", mode, True)
    assert fired["n"] > 0, "the pipelined run never launched on a refused slice"
    assert st_off["slices_guard_refused"] > 0, st_off
    assert st_on["slices_pass_one_oom"] == st_off["slices_pass_one_oom"] == 0
    assert st_on["pipeline_retried"] == 0, st_on
    _assert_identical(on, off, st_on, st_off, f"{mode}/{kind}")


@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_a_fused_failure_on_a_refused_slice_keeps_the_kernel_on(
    tmp_path, small, monkeypatch, mode
):
    """A fused-kernel failure deferred by a speculative launch on a refused
    slice is dropped (the guards-first run never ran that kernel); on a live
    slice it is applied, as before."""
    ds = _zero_rows_ds(tmp_path)
    fired = _rowmax_on_refused(monkeypatch,
                               lambda: twopass.fuse_disable("kernel boom"))
    off, st_off = _run(ds, tmp_path / "off", "cosine", mode, False)
    assert twopass._FUSE_OFF is None and fired["n"] == 0
    on, st_on = _run(ds, tmp_path / "on", "cosine", mode, True)
    assert fired["n"] > 0
    assert twopass._FUSE_OFF is None, twopass._FUSE_OFF
    assert twopass.fuse_pending() is None
    _assert_identical(on, off, st_on, st_off, mode)

    # Control: the same failure on every slice does disable it, both ways.
    real = _REAL["approx_rowmax"]

    def always(*a, **kw):
        twopass.fuse_disable("kernel boom")
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "approx_rowmax", always)
    for pipeline in (False, True):
        _run(ds, tmp_path / f"all{pipeline}", "cosine", mode, pipeline)
        assert twopass._FUSE_OFF == "kernel boom", pipeline


@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_a_discard_reads_nothing_back(ds_degenerate, tmp_path, small,
                                      monkeypatch, mode):
    """The hint flipping to plain discards a launch; telling whether it was
    refused needs a device read, which is deferred to the next point that
    synchronised anyway -- `_tp_discard` itself reads nothing."""
    import threading

    state = {"active": None, "reads": []}
    _install_read_counter(monkeypatch, state)
    real_discard = _REAL["_tp_discard"]
    n = [0]

    def discard(rec, retried=False, count=True):
        n[0] += 1
        state["active"] = threading.get_ident()
        try:
            return real_discard(rec, retried=retried, count=count)
        finally:
            state["active"] = None

    monkeypatch.setattr(compute, "_tp_discard", discard)
    monkeypatch.setattr(compute, "_TP_LIVE_HINT", _FlipHint())
    _, st = _run(ds_degenerate, tmp_path / "on", "cosine", mode, True)
    assert st["pipeline_discarded"] > 0 and n[0] > 0, st
    assert state["reads"] == [], state["reads"]
    assert _identity_gap(st) == 0, st


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["slice", "tiles"])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_phase_a_never_synchronises_on_cuda(tmp_path, monkeypatch, mode,
                                            metric):
    """torch's own sync detector around every `_twopass_launch` of a real
    pipelined CUDA run, a zero corpus row included (a refused slice)."""
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 256)
    monkeypatch.setenv("NOVA_BF_TWOPASS_FORCE_ENGAGE", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
    rng = np.random.default_rng(5)
    cdir = tmp_path / "c"
    cdir.mkdir()
    for fi in range(4):
        v = rng.standard_normal((3000, 128)).astype(np.float16).astype(np.float32)
        if fi == 2:
            v[7] = 0.0
        pq.write_table(pa.table({
            "dense_embedding": pa.array(v.tolist(), type=pa.list_(pa.float32())),
            "id": [f"c{fi}_{r}" for r in range(3000)]}), str(cdir / f"f{fi}.parquet"))
    qp = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "dense_embedding": pa.array(
            rng.standard_normal((512, 128)).astype(np.float32).tolist(),
            type=pa.list_(pa.float32())),
        "qid": [f"q{i}" for i in range(512)]}), str(qp))
    real_launch = compute._twopass_launch
    launched = [0]

    def launch(*a, **kw):
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = real_launch(*a, **kw)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        launched[0] += len(out or {})
        for rec in (out or {}).values():
            assert rec.error is None, rec.error
        return out

    monkeypatch.setattr(compute, "_twopass_launch", launch)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qp), id_column="qid"),
        output=OutputConfig(path=str(tmp_path / "o")),
        params=ParamsConfig(dense_batch_size=1024, two_pass_rescore=mode,
                            pipeline_pass_one=True),
        searches=[SearchSpec(name=metric, metric=metric, k=10)])
    (tmp_path / "o").mkdir()
    twopass.reset()
    run_compute(cfg)
    st = twopass.stats()
    assert launched[0] > 0 and st["pipeline_used"] > 0, st
    assert st["slices_fused"] > 0 and st["slices_guard_refused"] > 0, st



@pytest.mark.parametrize("mode", ["slice", "tiles"])
def test_a_flag_a_refused_launch_flipped_stays_if_a_later_launch_used_it(
    ds, tmp_path, small, monkeypatch, mode
):
    """REGRESSION (review). The row-max kernel fails deterministically on
    slices R and S; only R is refused. Pipelined, A(R) flips `_ROWMAX_OFF`
    and A(S) runs under it before R's verdict is read; restoring the flag
    then would leave `row_max` 'not attempted' and retry (and re-warn) the
    kernel, where the unpipelined run -- R never launched, S failed and
    flipped it -- ends 'portable'. The flag must stay."""
    R = _slice_keys(ds, [(6, 120)])
    S = _slice_keys(ds, [(6, 160)])

    def install():
        cur = _spy_slice(monkeypatch)
        real_guard = _REAL["norm_guard"]
        monkeypatch.setattr(twopass, "norm_guard",
                            lambda cn: real_guard(cn) | True
                            if cur["key"] in R else real_guard(cn))
        real_rm = _REAL["approx_rowmax"]

        def rowmax(*a, **kw):
            if cur["key"] in (R | S) and not twopass._ROWMAX_OFF:
                twopass._ROWMAX_OFF = True
                twopass._ROWMAX_WARNED = True
            return real_rm(*a, **kw)

        monkeypatch.setattr(twopass, "approx_rowmax", rowmax)

    install()
    off, st_off = _run(ds, tmp_path / "off", "cosine", mode, False)
    install()
    on, st_on = _run(ds, tmp_path / "on", "cosine", mode, True)
    assert st_off["row_max"] == "portable", st_off["row_max"]
    _assert_identical(on, off, st_on, st_off, mode)
