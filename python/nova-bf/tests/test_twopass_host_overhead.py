"""The two-pass decision path's host bookkeeping per slice: how many times the
GPU-feeding thread reads the device, and that the cheaper forms are the same
values.

On a GPU every device->host read (and every pageable host->device copy) in
phase B blocks the host until the stream drains, and every small op is a
launch with the GPU idle behind it. Fix 2 trimmed both without changing a
decision:

  * `twopass._finite` is `torch.isfinite` (a four-kernel composite) in two
    kernels -- pinned here over every fp16/bf16 bit pattern and the fp32 /
    fp64 special values;
  * the audits read their five scalars (and compute's deferred eps maximum)
    in ONE transfer; the live count, the member spans and the live-pair count
    ride one transfer too, and the live index is built at the size read;
  * the eps device copy and the member bounds' device copy are cached, so a
    steady-state slice makes no host->device copy from Python data.

These tests pin the per-slice read sequence of an engaged slice (phase B),
that phase A still reads nothing, and that a steady-state slice materialises
no Python data on the device (CPU instrumentation; `lift_fresh` is what
`torch.tensor` / `torch.as_tensor` of host data dispatch).
"""

from __future__ import annotations

import collections
import math
import random
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils._python_dispatch import TorchDispatchMode

from nova_bf import compute
from nova_bf import twopass
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)

DIM, K, NQ, TILE = 64, 5, 96, 16


# --- `_finite` is `isfinite` ----------------------------------------------------

def test_finite_is_isfinite_on_every_half_precision_bit_pattern():
    pat = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    for dt in (torch.float16, torch.bfloat16):
        x = pat.view(dt)
        assert torch.equal(twopass._finite(x), torch.isfinite(x)), dt


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
def test_finite_is_isfinite_on_special_and_random_values(dt):
    g = torch.Generator().manual_seed(0)
    info = torch.finfo(dt)
    special = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"), info.max,
         -info.max, info.tiny, -info.tiny, info.tiny / 4, -info.tiny / 4,
         1.0, -1.0], dtype=dt)
    x = torch.cat((special, torch.randn(4096, generator=g, dtype=dt) * 1e30))
    assert torch.equal(twopass._finite(x), torch.isfinite(x))
    # NaN payloads, both signs: exponent all ones, any non-zero mantissa.
    if dt is torch.float32:
        nan = torch.arange(1, 1 << 16, dtype=torch.int32) | 0x7F800000
        for sign in (0, -0x80000000):
            y = (nan | sign).view(torch.float32)
            assert not twopass._finite(y).any()
            assert torch.equal(twopass._finite(y), torch.isfinite(y))


def test_an_audit_reads_a_device_eps_exactly_as_the_float():
    """compute passes the slice's eps maximum as a 0-d device tensor (read
    inside the audit's one transfer); the audit must decide and record
    exactly what the host float gave."""
    g = torch.Generator().manual_seed(1)
    idx = torch.tensor([0, 2, 3])
    sc = torch.randn(3, 32, generator=g)
    sc[1, :16] = float("-inf")
    tile_up = torch.randn(2, 5, generator=g) + 3.0
    eps = torch.tensor(2.0 ** -12, dtype=torch.float32)
    got = []
    for e in (float(eps), eps):
        twopass.reset()
        n = twopass.audit_live_tiles(sc, idx, tile_up, 16, eps_used=e)
        got.append((n, dict(twopass._STATS)))
    assert got[0] == got[1]
    assert got[0][1]["audit_closest_margin"] is not None


@pytest.mark.parametrize("dt", [torch.float16, torch.float32, torch.float64])
def test_const_is_the_value_as_a_0d_tensor_of_like(dt):
    """`_const(v, like)` stands in for a Python scalar in `torch.where`: the
    exact value (sign of zero included), `like`'s dtype and device, 0-d --
    and a cached tensor per (device, dtype, value), never another value's."""
    twopass.reset()
    like = torch.ones(3, dtype=dt)
    for v in (float("inf"), float("-inf"), 0.0, -0.0, float("inf"), 0.0):
        c = twopass._const(v, like)
        assert c.shape == () and c.dtype is dt and c.device == like.device
        assert float(c) == v and math.copysign(1.0, float(c)) == math.copysign(1.0, v)
        assert c is twopass._const(v, like)
        out = torch.where(torch.tensor([True, False, False]), like, c)
        ref = torch.where(torch.tensor([True, False, False]), like,
                          torch.full_like(like, v))
        assert out.dtype is dt and torch.equal(out.view(torch.uint8),
                                               ref.view(torch.uint8))
    twopass.reset()
    assert not twopass._CONSTS


def _old_spans(live, bnds):
    idx = live.nonzero(as_tuple=True)[0]
    return torch.searchsorted(
        idx, torch.tensor(bnds, dtype=idx.dtype)).tolist(), idx


def test_prefix_count_spans_and_static_index_equal_searchsorted_and_nonzero(monkeypatch):
    """`_tp_live_counts` + `_tp_live_index` against the code they replaced
    (`nonzero`, then `searchsorted` of the member bounds), on random masks
    and member layouts, n = 0 / 1 / all live / none live included."""
    monkeypatch.setattr(compute, "_NONZERO_STATIC_OFF", set())
    rng = random.Random(0)
    for trial in range(400):
        n = rng.choice([0, 1, 2, 7, 128, 1000, 4097])
        p = rng.choice([0.0, 1.0, 0.01, 0.5])
        live = torch.rand(n) < p
        cuts = sorted(rng.sample(range(n + 1), min(n + 1, rng.randint(0, 4))))
        pts = [0] + cuts + [n]
        bnds = [b for ab in zip(pts[:-1], pts[1:]) for b in ab]
        if rng.random() < 0.3:
            bnds = [0, n]
        tile_live = torch.rand(3, n) < p if rng.random() < 0.5 else None
        spans, n_live, n_pairs = compute._tp_live_counts(live, bnds, tile_live)
        want, idx_old = _old_spans(live, bnds)
        assert spans == want and all(type(s) is int for s in spans)
        assert n_live == int(idx_old.numel())
        assert n_pairs == (None if tile_live is None else int(tile_live.sum()))
        idx = compute._tp_live_index(live, n_live, n)
        assert idx.dtype is idx_old.dtype and torch.equal(idx, idx_old)


@pytest.mark.parametrize("static", [True, False])
def test_an_overcounted_live_index_fails_loudly(monkeypatch, static):
    """If the count read and the index ever disagreed, the index must not
    quietly select a real row (`nonzero_static`'s default pad, -1, is row
    n - 1): the pad is out of range so the next gather raises (on CUDA it
    device-asserts); without the kernel, the length check raises."""
    # Pin the memo either way: a test that left "cpu" in it would silently
    # route the static cases through the fallback.
    monkeypatch.setattr(compute, "_NONZERO_STATIC_OFF",
                        set() if static else {"cpu"})
    live = torch.tensor([False, True, False, True, False])
    n = int(live.numel())
    # Advanced indexing wraps a negative index (-1 is row n - 1, silently);
    # the pad is past the end, so it raises.
    with pytest.raises((IndexError, RuntimeError)):
        idx = compute._tp_live_index(live, 3, n)
        torch.randn(n, 4)[idx]
    with pytest.raises((IndexError, RuntimeError)):
        idx = compute._tp_live_index(live, 3, n)
        torch.randn(n, 4).index_select(0, idx)
    with pytest.raises((IndexError, RuntimeError)):
        idx = compute._tp_live_index(live, 3, n)
        torch.zeros(n, dtype=torch.int32).index_copy_(
            0, idx, torch.arange(3, dtype=torch.int32))
    assert torch.equal(compute._tp_live_index(live, 2, n),
                       torch.tensor([1, 3]))


@pytest.mark.parametrize("static", [True, False])
def test_an_undercounted_live_index_fails_loudly(monkeypatch, static):
    """The worse direction: fewer counted than live would drop live rows
    from the index -- pruning them. It must raise, with no read on the
    static path (the extra padding slot is checked by `_assert_async`)."""
    # Pin the memo either way: a test that left "cpu" in it would silently
    # route the static cases through the fallback.
    monkeypatch.setattr(compute, "_NONZERO_STATIC_OFF",
                        set() if static else {"cpu"})
    live = torch.tensor([True, False, True, True, False])
    for n_live in (0, 1, 2):
        with pytest.raises(RuntimeError):
            compute._tp_live_index(live, n_live, 5)
    assert torch.equal(compute._tp_live_index(live, 3, 5),
                       torch.tensor([0, 2, 3]))
    idx = compute._tp_live_index(live, 3, 5)
    assert idx.dtype is torch.int64 and idx.is_contiguous()
    none = torch.zeros(4, dtype=torch.bool)
    assert compute._tp_live_index(none, 0, 4).numel() == 0
    with pytest.raises(RuntimeError):
        compute._tp_live_index(torch.ones(4, dtype=torch.bool), 3, 4)
    assert torch.equal(compute._tp_live_index(torch.ones(4, dtype=torch.bool),
                                              4, 4), torch.arange(4))


def test_without_assert_async_the_index_takes_the_checked_fallback(
        monkeypatch):
    monkeypatch.setattr(compute, "_NONZERO_STATIC_OFF", set())
    monkeypatch.delattr(torch, "_assert_async")
    live = torch.tensor([True, False, True, True, False])
    assert torch.equal(compute._tp_live_index(live, 3, 5),
                       torch.tensor([0, 2, 3]))
    with pytest.raises(RuntimeError):
        compute._tp_live_index(live, 2, 5)


def test_a_missing_nonzero_static_kernel_falls_back_to_nonzero(monkeypatch):
    live = torch.tensor([True, False, True, True])

    def missing(*a, **kw):
        raise NotImplementedError("no kernel for this backend")

    monkeypatch.setattr(compute, "_NONZERO_STATIC_OFF", set())
    monkeypatch.setattr(torch, "nonzero_static", missing)
    assert torch.equal(compute._tp_live_index(live, 3, 4),
                       torch.tensor([0, 2, 3]))
    assert compute._NONZERO_STATIC_OFF == {"cpu"}


def test_the_bounds_and_query_id_caches_evict_without_changing_results():
    """Both caches are capped and cleared wholesale; anything evicted is
    rebuilt identically, and `twopass.reset` / `release` empty them."""
    twopass.reset()
    for i in range(150):
        b = [0, i + 1, i // 2, i + 1]
        t = compute._tp_bounds_tensor(b, torch.device("cpu"))
        assert t.dtype is torch.int64 and t.tolist() == b
        assert len(compute._TP_BOUNDS) <= 64
    assert compute._tp_bounds_tensor([0, 5], "cpu").tolist() == [0, 5]
    from nova_bf import tile_rescore as tr

    g = torch.Generator().manual_seed(0)
    masks = [torch.rand(nt, nq, generator=g) < 0.3
             for nt, nq in ((3, 5), (4, 9), (1, 1), (7, 3), (2, 8), (5, 5),
                            (3, 5), (4, 9))]
    for m in masks:
        got = tr.build_worklist(m)
        want = tr.build_worklist(m.clone())        # cached qids
        tr._QIDS.clear()
        fresh = tr.build_worklist(m)               # rebuilt qids
        for a, b, c in zip(got, want, fresh):
            assert torch.equal(a, b) and torch.equal(a, c)
        assert len(tr._QIDS) <= 4
        qids = tr._query_ids(m.shape[1], m.shape[0], m.device)
        assert torch.equal(qids, torch.arange(
            m.shape[1], dtype=torch.int32).repeat(m.shape[0]))
    def fill():
        twopass._const(1.0, torch.ones(1))
        compute._tp_bounds_tensor([0, 3], "cpu")
        tr.build_worklist(torch.ones(2, 3, dtype=torch.bool))
        assert compute._TP_BOUNDS and tr._QIDS and twopass._CONSTS

    for clear in (twopass.release, twopass.reset):
        fill()
        clear()
        assert not compute._TP_BOUNDS and not tr._QIDS and not twopass._CONSTS
    fill()
    tr.reset()
    assert not tr._QIDS


def test_a_row_audit_reads_a_device_eps_exactly_as_the_float():
    """`audit_live_rows`' counterpart of the tiles test above: slice mode
    passes the same 0-d device eps."""
    g = torch.Generator().manual_seed(2)
    idx = torch.tensor([0, 2, 3])
    scores = torch.randn(5, 32, generator=g)     # padded: 3 live rows
    scores[1, :] = float("-inf")
    upper = torch.randn(6, generator=g) + 3.0
    upper[2] = float("inf")
    rs = torch.rand(6, generator=g) + 0.5
    eps = torch.tensor(2.0 ** -12, dtype=torch.float32)
    for row_scale in (None, rs):
        got = []
        for e in (float(eps), eps):
            twopass.reset()
            n = twopass.audit_live_rows(scores, 3, idx, upper,
                                        row_scale=row_scale, eps_used=e)
            got.append((n, dict(twopass._STATS)))
        assert got[0] == got[1]
        assert got[0][1]["audit_closest_margin"] is not None


def _unit(n, d=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_the_eps_device_copy_is_reused_and_rebuilt_after_a_write(metric):
    """The eps device copy is cached per eps key, so a steady-state slice
    makes no host->device copy -- and a write through the returned tensor
    (`_version`) is never served to a later slice. (On CPU the copy aliases
    the cached numpy value, so only the cache's own bookkeeping is visible:
    on a GPU a stale copy would carry the write.)"""
    twopass.reset()
    Q, C = _unit(24), _unit(40, seed=1)
    cn = C.norm(dim=1)
    cs = cn.clamp_min(1e-12).reciprocal() if metric == "cosine" else None
    rs = (Q.norm(dim=1).clamp_min(1e-12).reciprocal() if metric == "cosine"
          else None)

    def eps_of():
        _, _, eps = twopass.upper_bounds(Q, C, cs, rs, None, metric, cn=cn,
                                         corpus_exact_fp16=False,
                                         with_parts=True)
        return eps

    def cached():
        (entry,) = twopass._QCACHE[id(Q)]["eps_dev"].values()
        return entry[1]

    first = eps_of()
    copy = cached()
    want = first.clone()
    assert torch.equal(eps_of(), want) and cached() is copy
    copy.add_(0.0)                       # a write: bumps `_version` only
    assert torch.equal(eps_of(), want)
    assert cached() is not copy, "a written-to eps copy was served again"


# --- reads per slice ---------------------------------------------------------------

def _write(path, vectors, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(),
                                        type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(5)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    g = 0
    for fi in range(4):
        vecs = rng.standard_normal((384, DIM)).astype(np.float16).astype(
            np.float32)
        _write(cdir / f"f{fi}.parquet", vecs,
               id=[f"c{g + r:06d}" for r in range(384)])
        g += 384
    qpath = tmp_path / "queries.parquet"
    _write(qpath, rng.standard_normal((NQ, DIM)).astype(np.float32),
           qid=[f"q{i}" for i in range(NQ)])
    return {"cdir": str(cdir), "qpath": str(qpath)}


@pytest.fixture
def small(monkeypatch):
    """Fixture scale, as in `test_twopass_tiles.py`. Not correctness knobs."""
    for k in ("NOVA_BF_TWOPASS_RESCORE", "NOVA_BF_NO_TWOPASS",
              "NOVA_BF_TWOPASS_AUDIT", "NOVA_BF_TWOPASS_DEAD_AUDIT",
              "NOVA_BF_PIPELINE_PASS_ONE", "NOVA_BF_TWOPASS_FORCE_ENGAGE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 8)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 8)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 16)
    monkeypatch.setattr(twopass, "CPU_TILE", TILE)
    monkeypatch.setattr(twopass, "TILE_CROSSOVER", 1.0)
    monkeypatch.setattr(twopass, "_AUDIT_RATE", None)
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_DEVICE", "cpu")
    twopass.reset()
    yield
    twopass.reset()


def _cfg(ds, out, metric, mode, pipeline):
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=128,
                            pipeline_pass_one=pipeline,
                            two_pass_rescore=mode),
        searches=[SearchSpec(name="all", metric=metric, k=K)],
    )


class _Lift(TorchDispatchMode):
    """Counts Python host data materialised as tensors (`torch.tensor` /
    `torch.as_tensor`), on this thread, while `on`."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        st = self.state
        if (st["active"] == threading.get_ident()
                and func.overloadpacket.__name__ in ("lift_fresh",
                                                     "lift_fresh_copy")):
            st["lifts"] += 1
        return func(*args, **(kwargs or {}))


def _install_read_counter(monkeypatch, state):
    """What `tests/test_pass_one_pipeline.py`'s counter counts: scalar and
    array conversions, `torch.equal`, `.to("cpu")` and the data-dependent
    shape ops, on the thread in `state["active"]`."""
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


# An engaged slice's reads in phase B (`_twopass_prepare`, which includes the
# pass-one finish), in order:
#   * the bound: ONE transfer after a pipelined launch; unpipelined, the
#     verdict before the launch, plus (cosine) the magnitude maxima after it;
#   * the live count, member spans and live-pair count: ONE transfer;
#   * the audit: ONE transfer -- in slice mode after its index range check,
#     which must precede the gather it protects.
# Shape verification's `torch.equal` (once per padded height, slice mode) is
# not per slice and is left out. In tiles mode, pipelined, that is 3 reads; it
# was 10 before fix 2 (eps max, `nonzero`, the spans, five audit scalars and
# a boolean index) plus two hidden host->device copies (eps, member bounds).
def _engaged_reads(mode, metric, pipeline):
    bound = ["tolist"] if pipeline or metric == "dot" else ["tolist", "tolist"]
    audit = ["tolist"] if mode == "tiles" else ["__int__", "tolist"]
    return bound + ["tolist"] + audit


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("mode", ["tiles", "slice"])
def test_an_engaged_slice_reads_once_per_decision(
        ds, tmp_path, small, monkeypatch, mode, metric, pipeline):
    state = {"active": None, "reads": [], "lifts": 0, "nzs": 0}
    _install_read_counter(monkeypatch, state)
    real_nzs = torch.nonzero_static

    def nzs(*a, **kw):
        if state["active"] == threading.get_ident():
            state["nzs"] += 1
        return real_nzs(*a, **kw)

    monkeypatch.setattr(torch, "nonzero_static", nzs)
    calls = {"A": [], "B": []}

    def instrument(name, key):
        real = getattr(compute, name)

        def f(*a, **kw):
            state["active"] = threading.get_ident()
            state["reads"], state["lifts"], state["nzs"] = [], 0, 0
            try:
                out = real(*a, **kw)
            finally:
                state["active"] = None
            calls[key].append((list(state["reads"]), state["lifts"], out,
                               state["nzs"]))
            return out
        monkeypatch.setattr(compute, name, f)

    instrument("_twopass_launch", "A")
    instrument("_twopass_prepare", "B")
    with _Lift(state):
        run_compute(_cfg(ds, tmp_path / "o", metric, mode, pipeline),
                    num_jobs=1, job_rank=0)
    st = twopass.stats()
    assert st["slices_twopass"] > 0 and st["audit_slices"] > 0, st
    if mode == "tiles":
        assert st["slices_tiles"] > 0, st
    if pipeline:
        assert st["pipeline_used"] > 0, st
    # Phase A (pipelined runs only): no read, ever.
    assert bool(calls["A"]) == pipeline, calls["A"]
    assert all(r == [] and n == 0 and z == 0
               for r, n, _, z in calls["A"]), calls["A"]
    # Phase B, EVERY call from the first slice on. The first call that
    # reads anything certifies the configuration (its probe battery reads
    # freely, and it fills the first-use caches: the eps and member-bounds
    # device copies); every later call is held to the trimmed sequence --
    # including the first slices after it, the cache-miss path -- and
    # materialises no host data. A pipelined run's first slice after
    # certification had no launch issued yet (the hint did not exist), so it
    # runs pass one inline: the unpipelined sequence.
    B = calls["B"]
    cert = next(i for i, (r, _, _, _) in enumerate(B) if r)
    assert all(not r and n == 0 and z == 0 for r, n, _, z in B[:cert]), B[:cert]
    allowed = [_engaged_reads(mode, metric, pipeline),
               _engaged_reads(mode, metric, False)]
    seen = []
    for reads, lifts, plans, n_static in B[cert + 1:]:
        assert lifts == 0, (reads, lifts)
        # The live index: one `nonzero_static` per plan, none for a slice
        # scored plain (one score group here).
        assert n_static == len(plans), (n_static, plans)
        if any(p.n_live for p in plans.values()):
            reads = [r for r in reads if r != "equal"]
            assert reads in allowed, reads
            seen.append(reads)
    assert seen and allowed[0] in seen, seen


def test_an_oom_in_the_tiled_dead_audit_mask_falls_back_per_query(
        ds, tmp_path, small, monkeypatch):
    """Every allocation of the tiled union -- the dead audit's row mask
    included -- sits inside its OOM fallback (`tiles_bound_oom`: decide the
    slice per query). Fails the SECOND 2-D `any(dim=0)` of each decision:
    with the mask computed in the union that is `tile_live.any` (inside the
    fallback); had the mask moved after it, the OOM would escape."""
    monkeypatch.setenv("NOVA_BF_TWOPASS_DEAD_AUDIT", "1")
    seen = {"n": 0, "on": False, "raised": 0}
    real_any = torch.Tensor.any

    def any_(t, *a, **kw):
        if seen["on"] and t.ndim == 2 and kw.get("dim", a[0] if a else None) == 0:
            seen["n"] += 1
            if seen["n"] == 2:
                seen["raised"] += 1
                raise torch.OutOfMemoryError("CUDA out of memory (injected)")
        return real_any(t, *a, **kw)

    real_prep = compute._twopass_prepare

    def prep(*a, **kw):
        seen["on"], seen["n"] = True, 0
        try:
            return real_prep(*a, **kw)
        finally:
            seen["on"] = False

    monkeypatch.setattr(torch.Tensor, "any", any_)
    monkeypatch.setattr(compute, "_twopass_prepare", prep)
    run_compute(_cfg(ds, tmp_path / "o", "cosine", "tiles", False),
                num_jobs=1, job_rank=0)
    st = twopass.stats()
    assert seen["raised"] > 0
    assert st["tiles_bound_oom"] == seen["raised"], st
    assert st["slices_twopass"] > 0, st
