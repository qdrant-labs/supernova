"""Pass one's guards are decided ON THE DEVICE, and launch without a host read.

`twopass._upper_bounds` used to read ~6-7 device scalars back before launching
pass one (`float(col_scale.max())`, `float(cn.min())`, `float(cn.max())`,
`bool(norm_guard(cn).any())`, the reads in `overflow_guard` and
`scale_mismatch`, `bool(bad.all())`). Each blocks the host until everything on
the compute stream -- the previous slice's pass two and fold -- has finished,
so the pipelined launch could not run ahead.

Now every per-slice guard is a device tensor folded into the bound (`bad` rows
are `+inf`; a slice-wide refusal marks every row), and the verdict is read
back ONCE, after the launch, in the finish half, where the reason, the
outcome and the counters are reported exactly as before. These tests pin:

  1. each guard still refuses what it refused, prunes nothing there, and
     reports the same reason / outcome / counters -- inline, split
     (begin/finish) and discarded, slice and tiles;
  2. each device guard decides exactly what its host twin decides, at the
     boundaries (same comparisons, same precision -- binary64 where the proof
     requires it);
  3. eps is evaluated for the slice's OWN key: a slice whose sigma_max /
     max N_c exceed the previous slice's key is never bounded with that
     slice's eps, and later slices hit the cache under their own key;
  4. the launch half performs no host read at all (CPU instrumentation; a
     CUDA sync-debug test where a GPU is present), and the finish half one.
"""

from __future__ import annotations

import math
import threading

import pytest

torch = pytest.importorskip("torch")

from nova_bf import closed_form as cf
from nova_bf import twopass

DIM = 64
NQ, NC = 24, 40


@pytest.fixture(autouse=True)
def _fresh():
    twopass.reset()
    yield
    twopass.reset()


def _unit(n, d=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def _args(Q, C, metric, cs_edit=None, rs_edit=None, cn_edit=None):
    cn = C.norm(dim=1)
    if cn_edit is not None:
        # The norms are a caller input (compute passes the slice's own); the
        # guards must refuse any the closed form does not admit.
        cn = cn_edit(cn.clone())
    if metric == "cosine":
        cs = cn.clamp_min(1e-12).reciprocal()
        rs = Q.norm(dim=1).clamp_min(1e-12).reciprocal()
        cs = cs if cs_edit is None else cs_edit(cs.clone())
        rs = rs if rs_edit is None else rs_edit(rs.clone())
    else:
        cs = rs = None
    return cs, rs, cn


def _bound(Q, C, metric, via="inline", tiles=False, exact=False, **edits):
    """`(upper, outcome, tile_dict, refusal)` through `via`."""
    cs, rs, cn = _args(Q, C, metric, **edits)
    td = {} if tiles else None
    kw = dict(cn=cn, corpus_exact_fp16=exact, with_parts=True,
              with_outcome=True, tiles=td)
    if via == "inline":
        (upper, _, _), outcome = twopass.upper_bounds(
            Q, C, cs, rs, None, metric, **kw)
        return upper, outcome, td, None
    p = twopass.upper_bounds_begin(Q, C, cs, rs, None, metric, **kw)
    if via == "discard":
        return None, None, td, twopass.upper_bounds_discard(p, resolve=True)
    (upper, _, _), outcome = twopass.upper_bounds_finish(p)
    return upper, outcome, td, p.refusal


def _set_row(i, f):
    def edit(t):
        t[i] = f(t[i])
        return t
    return edit


# Each: (name, builder -> (Q, C, edits), metric set, expected reason).
def _slice_cases():
    Q, C = _unit(NQ), _unit(NC, seed=1)

    def corpus(f):
        C2 = C.clone()
        f(C2)
        return C2

    yield "zero corpus row", lambda: (Q, corpus(lambda c: c[3].zero_()), {}), "corpus norm"
    yield "nan corpus row", lambda: (Q, corpus(lambda c: c[5].fill_(float("nan"))), {}), "corpus norm"
    yield "inf corpus entry", lambda: (Q, corpus(lambda c: c[7].__setitem__(0, float("inf"))), {}), "corpus norm"
    yield "norm above 2^126", lambda: (Q, corpus(lambda c: c[2].mul_(2.0 ** 127)), {}), "corpus norm"
    yield "norm below 2^-40", lambda: (Q, corpus(lambda c: c[2].mul_(2.0 ** -45)), {}), "corpus norm"
    yield "fp16 conversion overflow", lambda: (Q, corpus(lambda c: c[1].mul_(1e5)), {}), "corpus overflow"
    yield "prunability cut (corpus)", lambda: (Q, corpus(lambda c: c[9].mul_(1e-4)), {}), "all rows refused"
    yield "all queries below the cut", lambda: (Q * 1e-4, C, {}), "all rows refused"
    # The corpus-norm guard ALONE. A binary32 norm computed from data
    # overflows to inf past ~2^64, so N_c in (2^126, 2^128) only arrives as a
    # supplied norm: 2^127 here, with an fp16 claim (no conversion guard) and
    # half-norm queries (N_q * N_c = 2^126, at the product guard's limit).
    yield ("norm above 2^126 alone",
           lambda: (Q * 0.5, C, {"exact": True,
                                 "cn_edit": _set_row(2, lambda v: v * 0 + 2.0 ** 127)}),
           "corpus norm")


def _exact_cases():
    Q, C = _unit(NQ), _unit(NC, seed=1)
    # sigma_max ALONE out of range: every (supplied) N_c = 2^41, so sigma =
    # 2^-41 < NORM_MIN while matching its reciprocal, the norms in range and
    # (fp16 claim) no conversion guard.
    yield ("sigma_max below 2^-40 alone",
           lambda: (Q, C, {"exact": True,
                           "cn_edit": lambda t: t * 0 + 2.0 ** 41}),
           "corpus scale")


def _cosine_only_cases():
    Q, C = _unit(NQ), _unit(NC, seed=1)
    yield ("column scale off its reciprocal",
           lambda: (Q, C, {"cs_edit": _set_row(4, lambda v: v * 1.01)}), "column scale")
    yield ("column scale out of range",
           # sigma_max = max sigma must leave [2^-40, 2^126].
           lambda: (Q, C, {"cs_edit": lambda t: t * 0 + 2.0 ** -45}),
           "corpus scale")
    yield ("column scale shape", lambda: (Q, C, {"cs_edit": lambda t: t[:-1]}),
           "column scale")
    yield ("query scale shape", lambda: (Q, C, {"rs_edit": lambda t: t[:-1]}),
           "all rows refused")


_SLICE = list(_slice_cases())
_COSINE = list(_cosine_only_cases()) + list(_exact_cases())
_VIAS = ("inline", "split", "discard")


def _check_refused(case, metric, via, tiles):
    name, build, reason = case
    Q, C, edits = build()
    edits = dict(edits)
    exact = edits.pop("exact", False)
    upper, outcome, td, refusal = _bound(Q, C, metric, via=via, tiles=tiles,
                                         exact=exact, **edits)
    st = twopass.stats()
    if via == "discard":
        # The verdict is read only on request; nothing is booked here (the
        # caller, `compute._tp_discard`, books nothing for a refused launch).
        assert refusal == reason, (name, refusal)
        assert st["slices_guard_refused"] == 0, st
        assert twopass._FUSE_LAUNCHES == 0
        return
    # Prunes NOTHING: every row's bound is +inf ...
    assert bool(torch.isposinf(upper).all()), (name, upper)
    # ... and the report is exactly the guards-first one.
    assert outcome == twopass.PassOneOutcome(twopass.PASS_ONE_GUARD_REFUSED,
                                             reason), (name, outcome)
    if via == "split":
        assert refusal == reason
    assert st["slices_guard_refused"] == 1, (name, st)
    assert st["last_refusal_reason"] == reason, (name, st)
    # The speculative kernel (if any) is not booked: the guards-first code
    # never launched one.
    assert st["slices_fused"] + st["slices_unfused"] == 0, (name, st)
    assert st["eps_evaluations"] + st["eps_cache_hits"] == 0, (name, st)
    assert st["rowmax_worst_ratio"] is None, st
    if tiles:
        # No tile bounds: the caller gives every tile the (+inf) row bound.
        assert td == {}, (name, td)
        assert st["tile_width"] is None


@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("via", _VIAS)
@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("case", _SLICE, ids=[c[0] for c in _SLICE])
def test_each_slice_guard_refuses_the_whole_slice(case, metric, via, tiles):
    _check_refused(case, metric, via, tiles)


@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("via", _VIAS)
@pytest.mark.parametrize("case", _COSINE, ids=[c[0] for c in _COSINE])
def test_each_scale_guard_refuses_the_whole_slice(case, via, tiles):
    _check_refused(case, "cosine", via, tiles)


def test_the_first_guard_that_fired_is_the_reason():
    """Several guards fire: the reason is the earliest in the host order."""
    Q, C = _unit(NQ), _unit(NC, seed=1)
    C = C.clone()
    C[1] *= 1e5       # corpus overflow
    C[3] = 0          # corpus norm (earlier)
    for via in ("inline", "split", "discard"):
        twopass.reset()
        upper, outcome, _, refusal = _bound(Q, C, "cosine", via=via)
        assert (refusal if via == "discard" else outcome.reason) == "corpus norm"
    # Query side refuses every row AND a corpus guard fires: the corpus one.
    twopass.reset()
    _, outcome, _, _ = _bound(Q * 1e-4, C, "dot")
    assert outcome.reason == "corpus norm"


# --- per-row guards -----------------------------------------------------------

def _row_cases():
    Q, C = _unit(NQ), _unit(NC, seed=1)

    def query(i, f):
        Q2 = Q.clone()
        Q2[i] = f(Q2[i])
        return Q2

    yield "nan query", lambda: (query(4, lambda r: r * float("nan")), C, {}), [4], ("cosine", "dot")
    yield "zero query", lambda: (query(5, lambda r: r * 0), C, {}), [5], ("cosine", "dot")
    yield "inf query", lambda: (query(3, lambda r: r.index_fill(0, torch.tensor([0]), float("inf"))), C, {}), [3], ("cosine", "dot")
    yield "query below the cut", lambda: (query(6, lambda r: r * 1e-3), C, {}), [6], ("cosine", "dot")
    # 7e4: kappa * N_q > 65504, yet every component (~7e4 / 8) converts
    # finitely -- so only the guard, not an inf from the conversion, binds.
    yield "query fp16 overflow", lambda: (query(7, lambda r: r * 7e4), C, {}), [7], ("cosine", "dot")
    # dot: N_q * max N_c > 2^126 (the corpus is at 2^20, well inside fp16?
    # no -- so the query side's own overflow fires too; both are per row).
    yield "query norm product", lambda: (query(8, lambda r: r * 2.0 ** 110), C, {}), [8], ("dot",)
    yield ("query scale off its reciprocal",
           lambda: (Q, C, {"rs_edit": _set_row(2, lambda v: v * 1.01)}), [2], ("cosine",))


_ROWS = list(_row_cases())


@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("via", ["inline", "split"])
@pytest.mark.parametrize("case", _ROWS, ids=[c[0] for c in _ROWS])
def test_each_row_guard_refuses_only_its_rows(case, via, tiles):
    name, build, rows, metrics = case
    for metric in metrics:
        twopass.reset()
        Q, C, edits = build()
        upper, outcome, td, _ = _bound(Q, C, metric, via=via, tiles=tiles,
                                       **edits)
        assert outcome.kind == twopass.PASS_ONE_CPU_WIDENED, (name, outcome)
        inf = torch.isposinf(upper)
        assert inf[rows].all(), (name, metric, upper)
        others = torch.ones(NQ, dtype=torch.bool)
        others[rows] = False
        assert torch.isfinite(upper[others]).all(), (name, metric, upper)
        assert twopass.stats()["slices_guard_refused"] == 0
        if tiles:
            # Every tile bound of a refused row is the row's (+inf) bound.
            assert torch.isposinf(td["upper"][:, rows]).all()


def test_the_product_guard_alone_forces_its_row_live(monkeypatch):
    """End to end the product guard never fires alone: N_q * max N_c > 2^126
    needs a query whose fp16 copy is already inf, and the query's conversion
    guard refuses it first. So the conversion guard is switched off and pass
    one's result faked FINITE (zeros): the product guard's fold is then the
    only thing that can force row 8 live."""
    no = lambda d, qn, fmt: torch.zeros_like(qn, dtype=torch.bool)  # noqa: E731
    monkeypatch.setattr(twopass, "overflow_mask", no)
    monkeypatch.setattr(
        twopass, "approx_rowmax",
        lambda Qh, Ch, cs, out_dtype, with_tiles=False:
            (torch.zeros(Qh.shape[0]), False))
    # dot: N_q = 2^63 and N_c = 1.5 * 2^63 are both computable binary32
    # norms in range (2^126 = NORM_MAX; a larger one squares to inf), and
    # their product 1.5 * 2^126 is not. eps_dot stays finite.
    Q = _unit(NQ).clone()
    Q[8] *= 2.0 ** 63
    C = _unit(NC, seed=1) * (1.5 * 2.0 ** 63)   # fp16 claim: no corpus guard
    assert math.isfinite(float(C.norm(dim=1).max()))
    upper, outcome, _, _ = _bound(Q, C, "dot", via="split", exact=True)
    assert outcome.kind == twopass.PASS_ONE_CPU_WIDENED
    assert bool(torch.isposinf(upper[8]))
    keep = torch.ones(NQ, dtype=torch.bool)
    keep[8] = False
    assert bool(torch.isfinite(upper[keep]).all())


def test_an_empty_slice_and_a_bad_dimension_exit_on_the_host():
    Q, C = _unit(NQ), _unit(NC, seed=1)
    upper, outcome, _, _ = _bound(Q, C[:0], "dot", via="split")
    assert outcome.kind == twopass.PASS_ONE_EMPTY
    assert bool(torch.isneginf(upper).all())
    twopass.reset()
    Q32, C32 = _unit(NQ, 32), _unit(NC, 32, seed=1)
    upper, outcome, _, _ = _bound(Q32, C32, "cosine", via="split")
    assert outcome == twopass.PassOneOutcome(twopass.PASS_ONE_GUARD_REFUSED,
                                             "dimension")
    assert bool(torch.isposinf(upper).all())
    assert twopass.stats()["slices_guard_refused"] == 1


def test_the_query_refusals_include_the_query_norm_range(monkeypatch):
    """For every computable query norm the norm-range guard is subsumed (a
    binary32 norm past ~2^64 is inf, and anything below 2^-40 also fails the
    prunability cut). The proof still requires it, so `_query_refusals` --
    what the bound folds -- is checked to carry it with the prunability cut
    lowered to 0 and the conversion guard switched off."""
    monkeypatch.setattr(twopass, "overflow_mask",
                        lambda d, qn, fmt: torch.zeros_like(qn, dtype=torch.bool))
    monkeypatch.setattr(cf, "PRUNE_MIN_QNORM", 0.0)
    Q = _unit(NQ).clone()
    Q[3] *= 2.0 ** -45
    assert 0.0 < float(Q[3].norm()) < cf.NORM_MIN
    qs = twopass.query_side(Q, None)
    bad, every = twopass._query_refusals(qs, DIM, cf.FP16, None)
    assert bool(bad[3]) and not every
    assert int(bad.sum()) == 1


# --- 2. device guards decide exactly what their host twins decide ------------

def _edges(x):
    return [math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf)]


def _f32_values(extra):
    vals = [0.0, -0.0, -1.0, 1.0, float("nan"), float("inf"), -float("inf"),
            1e-45, 2.0 ** -149, 3.4e38, 1e-3, 1e-2, 2.0 ** -40, 2.0 ** 126,
            2.0 ** 127]
    for e in extra:
        vals.extend(_edges(e))
    # Only binary32 values reach the guards (norms and scales are float32).
    return sorted({float(torch.tensor(v, dtype=torch.float32)) for v in vals
                   if not math.isnan(v)}) + [float("nan")]


@pytest.mark.parametrize("d", [64, 768, 4096])
@pytest.mark.parametrize("fmt", [cf.FP16, cf.EXACT], ids=lambda f: f.name)
def test_overflow_device_matches_host(d, fmt):
    limit = fmt.omega_t * (1.0 - cf.GUARD_MARGIN) / cf.kappa_bar(d)
    for v in _f32_values([limit, 65504.0]):
        dev = bool(twopass.overflow_refused_t(d, torch.tensor(v), fmt))
        host = not twopass.overflow_guard(d, torch.tensor([v]), fmt)
        assert dev == host, (d, fmt.name, v)


def test_scale_range_device_matches_host():
    for v in _f32_values([cf.NORM_MIN, cf.NORM_MAX, 1.0]):
        dev = bool(twopass.scale_range_refused_t(torch.tensor(v)))
        host = (not cf.norm_range_ok(v)) and v != 1.0
        assert dev == host, v


def test_product_and_prunability_device_match_host():
    qs = _f32_values([1e-2, 2.0 ** 63, 2.0 ** 100])
    cns = _f32_values([1e-3, 2.0 ** 63, 2.0 ** 26])
    qn = torch.tensor(qs, dtype=torch.float32)
    for c in cns:
        ct = torch.tensor(c, dtype=torch.float32)
        assert torch.equal(twopass.product_guard_t(qn, ct),
                           twopass.product_guard(qn, c)), c
        assert torch.equal(twopass.prunability_cut_t(qn, ct),
                           twopass.prunability_cut(qn, c)), c


def test_the_product_guard_is_binary64_on_the_device():
    """The product of two binary32 norms is exact in binary64, so the guard
    sits exactly at 2^126: one binary32 ulp above it is refused, at it not.
    (A binary32 product could round an above-threshold product down onto
    the threshold; `product_guard_t` widens first, like its host twin.)"""
    at = 2.0 ** 63
    over = float(torch.nextafter(torch.tensor(at), torch.tensor(math.inf)))
    assert over > at
    assert bool(twopass.product_guard_t(torch.tensor([over]),
                                        torch.tensor(at))[0])
    assert not bool(twopass.product_guard_t(torch.tensor([at]),
                                            torch.tensor(at))[0])
    # 2^63 (1 + 2^-23) * 2^63 (1 - 2^-24) = 2^126 (1 + 2^-24 - 2^-47): above
    # 2^126, yet ROUNDS to 2^126 in binary32 and would pass there.
    q = torch.tensor(2.0 ** 63 * (1 + 2.0 ** -23))
    c = torch.tensor(2.0 ** 63 * (1 - 2.0 ** -24))
    assert float(q * c) == 2.0 ** 126
    assert bool(twopass.product_guard_t(q.reshape(1), c)[0])
    assert bool(twopass.product_guard(q.reshape(1), float(c))[0])


def test_scale_mismatch_mask_matches_scale_mismatch():
    g = torch.Generator().manual_seed(3)
    n = torch.rand(500, generator=g) * 10 + 1e-3
    s = n.reciprocal()
    s[::7] = torch.nextafter(s[::7], torch.full_like(s[::7], math.inf))
    s[::11] = s[::11] * (1 + 2.0 ** -20)
    s[5] = float("nan")
    n[9] = 0.0
    bad, _ = twopass.scale_mismatch(s, n)
    assert torch.equal(twopass.scale_mismatch_mask(s, n), bad)
    assert bool(bad.any())
    short = twopass.scale_mismatch_mask(s[:-1], n)
    assert bool(short.all()) and short.shape == n.shape


# --- 3. eps is evaluated for the slice's own key -------------------------------

def _eps_of(Q, C, metric, rs):
    cs, _, cn = _args(Q, C, metric)
    p = twopass.upper_bounds_begin(Q, C, cs, rs, None, metric, cn=cn,
                                   corpus_exact_fp16=False, with_parts=True)
    return p


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_a_key_change_mid_run_never_uses_the_stale_eps(metric):
    """Slice 2's sigma_max (cosine) / max N_c (dot) exceeds slice 1's key: it
    must be bounded with eps at ITS OWN key -- launched before slice 1 is
    finished, as pipelining does -- and slice 3 (back at slice 1's scale)
    takes slice 1's cached eps."""
    Q = _unit(NQ)
    C = _unit(NC, seed=1)
    # cosine: sigma = 1/N_c, so a small corpus raises the key; dot: max N_c.
    big = C * (0.01 if metric == "cosine" else 300.0)
    # ONE row scale for the run, as compute caches it: the eps cache lives
    # with the query side, keyed on the row-scale tensor's identity.
    rs = None if metric == "dot" else Q.norm(dim=1).clamp_min(1e-12).reciprocal()
    p1 = _eps_of(Q, C, metric, rs)
    p2 = _eps_of(Q, big, metric, rs)      # launched before p1 is finished
    _, _, eps1 = twopass.upper_bounds_finish(p1)
    p3 = _eps_of(Q, C, metric, rs)
    _, _, eps2 = twopass.upper_bounds_finish(p2)
    _, _, eps3 = twopass.upper_bounds_finish(p3)

    qs = twopass.query_side(Q, rs)
    cn_big = big.norm(dim=1)
    if metric == "cosine":
        own = cf.ceil_pow2(float(cn_big.clamp_min(1e-12).reciprocal().max()))
        want2 = twopass.bound(qs, DIM, own, True, metric, cf.FP16, cf.FP16, 0.0)
    else:
        own = cf.ceil_pow2(float(cn_big.max()))
        want2 = twopass.bound(qs, DIM, 1.0, True, metric, cf.FP16, cf.FP16, own)
    want2 = torch.as_tensor(want2, dtype=torch.float32).expand(NQ)
    assert torch.equal(eps2, want2), (eps2[:3], want2[:3])
    # The stale (slice 1) eps would be too small for slice 2.
    assert bool((eps1 < eps2).all()), (eps1[:3], eps2[:3])
    assert torch.equal(eps3, eps1)
    st = twopass.stats()
    assert st["eps_evaluations"] == 2 and st["eps_cache_hits"] == 1, st


# --- 4. no host read before the launch -------------------------------------------

_READS = ("__bool__", "__float__", "__int__", "__index__", "item", "tolist",
          "numpy", "cpu")


class _ReadCounter:
    """Counts device->host reads (on the calling thread) while `active`."""

    def __init__(self, monkeypatch):
        self.n = 0
        self.where = []
        self.active = False
        me = threading.get_ident()

        def wrap(name, real):
            def f(*a, **kw):
                if self.active and threading.get_ident() == me:
                    self.n += 1
                    self.where.append(name)
                return real(*a, **kw)
            return f

        for name in _READS:
            monkeypatch.setattr(torch.Tensor, name,
                                wrap(name, getattr(torch.Tensor, name)))
        monkeypatch.setattr(torch, "equal", wrap("equal", torch.equal))
        # Data-dependent shapes synchronise too: nonzero, masked_select and
        # boolean-mask indexing (`t[mask]`).
        for name in ("nonzero", "masked_select"):
            monkeypatch.setattr(torch.Tensor, name,
                                wrap(name, getattr(torch.Tensor, name)))
            monkeypatch.setattr(torch, name, wrap(name, getattr(torch, name)))
        real_getitem = torch.Tensor.__getitem__

        def getitem(t, idx):
            items = idx if isinstance(idx, tuple) else (idx,)
            if (self.active and threading.get_ident() == me
                    and any(isinstance(i, torch.Tensor) and i.dtype is torch.bool
                            for i in items)):
                self.n += 1
                self.where.append("t[mask]")
            return real_getitem(t, idx)

        monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
        real_to = torch.Tensor.to

        def to(t, *a, **kw):
            dev = kw.get("device", a[0] if a else None)
            if (self.active and threading.get_ident() == me
                    and str(dev).startswith("cpu")):
                self.n += 1
                self.where.append("to(cpu)")
            return real_to(t, *a, **kw)

        monkeypatch.setattr(torch.Tensor, "to", to)

    def __enter__(self):
        self.n, self.where, self.active = 0, [], True
        return self

    def __exit__(self, *exc):
        self.active = False


def _data(kind):
    Q, C = _unit(NQ), _unit(NC, seed=1)
    if kind == "zero corpus row":
        C = C.clone()
        C[3] = 0
    elif kind == "fp16 overflow":
        C = C.clone()
        C[1] *= 1e5
    elif kind == "nan query":
        Q = Q.clone()
        Q[4] = float("nan")
    return Q, C


@pytest.mark.parametrize("kind", ["normal", "zero corpus row", "fp16 overflow",
                                  "nan query"])
@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_the_launch_half_reads_nothing_back(monkeypatch, metric, tiles, kind):
    Q, C = _data(kind)
    cs, rs, cn = _args(Q, C, metric)
    kw = dict(cn=cn, corpus_exact_fp16=False, with_parts=True)
    # The query side is built (and read, once) per query matrix, e.g. by
    # certification; warm it as a run does.
    twopass.upper_bounds(Q, C, cs, rs, None, metric, **kw)
    counter = _ReadCounter(monkeypatch)
    for _ in range(2):      # a second slice: steady state
        td = {} if tiles else None
        with counter:
            p = twopass.upper_bounds_begin(Q, C, cs, rs, None, metric,
                                           tiles=td, **kw)
        assert p.launched
        assert counter.n == 0, counter.where
        with counter:
            twopass.upper_bounds_finish(p)
        # ONE batched read in the finish half -- including tiles mode's
        # tile-magnitude cap, which rides the same transfer.
        assert counter.where == ["tolist"], counter.where


def test_the_discard_half_reads_only_when_asked(monkeypatch):
    Q, C = _data("zero corpus row")
    cs, rs, cn = _args(Q, C, "cosine")
    kw = dict(cn=cn, corpus_exact_fp16=False, with_parts=True)
    twopass.upper_bounds(Q, C, cs, rs, None, "cosine", **kw)
    counter = _ReadCounter(monkeypatch)
    p = twopass.upper_bounds_begin(Q, C, cs, rs, None, "cosine", **kw)
    with counter:
        assert twopass.upper_bounds_discard(p) is None
    assert counter.n == 0, counter.where
    p = twopass.upper_bounds_begin(Q, C, cs, rs, None, "cosine", **kw)
    with counter:
        assert twopass.upper_bounds_discard(p, resolve=True) == "corpus norm"
    assert counter.n == 1, counter.where


def _cuda_fused_ok():
    if not torch.cuda.is_available():
        return False
    return getattr(twopass, "_gemm_rowmax", None) is not None


@pytest.mark.skipif(not _cuda_fused_ok(), reason="needs CUDA + the fused kernel")
@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("kind", ["normal", "zero corpus row"])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_the_launch_half_never_synchronises_on_cuda(metric, kind, tiles,
                                                    side_stream):
    """torch's own sync detector: any stream sync or blocking copy inside the
    launch half raises. Includes the guards, the conversions, the fused
    launch and (side stream) the cross-stream handoff."""
    dev = torch.device("cuda", torch.cuda.current_device())
    Q = torch.nn.functional.normalize(torch.randn(256, 768, device=dev), dim=1)
    C = torch.nn.functional.normalize(torch.randn(1024, 768, device=dev), dim=1)
    if kind == "zero corpus row":
        C[3] = 0
    cs, rs, cn = _args(Q, C, metric)
    kw = dict(cn=cn, corpus_exact_fp16=False, with_parts=True)
    # Warm: query side, Triton compile, kernel info.
    twopass.upper_bounds(Q, C, cs, rs, None, metric, **kw)
    stream = torch.cuda.Stream(device=dev) if side_stream else None
    torch.cuda.synchronize()
    td = {} if tiles else None
    torch.cuda.set_sync_debug_mode("error")
    try:
        p = twopass.upper_bounds_begin(Q, C, cs, rs, None, metric, tiles=td,
                                       stream=stream, **kw)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert p.launched
    # The fused kernel really ran (on a refused slice too: speculatively).
    assert p.fused is True, p.fused
    upper, _, _ = twopass.upper_bounds_finish(p)
    if kind == "zero corpus row":
        assert bool(torch.isposinf(upper).all())
        assert twopass.stats()["last_refusal_reason"] == "corpus norm"


def test_the_counter_sees_boolean_indexing(monkeypatch):
    counter = _ReadCounter(monkeypatch)
    x = torch.arange(4.0)
    with counter:
        x[x > 1]
        x.nonzero()
    assert counter.where == ["t[mask]", "nonzero"], counter.where


# --- the unpipelined path reads first and launches nothing on a refusal ------

def _count_rowmax(monkeypatch):
    calls = []
    real = twopass.approx_rowmax

    def rowmax(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)
    return calls


@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("case", _SLICE, ids=[c[0] for c in _SLICE])
def test_unpipelined_refusals_launch_nothing(monkeypatch, case, metric, tiles):
    """`upper_bounds` (unpipelined, certification) gains nothing from a
    launch ahead of the verdict, so it reads the verdict first -- ONE read --
    and a refused slice runs no pass-one GEMM, as before."""
    name, build, reason = case
    Q, C, edits = build()
    edits = dict(edits)
    exact = edits.pop("exact", False)
    cs, rs, cn = _args(Q, C, metric, **edits)
    kw = dict(cn=cn, corpus_exact_fp16=exact, with_parts=True)
    twopass.upper_bounds(Q, C, cs, rs, None, metric, **kw)   # warm query side
    twopass.reset()
    twopass.upper_bounds(Q, C, cs, rs, None, metric, **kw)
    calls = _count_rowmax(monkeypatch)
    counter = _ReadCounter(monkeypatch)
    twopass.reset_stats()
    with counter:
        (upper, _, _), outcome = twopass.upper_bounds(
            Q, C, cs, rs, None, metric, with_outcome=True,
            tiles={} if tiles else None, **kw)
    assert calls == [], name
    assert outcome == twopass.PassOneOutcome(twopass.PASS_ONE_GUARD_REFUSED,
                                             reason)
    assert bool(torch.isposinf(upper).all())
    # One read (the verdict), or none when the host already knew.
    assert counter.where in ([], ["tolist"]), counter.where


@pytest.mark.parametrize("tiles", [False, True])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_unpipelined_reads_at_most_twice(monkeypatch, metric, tiles):
    """A slice that runs: the verdict before the launch, then (cosine) the
    magnitude maxima after it -- two transfers in all, from ~8."""
    Q, C = _data("normal")
    cs, rs, cn = _args(Q, C, metric)
    kw = dict(cn=cn, corpus_exact_fp16=False, with_parts=True)
    twopass.upper_bounds(Q, C, cs, rs, None, metric, **kw)
    counter = _ReadCounter(monkeypatch)
    with counter:
        twopass.upper_bounds(Q, C, cs, rs, None, metric,
                             tiles={} if tiles else None, **kw)
    want = ["tolist", "tolist"] if metric == "cosine" else ["tolist"]
    assert counter.where == want, counter.where


# --- a failed speculative launch on a refused slice --------------------------

def _failing_rowmax(monkeypatch, exc, bump=True):
    """approx_rowmax that, like the real unfused path, books `slices_unfused`
    and then raises."""
    def rowmax(*a, **kw):
        if bump:
            twopass._STATS["slices_unfused"] += 1
        raise exc()

    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)


@pytest.mark.parametrize("exc", [
    lambda: twopass.PassOneUnavailable("CUDA out of memory"),
    lambda: RuntimeError("cublas boom")], ids=["oom", "error"])
@pytest.mark.parametrize("via", ["split", "discard"])
def test_a_failed_speculative_launch_on_a_refused_slice_reports_the_refusal(
    monkeypatch, exc, via
):
    """The guards-first code never launched on a refused slice, so never
    failed there: the failure is swallowed, its counters taken back, and the
    refusal reported exactly as before."""
    Q, C = _data("zero corpus row")
    _failing_rowmax(monkeypatch, exc)
    upper, outcome, _, refusal = _bound(Q, C, "cosine", via=via)
    st = twopass.stats()
    assert refusal == "corpus norm"
    if via == "split":
        assert outcome == twopass.PassOneOutcome(
            twopass.PASS_ONE_GUARD_REFUSED, "corpus norm")
        assert bool(torch.isposinf(upper).all())
        assert st["slices_guard_refused"] == 1
    assert st["slices_pass_one_oom"] == 0, st
    assert st["slices_unfused"] == 0, st


def test_a_failed_launch_on_a_live_slice_still_fails(monkeypatch):
    Q, C = _data("normal")
    _failing_rowmax(monkeypatch, lambda: twopass.PassOneUnavailable("oom"))
    _, outcome, _, _ = _bound(Q, C, "cosine", via="split")
    assert outcome.kind == twopass.PASS_ONE_OOM
    assert twopass.stats()["slices_pass_one_oom"] == 1
    twopass.reset()
    _failing_rowmax(monkeypatch, lambda: RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _bound(Q, C, "cosine", via="split")


# --- everything a refused speculative launch changed is taken back -----------

def test_a_refused_launch_takes_back_its_side_effects(monkeypatch):
    """Counters by delta (another slice's launch in between keeps its own),
    `ever ran` records with their counts, one-way flags it flipped."""
    real = twopass.approx_rowmax

    def rowmax(*a, **kw):
        twopass._FUSE_LAUNCHES += 1
        twopass._FUSE_KERNEL_INFO.setdefault("config", [1, 2, 3])
        twopass._ROWMAX_RUNS += 1
        twopass._ROWMAX_USED = True
        twopass._ROWMAX_OFF = True
        twopass._ROWMAX_WARNED = True
        twopass._FUSE_OOM_WARNED = True
        twopass.fuse_disable("kernel boom")      # not deferred: immediate
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)
    Qz, Cz = _data("zero corpus row")
    p = twopass.upper_bounds_begin(Qz, Cz, *_args(Qz, Cz, "dot")[:2], None,
                                   "dot", cn=Cz.norm(dim=1),
                                   corpus_exact_fp16=False, with_parts=True)
    assert twopass._FUSE_OFF == "kernel boom" and twopass._FUSE_LAUNCHES == 1
    twopass.upper_bounds_finish(p)
    assert twopass._FUSE_OFF is None
    assert twopass._FUSE_LAUNCHES == 0 and twopass._FUSE_KERNEL_INFO == {}
    assert twopass._ROWMAX_RUNS == 0 and twopass._ROWMAX_USED is False
    assert twopass._ROWMAX_OFF is False and twopass._ROWMAX_WARNED is False
    assert twopass._FUSE_OOM_WARNED is False
    st = twopass.stats()
    assert st["slices_unfused"] == 0 and st["row_max"] == "not attempted", st


def test_take_back_is_by_delta_across_interleaved_launches(monkeypatch):
    """Pipelining: slice N+1 launches before slice N's verdict is read. A
    refused N takes back only its own launch."""
    real = twopass.approx_rowmax

    def rowmax(*a, **kw):
        twopass._FUSE_LAUNCHES += 1
        twopass._FUSE_KERNEL_INFO.setdefault("config", [1, 2, 3])
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)
    Qz, Cz = _data("zero corpus row")
    _, Cn = _data("normal")
    kw = dict(corpus_exact_fp16=False, with_parts=True)
    p1 = twopass.upper_bounds_begin(Qz, Cz, None, None, None, "dot",
                                    cn=Cz.norm(dim=1), **kw)
    p2 = twopass.upper_bounds_begin(Qz, Cn, None, None, None, "dot",
                                    cn=Cn.norm(dim=1), **kw)
    twopass.upper_bounds_finish(p1)          # refused
    assert twopass._FUSE_LAUNCHES == 1 and twopass._FUSE_KERNEL_INFO
    twopass.upper_bounds_finish(p2)
    assert twopass._FUSE_LAUNCHES == 1
    assert twopass.stats()["slices_unfused"] == 1


def test_a_discarded_refused_launch_is_settled_by_its_verdict(monkeypatch):
    real = twopass.approx_rowmax

    def rowmax(*a, **kw):
        twopass._FUSE_LAUNCHES += 1
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)
    Qz, Cz = _data("zero corpus row")
    p = twopass.upper_bounds_begin(Qz, Cz, None, None, None, "dot",
                                   cn=Cz.norm(dim=1), corpus_exact_fp16=False)
    twopass.upper_bounds_discard(p)          # deferred: nothing read yet
    assert twopass._FUSE_LAUNCHES == 1 and p.refusal is None
    assert twopass.discard_resolved(p) == "corpus norm"
    assert twopass._FUSE_LAUNCHES == 0
    assert twopass.discard_resolved(p) == "corpus norm"   # idempotent
    assert twopass._FUSE_LAUNCHES == 0



def test_a_flag_is_restored_only_if_no_later_launch_ran_under_it(monkeypatch):
    """`_take_back` restores a one-way flag a refused launch flipped only if
    no launch has started since (`twopass._LAUNCH_GEN`): a later launch that
    ran under it is what the guards-first run would have failed on itself."""
    real = twopass.approx_rowmax

    def rowmax(*a, **kw):
        twopass._ROWMAX_OFF = True
        return real(*a, **kw)

    monkeypatch.setattr(twopass, "approx_rowmax", rowmax)
    Qz, Cz = _data("zero corpus row")
    _, Cn = _data("normal")
    kw = dict(corpus_exact_fp16=False, with_parts=True)
    # Alone: restored.
    p = twopass.upper_bounds_begin(Qz, Cz, None, None, None, "dot",
                                   cn=Cz.norm(dim=1), **kw)
    twopass.upper_bounds_finish(p)
    assert twopass._ROWMAX_OFF is False
    # A later launch ran under it: kept.
    p1 = twopass.upper_bounds_begin(Qz, Cz, None, None, None, "dot",
                                    cn=Cz.norm(dim=1), **kw)
    p2 = twopass.upper_bounds_begin(Qz, Cn, None, None, None, "dot",
                                    cn=Cn.norm(dim=1), **kw)
    twopass.upper_bounds_finish(p1)
    assert twopass._ROWMAX_OFF is True
    twopass.upper_bounds_finish(p2)
    # Counters are by delta either way.
    assert twopass.stats()["slices_unfused"] == 1


@pytest.mark.parametrize("how", ["reset", "reset_stats"])
def test_resetting_drops_unsettled_bookings(how):
    booked = []
    twopass.defer_booking(lambda: booked.append(1))
    getattr(twopass, how)()
    twopass.stats()
    assert booked == []
    twopass.defer_booking(lambda: booked.append(1))
    twopass.stats()
    assert booked == [1]


@pytest.mark.parametrize("refused", [True, False])
def test_an_oom_before_the_verdict_on_a_refused_slice_reports_the_refusal(
    monkeypatch, refused
):
    """The finish half's work before the verdict exists only because the
    launch ran ahead of it; on a refused slice an allocation failure there
    must not surface (the guards-first code did none of that work)."""
    Q, C = _data("zero corpus row" if refused else "normal")
    p = twopass.upper_bounds_begin(Q, C, *_args(Q, C, "cosine")[:2], None,
                                   "cosine", cn=C.norm(dim=1),
                                   corpus_exact_fp16=False, with_parts=True,
                                   with_outcome=True)

    def boom(*a, **kw):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(twopass, "_rowmax_worst", boom)
    if not refused:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            twopass.upper_bounds_finish(p)
        return
    (upper, _, _), outcome = twopass.upper_bounds_finish(p)
    assert outcome == twopass.PassOneOutcome(twopass.PASS_ONE_GUARD_REFUSED,
                                             "corpus norm")
    assert bool(torch.isposinf(upper).all())
    st = twopass.stats()
    assert st["slices_guard_refused"] == 1 and st["slices_unfused"] == 0, st


def test_a_verdict_read_early_is_not_read_again(monkeypatch):
    """`resolve_refusal` (phase B, before a deferred fused failure is
    applied) keeps its read; the finish half reuses it."""
    Q, C = _data("normal")
    cs, rs, cn = _args(Q, C, "dot")
    kw = dict(cn=cn, corpus_exact_fp16=False, with_parts=True)
    twopass.upper_bounds(Q, C, cs, rs, None, "dot", **kw)
    p = twopass.upper_bounds_begin(Q, C, cs, rs, None, "dot", **kw)
    counter = _ReadCounter(monkeypatch)
    with counter:
        assert twopass.resolve_refusal(p) is None
        twopass.upper_bounds_finish(p)
    assert counter.where == ["tolist"], counter.where
