"""Tiles-mode pass two (`params.two_pass_rescore: tiles`), on CPU.

Three layers, bottom up:

  * the per-tile BOUND (`upper_bounds(..., tiles=...)`): every exact score is
    <= its tile's bound, a tile bound is <= its row's bound (and the row bound
    is the max over its tiles, bit for bit), and a (query, tile) called dead
    holds no row reaching the threshold -- on the same random and adversarial
    data `test_twopass_bound.py` uses for rows;
  * the PASS TWO contract (`tile_rescore.reference`): `_scores`' arithmetic
    on live pairs, `-inf` on dead ones, with the very scale tensors slice mode
    and pass one use;
  * the PIPELINE (`run_compute`): tiles vs slice vs two-pass off, with
    filtered and unfiltered members sharing a score group, partial last tiles,
    files smaller than a slice, exact duplicate rows (real ties) -- and every
    fallback, which must hand back exactly the slice-mode result.

The pipeline runs go through the CPU opt-in (`NOVA_BF_TWOPASS_ON_CPU`) with
the same fixture-scale knobs as `test_twopass_pipeline.py`; `CPU_TILE` is
shrunk to 16 so a fixture slice has several tiles. The Triton kernel itself is
covered by `test_tile_rescore_interpret.py` (interpreter, here) and
`test_tile_rescore_gpu.py` (hardware, CUDA only).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf import compute
from nova_bf import tile_rescore
from nova_bf import twopass
from nova_bf.compute import _scores, run_compute
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
from test_twopass_bound import CASES, DIM as BOUND_DIM, _exact64, _fp32_scores, _make, _scales
from twopass_compare import assert_same_results, norm_product, tolerance

DIM, K, NQ = 64, 5, 48
BATCH = 72          # 4.5 tiles of TILE: every full slice ends in a partial tile
TILE = 16


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for k in ("NOVA_BF_TWOPASS_RESCORE", "NOVA_BF_NO_TWOPASS",
              "NOVA_BF_TWOPASS_NO_VERIFY", "NOVA_BF_NO_TILE_KERNEL",
              "NOVA_BF_TWOPASS_TILE_THRESHOLD"):
        monkeypatch.delenv(k, raising=False)
    twopass.reset()
    yield
    twopass.reset()


@pytest.fixture
def small(monkeypatch):
    """Fixture scale, as in `test_twopass_pipeline.py`. Not correctness knobs."""
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 8)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 8)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 16)
    monkeypatch.setattr(twopass, "CPU_TILE", TILE)
    monkeypatch.setattr(twopass, "TILE_CROSSOVER", 1.0)
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    # These are CPU-path tests (CPU_TILE, the torch reference, simulated GPU
    # declines). On a CUDA machine nova-bf would otherwise pick the GPU and the
    # real fused kernel's width; test_tile_rescore_gpu.py covers that path.
    monkeypatch.setenv("NOVA_BF_DEVICE", "cpu")


# --- 1. the per-tile bound ------------------------------------------------------


def _tile_bounds(Q, C, metric, out_dtype=None):
    cs, rs = _scales(C, Q, metric)
    info = {}
    upper = twopass.upper_bounds(Q, C, cs, rs, out_dtype, metric, tiles=info)
    return upper, info


@pytest.mark.parametrize("width", [16, 128, 48 + 16])
@pytest.mark.parametrize("out_dtype", [None, torch.float32],
                         ids=["gram_f16", "gram_f32"])
@pytest.mark.parametrize("fp16_corpus,q_scale,c_scale,metric,seed", CASES)
def test_every_exact_score_is_within_its_tile_bound(
    fp16_corpus, q_scale, c_scale, metric, seed, out_dtype, width, monkeypatch
):
    monkeypatch.setattr(twopass, "CPU_TILE", width)
    n_q, w = 256, 200          # 200 rows: the last tile is partial at every width
    Q, C = _make(n_q, w, BOUND_DIM, fp16_corpus=fp16_corpus, q_scale=q_scale,
                 c_scale=c_scale, seed=seed)
    upper, info = _tile_bounds(Q, C, metric, out_dtype)
    tup = info["upper"]
    n_tiles = -(-w // width)
    assert info["width"] == width and tup.shape == (n_tiles, n_q)

    # Every exact score, float32 and float64, is within its tile's bound.
    t32 = twopass.tile_amax(_fp32_scores(Q, C, metric), width).T
    t64 = twopass.tile_amax(_exact64(Q, C, metric), width).T
    assert bool((tup >= t32).all())
    assert bool((tup.double() >= t64 - 1e-12).all())

    # A tile bound never exceeds its row's, and the row's IS the max of its
    # tiles -- same arithmetic on the same maximum, so bit for bit.
    assert bool((tup <= upper[None, :]).all())
    fin = torch.isfinite(upper)
    assert torch.equal(tup.amax(dim=0)[fin], upper[fin])

    # The decision: walk thresholds across each tile's exact max. A pair the
    # tile bound calls dead never holds a row reaching the threshold.
    eps = (tup - t32).clamp_min(1e-9)
    for mult in (-1.0, 0.0, 0.5, 1.0, 4.0):
        thr = t32 + mult * eps
        dead = tup < thr
        assert bool((t32[dead] < thr[dead]).all())
        assert bool((t64[dead] < thr[dead].double()).all())


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_a_tile_holding_the_candidate_at_the_threshold_stays_live(metric):
    """The hard case per tile: one row EXACTLY at the threshold in one tile,
    the rest of that query's tiles far below it. The tile holding it must be
    live; the others may die, which is what tiles mode buys."""
    g = torch.Generator().manual_seed(9)
    Q = torch.randn(64, BOUND_DIM, generator=g)
    C = torch.randn(256, BOUND_DIM, generator=g) * 0.05
    C[200] = Q[3] * 3.0                       # the one strong match, tile 1
    _, info = _tile_bounds(Q, C, metric)
    exact = _fp32_scores(Q, C, metric)
    thr = exact[3, 200]
    tup = info["upper"]
    assert not bool(tup[200 // 128, 3] < thr), "the tile with the candidate died"
    assert bool(tup[0, 3] < thr), (
        "the far-below tile did not die, so this case shows nothing tile-level")


def test_guards_force_every_tile_of_the_guarded_row_live():
    """A query row any guard refuses is +inf in EVERY tile, whatever its tile
    maxima say -- the per-query bound is +inf and the tile bound takes it."""
    g = torch.Generator().manual_seed(2)
    Q = torch.randn(32, BOUND_DIM, generator=g)
    Q[5] = 0.0                                # zero norm: the norm guard
    C = torch.randn(100, BOUND_DIM, generator=g)
    cs = C.norm(dim=1).clamp_min(1e-12).reciprocal()
    rs = Q.norm(dim=1).clamp_min(1e-12).reciprocal()
    info = {}
    upper = twopass.upper_bounds(Q, C, cs, rs, None, "cosine", tiles=info)
    assert math.isinf(float(upper[5]))
    assert bool(torch.isposinf(info["upper"][:, 5]).all())
    assert bool(torch.isfinite(info["upper"][:, 4]).all())


def test_a_slice_wide_refusal_leaves_no_tile_bounds():
    """An exit that computes no tile maxima leaves the dict EMPTY, and the
    caller then gives every tile the (+inf) per-query bound."""
    g = torch.Generator().manual_seed(3)
    Q = torch.randn(16, BOUND_DIM, generator=g)
    C = torch.randn(40, BOUND_DIM, generator=g)
    C[7] = float("nan")                       # corpus norm guard: all live
    info = {}
    upper = twopass.upper_bounds(Q, C, None, None, None, "dot", tiles=info)
    assert bool(torch.isposinf(upper).all()) and info == {}


def test_the_unfused_tile_maxima_reduce_to_the_row_max_exactly():
    g = torch.Generator().manual_seed(5)
    Qh = torch.randn(30, 64, generator=g).half()
    Ch = torch.randn(70, 64, generator=g).half()
    cs = torch.rand(70, generator=g) + 0.5
    plain, fused = twopass.approx_rowmax(Qh, Ch, cs, None)
    out, fused2, part, width = twopass.approx_rowmax(Qh, Ch, cs, None,
                                                     with_tiles=True)
    assert not fused and not fused2 and width == twopass.CPU_TILE
    assert torch.equal(plain, out)
    assert torch.equal(part.amax(dim=0), out)
    assert part.shape == (1, 30)              # 70 rows < one 128-row tile


# --- 2. the pass-two contract ------------------------------------------------------


@pytest.mark.parametrize("metric,packer", [("dot", False), ("cosine", True),
                                           ("cosine", False)])
def test_the_reference_is_scores_arithmetic_on_live_pairs(metric, packer):
    """Live pairs carry EXACTLY what slice mode's `_scores` computes for the
    same gathered rows -- the same GEMM and the same divides in the same
    order -- and dead pairs `-inf`."""
    g = torch.Generator().manual_seed(11)
    Q = torch.randn(40, 64, generator=g)
    C = torch.randn(90, 64, generator=g)
    tl_ = torch.rand(-(-90 // 16), 40, generator=g) < 0.3
    idx = tl_.any(dim=0).nonzero().flatten()
    qn_full = Q.norm(dim=1)
    cn = C.norm(dim=1).clamp_min(1e-12) if metric == "cosine" else None
    qn = qn_full if (metric == "cosine" and not packer) else None
    got, used = tile_rescore.rescore(Q, C, idx, tl_, 16, cn=cn, qn=qn)
    assert not used, "CPU without the interpreter must take the reference"
    want = _scores(Q.index_select(0, idx), C, metric,
                   None if metric == "dot" else qn_full.index_select(0, idx),
                   scale_in_packer=packer)
    keep = tl_.index_select(1, idx).T.repeat_interleave(16, dim=1)[:, :90]
    assert torch.equal(got[keep], want[keep])
    assert bool(torch.isneginf(got[~keep]).all())


def test_rescore_rejects_a_mis_shaped_tile_mask():
    Q, C = torch.randn(8, 64), torch.randn(40, 64)
    with pytest.raises(ValueError, match="tile_live has shape"):
        tile_rescore.rescore(Q, C, torch.arange(8), torch.ones(2, 8, dtype=torch.bool), 16)


def test_the_self_check_sample_is_deterministic_bounded_and_keeps_the_ends():
    rows = tile_rescore._self_check_rows
    BQ = tile_rescore.BQ
    assert rows(0) == [] and rows(5) == [0, 1, 2, 3, 4]
    cap = tile_rescore.SELF_CHECK_ROWS
    assert rows(cap + BQ) == list(range(cap + BQ))
    for n in (cap + BQ + 1, 2 * cap - 1, 1000, 8161, 8192, 10 ** 6):
        got = rows(n)
        assert cap < len(got) <= cap + BQ and got[0] == 0 and got[-1] == n - 1, n
        assert all(a < b for a, b in zip(got, got[1:])), n
        assert got == rows(n), "the sample must not depend on the call"
        # Every lane of a query block, whatever the even spacing.
        assert {r % BQ for r in got} == set(range(BQ)), n
    # At 8161 the even spacing is exactly 32: without the run, one lane only.
    even = [(i * 8160) // (cap - 1) for i in range(cap)]
    assert {r % BQ for r in even} == {0}
    assert rows(100, cap=1)[0] == 0 and len(rows(100, cap=1)) == BQ + 1


def _self_check_case(nq=600, n_rows=90, width=16, seed=12, density=0.3, d=64):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(nq, d, generator=g)
    C = torch.randn(n_rows, d, generator=g)
    tl_ = torch.rand(-(-n_rows // width), nq, generator=g) < density
    idx = tl_.any(dim=0).nonzero().flatten()
    cn = C.norm(dim=1).clamp_min(1e-12)
    qn = Q.norm(dim=1)
    out = tile_rescore.reference(Q, C, idx, tl_, width, cn, qn)
    return Q, C, idx, tl_, width, cn, qn, out


def _sparse_case():
    """The review's shape, scaled down: a few live queries per tile, a
    partial last tile (4100 = 256 x 16 + 4 rows) whose live queries are
    all rows the row sample does not hold, and many other such tiles."""
    g = torch.Generator().manual_seed(0)
    nq, n_rows, width, d = 4096, 4100, 16, 32
    Q, C = torch.randn(nq, d, generator=g), torch.randn(n_rows, d, generator=g)
    tl_ = torch.rand(-(-n_rows // width), nq, generator=g) < 0.002
    tl_[-1] = False
    idx = tl_.any(dim=0).nonzero().flatten()
    rows = set(tile_rescore._self_check_rows(int(idx.numel())))
    off = [int(idx[i]) for i in range(int(idx.numel())) if i not in rows]
    tl_[-1, off[5:9]] = True               # already live elsewhere: idx unchanged
    assert torch.equal(tl_.any(dim=0).nonzero().flatten(), idx)
    cn, qn = C.norm(dim=1).clamp_min(1e-12), Q.norm(dim=1)
    out = tile_rescore.reference(Q, C, idx, tl_, width, cn, qn)
    sampled = torch.zeros(nq, dtype=torch.bool)
    sampled[idx[sorted(rows)]] = True
    unsampled = tl_.any(1) & ~(tl_ & sampled[None, :]).any(1)
    assert bool(unsampled[-1]) and int(unsampled.sum()) > 20, int(unsampled.sum())
    return Q, C, idx, tl_, width, cn, qn, out


def test_the_self_check_reference_and_tolerance_work_is_bounded(monkeypatch):
    """The first-launch check's full-row reference GEMM and fp64 tolerance
    GEMM see exactly the rows `_self_check_rows` names (at most
    SELF_CHECK_ROWS + BQ), whatever the launch's live count; the per-tile
    part sees at most 2 pairs per tile plus three blocks."""
    Q, C, idx, tl_, width, cn, qn, out = _self_check_case()
    n_live = int(idx.numel())
    assert n_live > 2 * tile_rescore.SELF_CHECK_ROWS
    seen = {}
    real_ref, real_tol = tile_rescore.reference, tile_rescore.score_tolerance
    real_pairs = tile_rescore._check_pairs

    def ref(Q_, C_, idx_, *a, **kw):
        seen["ref"] = idx_.clone()
        return real_ref(Q_, C_, idx_, *a, **kw)

    def tol(Q_, C_, idx_, *a, **kw):
        seen["tol"] = idx_.clone()
        return real_tol(Q_, C_, idx_, *a, **kw)

    def pairs(Q_, C_, out_, idx_, tiles, qids, *a):
        seen["pairs"] = int(tiles.numel())
        return real_pairs(Q_, C_, out_, idx_, tiles, qids, *a)

    monkeypatch.setattr(tile_rescore, "reference", ref)
    monkeypatch.setattr(tile_rescore, "score_tolerance", tol)
    monkeypatch.setattr(tile_rescore, "_check_pairs", pairs)
    tile_rescore._self_check(Q, C, idx, tl_, width, cn, qn, out)
    want = idx[tile_rescore._self_check_rows(n_live)]
    assert torch.equal(seen["ref"], want) and torch.equal(seen["tol"], want)
    assert (int(want.numel())
            <= tile_rescore.SELF_CHECK_ROWS + tile_rescore.BQ)
    assert int(want[0]) == int(idx[0]) and int(want[-1]) == int(idx[-1])
    assert 0 < seen["pairs"] <= 2 * tl_.shape[0] + 3 * tile_rescore.BQ
    assert tile_rescore.usage()["selfchecks"] == 1


def _expect_caught(Q, C, idx, tl_, width, cn, qn, out):
    with pytest.raises(tile_rescore.KernelMismatch):
        tile_rescore._self_check(Q, C, idx, tl_, width, cn, qn, out)
    assert "disagreed with the reference" in (tile_rescore.kernel_off() or "")
    assert tile_rescore.usage()["selfcheck_failures"] == 1


def test_a_correct_launch_passes_every_part_of_the_check():
    for case in (_self_check_case(), _sparse_case(),
                 _self_check_case(nq=60)):
        twopass.reset()
        tile_rescore._self_check(*case)
        assert tile_rescore.kernel_off() is None


@pytest.mark.parametrize("where", ["first", "last", "middle"])
@pytest.mark.parametrize("fault", ["score", "dead_pair"])
def test_the_sampled_self_check_still_catches_a_wrong_sampled_row(where, fault):
    """A wrong live score or a dead pair not at -inf, on a sampled row, fails
    the check, disables the kernel and raises -- as the full check did."""
    Q, C, idx, tl_, width, cn, qn, out = _self_check_case()
    rows = tile_rescore._self_check_rows(int(idx.numel()))
    r = {"first": rows[0], "last": rows[-1], "middle": rows[len(rows) // 2]}[where]
    live = ~torch.isneginf(out[r])
    j = int((live if fault == "score" else ~live).nonzero()[0])
    out[r, j] = out[r, j] + 1e-3 if fault == "score" else 0.0
    _expect_caught(Q, C, idx, tl_, width, cn, qn, out)


@pytest.mark.parametrize("fault", ["score", "unwritten"])
def test_a_fault_confined_to_the_partial_last_tile_is_caught(fault):
    """The review's probe: the row sample holds no live pair of most tiles,
    so a kernel wrong only in the last (partial) tile -- every live score
    +1.0, or the tile never written (left at -inf) -- used to pass."""
    Q, C, idx, tl_, width, cn, qn, out = _sparse_case()
    nt = tl_.shape[0]
    rows = tl_[-1].index_select(0, idx)
    assert bool(rows.any())
    blk = out[rows, (nt - 1) * width:]
    out[rows, (nt - 1) * width:] = (blk + 1.0 if fault == "score"
                                    else torch.full_like(blk, float("-inf")))
    _expect_caught(Q, C, idx, tl_, width, cn, qn, out)


def test_an_unwritten_live_pair_off_every_sampled_row_is_caught():
    """One live pair, on a row neither sampled nor any tile's edge query,
    left at -inf: the whole-launch dead-pair mask catches it."""
    Q, C, idx, tl_, width, cn, qn, out = _self_check_case()
    rows = set(tile_rescore._self_check_rows(int(idx.numel())))
    tiles, qids = tile_rescore._self_check_pairs(tl_)
    edge = set(torch.searchsorted(idx, qids).tolist())
    r = next(i for i in range(int(idx.numel())) if i not in rows | edge)
    j = int((~torch.isneginf(out[r])).nonzero()[0])
    out[r, j] = float("-inf")
    _expect_caught(Q, C, idx, tl_, width, cn, qn, out)


def test_a_wrong_score_in_an_unsampled_tile_edge_is_caught(monkeypatch):
    """The pair part on its own: a wrong score on a tile's last live query,
    on a row the row sample does not hold."""
    Q, C, idx, tl_, width, cn, qn, out = _sparse_case()
    rows = set(tile_rescore._self_check_rows(int(idx.numel())))
    pos = {int(q): i for i, q in enumerate(idx.tolist())}
    t, r = next((t, pos[int(tl_[t].nonzero()[-1])])
                for t in range(tl_.shape[0])
                if bool(tl_[t].any()) and pos[int(tl_[t].nonzero()[-1])] not in rows)
    out[r, t * width] += 1.0
    _expect_caught(Q, C, idx, tl_, width, cn, qn, out)


def test_a_lane_aliased_fault_is_caught_by_the_row_sample_alone(monkeypatch):
    """n_live = 8161 makes the even spacing exactly BQ = 32, so every evenly
    spaced row is lane 0 of an all-live tile's query blocks. A kernel wrong
    on lanes 16..31 only is caught by the BQ-row run -- with the pair part
    switched off to show it."""
    nq, width = 8161, 16
    g = torch.Generator().manual_seed(1)
    Q, C = torch.randn(nq, 32, generator=g), torch.randn(width, 32, generator=g)
    tl_ = torch.ones(1, nq, dtype=torch.bool)
    idx = torch.arange(nq)
    out = tile_rescore.reference(Q, C, idx, tl_, width)
    lanes = (torch.arange(nq) % tile_rescore.BQ) >= 16
    out[lanes] += 1.0
    e = torch.empty(0, dtype=torch.int64)
    monkeypatch.setattr(tile_rescore, "_self_check_pairs", lambda *a: (e, e))
    _expect_caught(Q, C, idx, tl_, width, None, None, out)


def test_a_small_launch_is_self_checked_on_every_row(monkeypatch):
    """At or under the cap the sample is the whole launch."""
    Q, C, idx, tl_, width, cn, qn, out = _self_check_case(nq=60)
    assert int(idx.numel()) <= tile_rescore.SELF_CHECK_ROWS
    out[-1][~torch.isneginf(out[-1])] += 1e-3
    _expect_caught(Q, C, idx, tl_, width, cn, qn, out)


def test_the_pairs_cover_each_live_tiles_ends_and_the_edge_blocks():
    BQ = tile_rescore.BQ
    tl_ = torch.zeros(4, 100, dtype=torch.bool)
    tl_[0, [3, 50]] = True
    tl_[2, :70] = True                         # hottest: 70 = 2 full + 6
    tl_[3, [9]] = True                         # last live tile, one query
    tiles, qids = tile_rescore._self_check_pairs(tl_)
    got = set(zip(tiles.tolist(), qids.tolist()))
    assert {(0, 3), (0, 50), (2, 0), (2, 69), (3, 9)} <= got
    assert {(2, q) for q in range(BQ)} <= got            # first block
    assert {(2, q) for q in range(2 * BQ, 70)} <= got    # partial last block
    assert all(t != 1 for t, _ in got)                   # dead tile: nothing
    assert all(bool(tl_[t, q]) for t, q in got)          # live pairs only


def test_each_compiled_configuration_is_self_checked_once():
    """`_CHECKED` keys on what selects a compile -- NT_P2 included, so a
    short last slice (a smaller tile-count bucket) is checked on its first
    launch -- and on Triton's integer and pointer-alignment
    specializations."""
    if tile_rescore.triton is None:
        pytest.skip("no triton")
    k = tile_rescore._compile_key
    dev = torch.device("cpu")

    def key(n_tiles, n_rows=2048, width=32, cn=None, tensors=()):
        return k(dev, width, cn, None, n_tiles,
                 (n_rows, 64, 64, 64, n_rows, n_tiles), tensors)

    base = key(64)
    assert base == key(64)
    assert key(33) == key(63)                        # one bucket, ints unaligned
    assert base != key(30)                           # another NT_P2 bucket
    assert base != key(64, n_rows=2047)              # n_rows % 16
    assert base != key(64, width=16)
    assert base != key(64, cn=object())
    buf = torch.zeros(8, dtype=torch.int32)
    assert buf.data_ptr() % 16 == 0
    assert key(64, tensors=(buf,)) != key(64, tensors=(buf[1:],)), (
        "a misaligned view compiles separately and must be checked")
    assert key(64, tensors=(buf,)) == key(64, tensors=(buf[4:],))


@pytest.mark.parametrize("n_tiles", [50, 51, 52, 1001, 1004])
def test_the_work_list_block_count_is_its_own_aligned_allocation(n_tiles):
    """`nblk` used to be the view `bcum[-1:]`, at byte offset 4 * n_tiles:
    its 16-byte alignment (a Triton specialization) flipped with
    n_tiles % 4 -- 50 and 52 tiles, same key, different compiles."""
    tl_ = torch.ones(n_tiles, 4, dtype=torch.bool)
    *_, bcum, nblk = tile_rescore.build_worklist(tl_)
    assert nblk.data_ptr() % 16 == 0
    assert nblk.data_ptr() != bcum[-1:].data_ptr()
    assert int(nblk) == int(bcum[-1]) == n_tiles


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_live_pair_off_every_sampled_row_is_caught(value):
    """The whole-launch mask also requires every live pair to be finite: a
    NaN or +inf on a row neither sampled nor any tile's edge query."""
    Q, C, idx, tl_, width, cn, qn, out = _self_check_case()
    rows = set(tile_rescore._self_check_rows(int(idx.numel())))
    tiles, qids = tile_rescore._self_check_pairs(tl_)
    edge = set(torch.searchsorted(idx, qids).tolist())
    r = next(i for i in range(int(idx.numel())) if i not in rows | edge)
    j = int((~torch.isneginf(out[r])).nonzero()[0])
    out[r, j] = value
    with pytest.raises(tile_rescore.KernelMismatch, match="mask"):
        tile_rescore._self_check(Q, C, idx, tl_, width, cn, qn, out)


def test_the_failure_names_the_part_that_failed():
    """A mask-only failure says so, with no "off by" (negative) figure; a
    score failure on a sampled row names the rows and a positive excess."""
    Q, C, idx, tl_, width, cn, qn, out = _self_check_case()
    rows = set(tile_rescore._self_check_rows(int(idx.numel())))
    tiles, qids = tile_rescore._self_check_pairs(tl_)
    edge = set(torch.searchsorted(idx, qids).tolist())
    r = next(i for i in range(int(idx.numel())) if i not in rows | edge)
    bad = out.clone()
    bad[r, int((~torch.isneginf(bad[r])).nonzero()[0])] = float("-inf")
    with pytest.raises(tile_rescore.KernelMismatch) as e:
        tile_rescore._self_check(Q, C, idx, tl_, width, cn, qn, bad)
    msg = str(e.value)
    assert "mask" in msg and "off by" not in msg, msg
    twopass.reset()
    bad = out.clone()
    r = sorted(rows)[3]
    j = int((~torch.isneginf(bad[r])).nonzero()[0])
    bad[r, j] += 1.0
    with pytest.raises(tile_rescore.KernelMismatch) as e:
        tile_rescore._self_check(Q, C, idx, tl_, width, cn, qn, bad)
    msg = str(e.value)
    assert "rows (a score off by" in msg and "mask" not in msg, msg
    assert "off by -" not in msg, msg


def test_the_tolerance_pays_for_the_divides_and_its_own_fp64_rounding():
    """`_tolerance` is a true upper bound: the gamma_d term on an inflated
    fp64 magnitude, plus 8u of the score for the two sides' divides."""
    u = 2.0 ** -24
    d = 64
    mag = torch.tensor([1.0, 0.0], dtype=torch.float64)
    score = torch.tensor([0.5, 3.0], dtype=torch.float64)
    plain = tile_rescore._tolerance(d, mag)
    full = tile_rescore._tolerance(d, mag, score)
    gamma = d * u / (1.0 - d * u)
    # The fp64 inflation, pinned: (d + 4) * 2^-52, not less.
    want = 2.0 * gamma * (1.0 * (1.0 + (d + 4) * 2.0 ** -52)) * (1.0 + 4.0 * u)
    assert float(plain[0]) >= want + 1e-37
    assert float(plain[0]) > 2.0 * gamma * (1.0 + (d + 3) * 2.0 ** -52) * (
        1.0 + 4.0 * u) + 1e-37
    assert torch.allclose(full - plain, 8.0 * u * score, rtol=0, atol=1e-30)


def _exact_dot_case():
    """Small-integer operands: every fp32 dot is EXACT (any order), so the
    reference score R is exact and a planted error is known to the ulp. At
    d = 2 the gamma term is only ~4u |R|, so the 8u |R| divide term decides
    an error of ~6.7u |R| (R = 76: 4 ulp of 2^-17)."""
    Q = torch.tensor([[3.0, 5.0], [1.0, 2.0]])
    C = torch.tensor([[7.0, 11.0]] + [[1.0, 1.0]] * 15)
    tl_ = torch.ones(1, 2, dtype=torch.bool)
    idx = torch.arange(2)
    out = tile_rescore.reference(Q, C, idx, tl_, 16)
    assert float(out[0, 0]) == 76.0
    u = 2.0 ** -24
    out[0, 0] = float(torch.tensor(76.0 * (1.0 + 7.0 * u), dtype=torch.float32))
    err = float(out[0, 0]) - 76.0
    assert 4.5 * u * 76 < err < 8.0 * u * 76, err
    return Q, C, idx, tl_, out


def test_the_pair_check_charges_the_divides_on_the_score():
    """`_check_pairs` passes the reference score to `_tolerance`: an error
    above the gamma term but within it + 8u |R| is accepted."""
    Q, C, idx, tl_, out = _exact_dot_case()
    tiles = torch.tensor([0], dtype=torch.int64)
    qids = torch.tensor([0], dtype=torch.int64)
    assert tile_rescore._check_pairs(Q, C, out, idx, tiles, qids, 16,
                                     None, None) <= 0.0
    # And the gamma term alone would NOT accept it (the test's premise).
    mag = torch.tensor([76.0], dtype=torch.float64)
    assert float(out[0, 0]) - 76.0 > float(tile_rescore._tolerance(2, mag)[0])


def test_the_row_check_charges_the_divides_on_the_score(monkeypatch):
    """Same for the row part (`score_tolerance(..., score=want)`), with the
    pair part switched off so only the rows judge the planted error."""
    Q, C, idx, tl_, out = _exact_dot_case()
    e = torch.empty(0, dtype=torch.int64)
    monkeypatch.setattr(tile_rescore, "_self_check_pairs", lambda *a: (e, e))
    tile_rescore._self_check(Q, C, idx, tl_, 16, None, None, out)
    assert tile_rescore.kernel_off() is None


def test_a_self_check_oom_falls_back_for_that_slice_and_rechecks_later(
    ds, tmp_path, small, monkeypatch
):
    """An allocation failure inside the self-check is not a verdict: that
    slice takes the slice path, the kernel stays on, the configuration is
    not marked checked, and a later launch checks it (and passes)."""
    class _Right:
        def __getitem__(self, grid):
            def launch(Q, C, CN, QN, QLIST, TOFF, TCNT, BCUM, NBLK, OUT, POS,
                       *a, **kw):
                OUT.copy_(_launch_ref[0]())
            return launch

    _launch_ref = [None]
    real_rescore = tile_rescore.rescore

    def rescore(Q, C, idx, tl_, width, cn=None, qn=None):
        _launch_ref[0] = lambda: tile_rescore.reference(Q, C, idx, tl_, width,
                                                        cn, qn)
        return real_rescore(Q, C, idx, tl_, width, cn=cn, qn=qn)

    calls = [0]
    real_check = tile_rescore._self_check

    def check(*a):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real_check(*a)

    monkeypatch.setattr(tile_rescore, "_tile_rescore", _Right())
    monkeypatch.setattr(tile_rescore, "_kernel_ok",
                        lambda *a, **kw: tile_rescore.kernel_off() is None)
    monkeypatch.setattr(tile_rescore, "rescore", rescore)
    monkeypatch.setattr(tile_rescore, "_self_check", check)
    off, _ = _run(ds, tmp_path / "off", "cosine", "off")
    ti, st = _run(ds, tmp_path / "t", "cosine", "tiles")
    assert calls[0] >= 2, calls
    assert st["tiles_fallbacks"] >= 1 and st["tiles_selfcheck_failures"] == 0, st
    assert "out of memory" in (st["tiles_fallback_reason"] or ""), st
    assert tile_rescore.kernel_off() is None
    assert tile_rescore.usage()["selfchecks"] >= 1 and st["tiles_kernel"] > 0, st
    for name in off:
        assert_same_results(ti[name], off[name], abs_tol=_tol(ds, "cosine"),
                            label=name)


def test_the_kernel_source_pins_ieee_fp32_and_correctly_rounded_divides():
    """(P3): IEEE fp32 dot, no TF32; divides correctly rounded. A plain `/`
    in Triton is the approximate fp32 divide."""
    import inspect

    src = inspect.getsource(tile_rescore)
    body = src[src.index("def _tile_rescore("):src.index("_UNAVAILABLE = None")]
    assert 'input_precision="ieee"' in body
    assert body.count("tl.math.div_rn(") == 2
    assert " / " not in body


# --- 3. the pipeline ---------------------------------------------------------------


def _write(path, vectors, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(),
                                        type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture
def ds(tmp_path):
    """Several 200-row files (72-row slices + an 56-row tail that is not
    two-passed), a 37-row file (one slice SMALLER than the batch: a full
    slice with a partial tile) and exact duplicates of earlier rows in a later
    file, so real score ties reach the tie-break."""
    rng = np.random.default_rng(23)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    sizes = [200, 200, 37, 200, 144, 200]
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
    return {"cdir": str(cdir), "qpath": str(qpath),
            "B": norm_product(cdir, qpath)}


def _tol(ds, metric):
    """The honest |tiles - one-pass| bound: 2*gamma_d*B (`twopass_compare`)."""
    return tolerance(DIM, 1.0 if metric == "cosine" else ds["B"])


def _cfg(ds, out, metric, mode, filtered=True, tiebreak="ordinal"):
    searches = [
        SearchSpec(name="all", metric=metric, k=K),
        SearchSpec(name="half", metric=metric, k=K,
                   rows={"column": "qset", "isin": ["b"]}),
    ]
    if filtered:
        searches.append(SearchSpec(
            name="eng", metric=metric, k=K,
            filter=Filter(must=[FilterCondition(field="lang", match="eng")])))
    params = dict(io_workers=1, dense_batch_size=BATCH, tiebreak=tiebreak)
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


def _run(ds, out, metric, mode, **kw):
    out.mkdir(parents=True, exist_ok=True)
    twopass.reset()
    paths = run_compute(_cfg(ds, out, metric, mode, **kw), num_jobs=1, job_rank=0)
    return {n: pq.read_table(p) for n, p in paths.items()}, twopass.stats()


class _Spy:
    """Wrap `tile_rescore.rescore`, recording what each tiles slice decided."""

    def __init__(self, monkeypatch, inner=None):
        self.calls = []
        self._inner = inner or tile_rescore.rescore
        monkeypatch.setattr(tile_rescore, "rescore", self)

    def __call__(self, Q, C, idx, tile_live, width, cn=None, qn=None):
        self.calls.append(dict(Q=Q, C=C, idx=idx.clone(), tl=tile_live.clone(),
                               width=width, cn=cn, qn=qn))
        return self._inner(Q, C, idx, tile_live, width, cn=cn, qn=qn)


def _assert_tiles_did_tile_work(st, spy):
    assert st["rescore"] == "tiles" and st["slices_tiles"] > 0, st
    assert st["tile_pairs_live"] < st["tile_pairs_full"], st
    assert st["tile_width"] == TILE and st["tiles_fallbacks"] == 0, st
    assert spy.calls, "no tiles slice reached pass two"
    # Some live query was dead in some tile: the tile-level decision did
    # something the row-level one could not.
    assert any(bool((c["tl"].index_select(1, c["idx"]).sum(0)
                     < c["tl"].shape[0]).any()) for c in spy.calls if c["idx"].numel())
    # A partial last tile, and a slice smaller than the batch, both took it.
    assert any(c["C"].shape[0] % TILE for c in spy.calls)
    assert any(c["C"].shape[0] == 37 for c in spy.calls)


@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("filtered", [True, False])
def test_tiles_agrees_with_slice_and_one_pass(ds, tmp_path, small, monkeypatch,
                                              metric, filtered):
    off, _ = _run(ds, tmp_path / "off", metric, "off", filtered=filtered)
    sl, st_s = _run(ds, tmp_path / "slice", metric, "slice", filtered=filtered)
    assert st_s["slices_twopass"] > 0 and st_s["slices_tiles"] == 0
    spy = _Spy(monkeypatch)
    ti, st_t = _run(ds, tmp_path / "tiles", metric, "tiles", filtered=filtered)
    _assert_tiles_did_tile_work(st_t, spy)
    assert st_t["audit_violations"] == 0 and st_t["audit_tile_pairs"] > 0
    for name in off:
        assert sl[name].equals(off[name]), f"slice mode is no longer exact: {name}"
        assert_same_results(ti[name], off[name], abs_tol=_tol(ds, metric),
                            label=f"{metric}/{name}")


@pytest.mark.parametrize("metric", ["cosine", "dot"])
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_with_a_full_height_gemm_tiles_is_byte_identical(ds, tmp_path, small,
                                                          monkeypatch, metric,
                                                          tiebreak):
    """The decision and the plumbing, isolated from summation order.

    With the reference's GEMM run at FULL height (rows identical to the
    one-pass product), every live pair's score is the one-pass score, so the
    output must be the one-pass output exactly -- any pair wrongly called dead,
    any mis-scattered row, any -inf leaking into a result, or a broken union
    across members is a visible difference here rather than a tolerance."""
    def full_height(Q, C, idx, tile_live, width, cn=None, qn=None):
        s = Q @ C.T
        if cn is not None:
            s.div_(cn[None, :])
        if qn is not None:
            s.div_(qn[:, None])
        s = s.index_select(0, idx)
        keep = tile_live.index_select(1, idx).T.repeat_interleave(width, dim=1)
        return s.masked_fill_(~keep[:, :C.shape[0]], float("-inf"))

    monkeypatch.setattr(tile_rescore, "reference", full_height)
    monkeypatch.setenv("NOVA_BF_POISON_DEAD_ROWS", "1")
    off, _ = _run(ds, tmp_path / "off", metric, "off", tiebreak=tiebreak)
    spy = _Spy(monkeypatch)
    ti, st = _run(ds, tmp_path / "tiles", metric, "tiles", tiebreak=tiebreak)
    _assert_tiles_did_tile_work(st, spy)
    for name in off:
        assert ti[name].equals(off[name]), f"{metric}/{tiebreak}/{name} differs"


def test_members_with_different_k_union_their_live_tiles(ds, tmp_path, small,
                                                        monkeypatch):
    """Two whole-matrix members of one score group, the LAST with the higher
    thresholds (k=1): each (query, tile) live for either must be re-scored.
    A tile mask assigned per member instead of unioned keeps only the last
    member's tiles and silently drops the k=5 member's candidates."""
    def cfg(mode, out):
        params = {"io_workers": 1, "dense_batch_size": BATCH}
        params.update({"two_pass": "off"} if mode == "off"
                      else {"two_pass_rescore": mode})
        return BruteForceConfig(
            corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
            queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(**params),
            searches=[SearchSpec(name="wide", metric="cosine", k=8),
                      SearchSpec(name="top1", metric="cosine", k=1)])

    def run(mode, out):
        out.mkdir()
        twopass.reset()
        return ({n: pq.read_table(p) for n, p in run_compute(cfg(mode, out)).items()},
                twopass.stats())

    off, _ = run("off", tmp_path / "off")
    ti, st = run("tiles", tmp_path / "tiles")
    assert st["slices_tiles"] > 0 and st["tile_pairs_live"] < st["tile_pairs_full"]
    for name in off:
        assert_same_results(ti[name], off[name], abs_tol=_tol(ds, "cosine"),
                            label=name)


def test_tiles_mode_uses_the_slice_modes_scale_tensors(ds, tmp_path, small,
                                                       monkeypatch):
    """(P1): pass two divides by the SAME computed norms pass one and eps
    use -- the slice's `col_norms()` (the clamp of `col_norms_raw()`, which
    `upper_bounds` reads) and the group's `q_norms` -- never recomputed ones.
    And their values are bit-identical to what slice mode's `_scores`
    divides by (`C.norm(dim=1).clamp_min(1e-12)`, the gathered `q_norms`)."""
    seen = []
    real = compute._tp_rescore_tiles

    def spy_prepare(g, sl, metric, idx, tile_live, tile_w):
        seen.append((g, sl))
        return real(g, sl, metric, idx, tile_live, tile_w)

    monkeypatch.setattr(compute, "_tp_rescore_tiles", spy_prepare)
    spy = _Spy(monkeypatch)
    _run(ds, tmp_path / "t", "cosine", "tiles")
    assert spy.calls and len(spy.calls) == len(seen)
    for c, (g, sl) in zip(spy.calls, seen):
        assert c["cn"] is sl.col_norms(), "a recomputed corpus norm"
        assert torch.equal(c["cn"], sl.Cb.norm(dim=1).clamp_min(1e-12))
        assert c["Q"] is g["Q"] and c["C"] is sl.Cb
        if g["scale_in_packer"]:
            assert c["qn"] is None, "the query scale would be applied twice"
        else:
            assert c["qn"] is g["q_norms"]

    # And the direct call with the packer off hands over the group's q_norms.
    Q = torch.randn(20, 64)
    sl = compute.DenseBatchSlice(torch.randn(30, 64), False)
    g = {"Q": Q, "q_norms": Q.norm(dim=1), "scale_in_packer": False}
    tl_ = torch.ones(-(-30 // TILE), 20, dtype=torch.bool)
    spy.calls.clear()
    compute._tp_rescore_tiles(g, sl, "cosine", torch.arange(20), tl_, TILE)
    assert spy.calls[-1]["qn"] is g["q_norms"]
    assert spy.calls[-1]["cn"] is sl.col_norms()


@pytest.mark.parametrize("exc", [RuntimeError("boom"),
                                 torch.OutOfMemoryError("CUDA out of memory")])
def test_a_failed_tile_rescore_falls_back_to_the_slice_result(
    ds, tmp_path, small, monkeypatch, exc
):
    """Any failure in the tile re-score hands the slice to the slice-mode
    pass two -- whose output is byte-identical to one-pass -- never to a
    partial or wrong answer."""
    def broken(*a, **kw):
        raise exc

    monkeypatch.setattr(tile_rescore, "rescore", broken)
    off, _ = _run(ds, tmp_path / "off", "cosine", "off")
    ti, st = _run(ds, tmp_path / "tiles", "cosine", "tiles")
    assert st["tiles_fallbacks"] > 0 and st["slices_tiles"] == 0, st
    assert type(exc).__name__ in st["tiles_fallback_reason"]
    assert st["slices_twopass"] > 0, "the fallback did not take the slice path"
    ran = st["slices_fused"] + st["slices_unfused"]
    assert ran == st["slices_twopass"] + st["slices_discarded"], st
    for name in off:
        assert ti[name].equals(off[name]), name


def test_a_kernel_launch_failure_disables_the_kernel_not_the_mode(
    ds, tmp_path, small, monkeypatch
):
    """A deterministic kernel failure: that slice falls back to slice mode,
    the kernel is off for the run, and later slices use the torch reference."""
    class _Raiser:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                raise ValueError("compile failed")
            return launch

    monkeypatch.setattr(tile_rescore, "_tile_rescore", _Raiser())
    monkeypatch.setattr(tile_rescore, "_kernel_ok",
                        lambda *a, **kw: tile_rescore.kernel_off() is None)
    off, _ = _run(ds, tmp_path / "off", "dot", "off")
    ti, st = _run(ds, tmp_path / "tiles", "dot", "tiles")
    assert st["tiles_fallbacks"] == 1, st
    assert st["tiles_reference"] > 0 and st["tiles_kernel"] == 0, st
    assert "compile failed" in (tile_rescore.kernel_off() or "")
    for name in off:
        assert_same_results(ti[name], off[name], abs_tol=_tol(ds, "dot"),
                            label=name)


def test_the_manifest_records_the_mode_the_gate_and_the_tile_work(
    ds, tmp_path, small
):
    _run(ds, tmp_path / "t", "cosine", "tiles")
    doc = json.loads(next((tmp_path / "t").rglob("*manifest*/*.json")).read_text())
    assert doc["params"]["two_pass_rescore"] == "tiles"
    tp = doc["params"]["kernels"]["twopass"]
    assert tp["rescore"] == "tiles" and tp["engage_fraction"] == "tile"
    assert tp["engage_threshold"] == 1.0
    assert tp["tile_width"] == TILE and tp["slices_tiles"] > 0
    assert 0 < tp["tile_live_fraction"] < 1
    assert tp["tile_live_fraction"] == round(
        tp["tile_pairs_live"] / tp["tile_pairs_full"], 6)
    assert tp["tile_rescore_kernel"]["launches"] == 0      # CPU: the reference
    assert tp["counters_consistent"]

    _run(ds, tmp_path / "s", "cosine", "slice")
    doc = json.loads(next((tmp_path / "s").rglob("*manifest*/*.json")).read_text())
    tp = doc["params"]["kernels"]["twopass"]
    assert tp["rescore"] == "slice" and tp["engage_fraction"] == "row"
    assert tp["slices_tiles"] == 0 and tp["tile_live_fraction"] is None


def test_the_tiles_hint_is_a_tile_fraction_and_gates_on_its_own_cutoff(
    ds, tmp_path, small, monkeypatch
):
    """Tiles mode seeds and updates `_TP_LIVE_HINT` with a (query, tile)
    fraction and compares it with `NOVA_BF_TWOPASS_TILE_THRESHOLD`, leaving
    slice mode's row cutoff alone."""
    hints = []
    real = compute._twopass_prepare
    rs = _Spy(monkeypatch)

    def spy(tp_groups, *a, **kw):
        before = len(rs.calls)
        out = real(tp_groups, *a, **kw)
        if len(rs.calls) > before:
            (g,) = tp_groups.values()
            tl_ = rs.calls[-1]["tl"]
            hints.append((compute._TP_LIVE_HINT[g["key"]],
                          int(tl_.sum()) / tl_.numel()))
        return out

    monkeypatch.setattr(compute, "_twopass_prepare", spy)
    _, st = _run(ds, tmp_path / "t", "cosine", "tiles")
    # After a tiles slice the hint IS that slice's live-pair fraction.
    assert hints and all(h == want for h, want in hints), hints
    # Over the run the tile fraction sits below the row fraction.
    assert (st["tile_pairs_live"] / st["tile_pairs_full"]
            < st["rows_live"] / st["rows_full"])

    # A tile cutoff of 0 refuses every slice whose hint is above zero, while
    # the (open) slice cutoff is not consulted at all.
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "0.0")
    _, st0 = _run(ds, tmp_path / "t0", "cosine", "tiles")
    assert st0["slices_tiles"] < st["slices_tiles"]
    assert st0["engage_threshold"] == 0.0


def test_the_probe_measures_tile_liveness_in_final_units():
    g = torch.Generator().manual_seed(1)
    raw = torch.randn(12, 40, generator=g)
    cs = torch.rand(40, generator=g) + 0.5           # fused-cnorm column divide
    qs = torch.rand(8, generator=g) + 0.5            # packer query divide
    qsel = slice(2, 10)
    final = raw[qsel] / cs[None, :] / qs[:, None]
    thr = twopass.tile_amax(final, 16)[:, 1].clone()  # exactly at tile 1's max
    from nova_bf.tiebreak import pack

    thr_packed = pack(thr[:, None], torch.zeros(1, dtype=torch.int64))[:, 0]
    got = compute._tp_probe_tiles(raw, qsel, thr_packed, qs, cs, 16)
    want = ~(twopass.tile_amax(final, 16) < thr[:, None])
    assert torch.equal(got, want) and bool(got[:, 1].all())


# --- the audits, tightened to tiles ---------------------------------------------


def _buf_case():
    g = torch.Generator().manual_seed(6)
    Q, C = torch.randn(10, 64, generator=g), torch.randn(50, 64, generator=g)
    tl_ = torch.rand(-(-50 // 16), 10, generator=g) < 0.5
    tl_[:, 0] = True
    idx = tl_.any(dim=0).nonzero().flatten()
    buf = tile_rescore.reference(Q, C, idx, tl_, 16)
    exact = twopass.tile_amax(Q @ C.T, 16).T          # (n_tiles, n_q)
    return buf, idx, exact


def test_the_tile_audit_passes_a_true_bound_and_catches_a_tile_violation():
    buf, idx, exact = _buf_case()
    assert twopass.audit_live_tiles(buf, idx, exact + 1e-6, 16) == 0
    assert twopass.enabled() and twopass.stats()["audit_tile_pairs"] > 0

    # Below the exact max on ONE non-maximal tile of query 0 -- a row-level
    # audit (row max vs row bound) cannot see this; the tile audit must.
    bad = exact + 1e-6
    t = int(exact[:, 0].argmin())
    bad[t, 0] = exact[t, 0] - 1e-3
    assert twopass.audit_live_tiles(buf, idx, bad, 16) == 1
    assert not twopass.enabled()


def test_the_decision_audit_grades_tile_decisions():
    g = torch.Generator().manual_seed(8)
    Q, C = torch.randn(6, 64, generator=g), torch.randn(40, 64, generator=g)
    tmax = twopass.tile_amax(Q @ C.T, 16).T
    thr = tmax.amin(dim=0)                       # every query live in >= 1 tile
    tile_live = ~(tmax < thr[None, :])
    out = twopass.audit_decisions(Q, C, "dot", tile_live.any(0), thr,
                                  tile_live=tile_live, tile_width=16)
    assert out["false_prune"] == 0 and out["checked"] == tile_live.numel()
    wrong = tile_live.clone()
    wrong[int(tmax[:, 0].argmax()), 0] = False   # kill query 0's best tile
    out = twopass.audit_decisions(Q, C, "dot", wrong.any(0), thr,
                                  tile_live=wrong, tile_width=16)
    assert out["false_prune"] == 1 and not twopass.enabled()


# --- the knob -------------------------------------------------------------------


def test_the_default_is_tiles_and_slice_is_opt_in(monkeypatch, caplog):
    from nova_bf import config

    assert ParamsConfig().two_pass_rescore == config.TWO_PASS_RESCORE_DEFAULT == "tiles"
    assert twopass.rescore_mode() == "tiles" and twopass.tiles_mode()
    with pytest.raises(Exception):
        ParamsConfig(two_pass_rescore="rows")
    twopass.set_rescore_mode("slice")
    assert twopass.rescore_mode() == "slice" and not twopass.tiles_mode()
    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", "tiles")
    assert twopass.rescore_mode() == "tiles"          # the env override wins
    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", "bogus")
    assert twopass.rescore_mode() == "slice"          # ignored, with a warning
    assert any("NOVA_BF_TWOPASS_RESCORE" in r.getMessage() for r in caplog.records)
    twopass.reset()
    assert twopass.rescore_mode() == "tiles"


def test_the_tile_cutoff_defaults_to_its_own_value(monkeypatch):
    monkeypatch.delenv("NOVA_BF_TWOPASS_THRESHOLD", raising=False)
    assert twopass.threshold("tiles") == twopass.DEFAULT_TILE_THRESHOLD == 0.2
    assert twopass.threshold("slice") == twopass.DEFAULT_THRESHOLD == 0.5
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "0.9")
    assert twopass.threshold("tiles") == 0.2          # slice's knob only
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "0.35")
    assert twopass.threshold("tiles") == 0.35
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.5")
    assert twopass.threshold("tiles") == 0.2


def test_only_a_tiles_run_is_stamped_and_identity_ignores_the_mode(tmp_path):
    from nova_bf.results import config_identity, provenance

    def cfg(mode):
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(tmp_path)),
            queries=QueriesConfig(path=str(tmp_path / "q.parquet")),
            output=OutputConfig(path=str(tmp_path / "o")),
            params=ParamsConfig(two_pass_rescore=mode),
            searches=[SearchSpec(name="s", metric="cosine", k=3)])

    s, t = cfg("slice"), cfg("tiles")
    sp = s.searches[0]
    assert config_identity(s, sp) == config_identity(t, sp)
    assert b"nova_bf.two_pass_rescore" not in provenance(s, sp)
    assert provenance(t, sp)[b"nova_bf.two_pass_rescore"] == b"tiles"


# --- review hardening -------------------------------------------------------------


def test_past_the_crossover_a_tiles_slice_takes_the_slice_path(ds, tmp_path,
                                                               small,
                                                               monkeypatch):
    """Above `TILE_CROSSOVER` live pairs the slice uses slice mode's own rule
    (padded, verified GEMM), so the output is the one-pass output exactly."""
    monkeypatch.setattr(twopass, "TILE_CROSSOVER", 0.0)
    off, _ = _run(ds, tmp_path / "off", "cosine", "off")
    ti, st = _run(ds, tmp_path / "t", "cosine", "tiles")
    assert st["tiles_crossover_bails"] > 0 and st["slices_tiles"] == 0, st
    assert st["slices_twopass"] > 0 and st["gemms"] > 0, st
    for name in off:
        assert ti[name].equals(off[name]), name


def test_a_gpu_slice_whose_kernel_declines_takes_the_slice_path(
    ds, tmp_path, small, monkeypatch, caplog
):
    """On a GPU the torch reference is never used: a declining kernel hands
    the slice to the verified slice path, warns once with the reason, and
    the manifest records it. (Simulated: the CPU is treated as a GPU.)"""
    monkeypatch.setattr(compute, "_tiles_reference_allowed", lambda dev: False)
    off, _ = _run(ds, tmp_path / "off", "dot", "off")
    with caplog.at_level("WARNING", logger="nova_bf.compute"):
        ti, st = _run(ds, tmp_path / "t", "dot", "tiles")
    assert st["tiles_kernel_declined"] > 0 and st["slices_tiles"] == 0, st
    assert st["tiles_reference"] == 0, st
    assert st["tiles_kernel_decline_reason"] == "not a CUDA device"
    warned = [r for r in caplog.records if "cannot use its kernel" in r.getMessage()]
    assert len(warned) == 1 and "not a CUDA device" in warned[0].getMessage()
    for name in off:
        assert ti[name].equals(off[name]), name
    doc = json.loads(next((tmp_path / "t").rglob("*manifest*/*.json")).read_text())
    assert doc["params"]["kernels"]["twopass"]["tiles_kernel_decline_reason"] \
        == "not a CUDA device"


@pytest.mark.parametrize("why,kw", [
    ("tile width 512", dict(width=512)),
    ("tile width 24", dict(width=24)),
    ("tiles in the slice", dict(n_tiles=5000)),
    ("NOVA_BF_NO_TILE_KERNEL", dict(env=True)),
])
def test_every_kernel_decline_names_its_reason(monkeypatch, why, kw):
    monkeypatch.setenv("TRITON_INTERPRET", "1")     # past the device check
    if kw.get("env"):
        monkeypatch.setenv("NOVA_BF_NO_TILE_KERNEL", "1")
    Q, C = torch.randn(4, 64), torch.randn(32, 64)
    tl_ = torch.ones(kw.get("n_tiles", 2), 4, dtype=torch.bool)
    got = tile_rescore.decline_reason(Q, C, tl_, kw.get("width", 16))
    if tile_rescore._tile_rescore is None and not kw.get("env"):
        pytest.skip("no triton: every decline is 'triton unavailable'")
    assert got is not None and why in got, got
    assert tile_rescore.decline_reason(Q, C, torch.ones(2, 4, dtype=torch.bool),
                                       16) is None or kw.get("env")


def test_a_kernel_that_disagrees_with_the_reference_is_switched_off(
    ds, tmp_path, small, monkeypatch
):
    """The first-launch self-check: a kernel writing wrong scores is caught
    against the reference, disabled for the run, and its slice falls back to
    the slice path; later slices use the reference (CPU)."""
    class _Wrong:
        def __getitem__(self, grid):
            def launch(Q, C, CN, QN, QLIST, TOFF, TCNT, BCUM, NBLK, OUT, *a, **kw):
                OUT.zero_()                      # live AND dead pairs at 0.0
            return launch

    monkeypatch.setattr(tile_rescore, "_tile_rescore", _Wrong())
    monkeypatch.setattr(tile_rescore, "_kernel_ok",
                        lambda *a, **kw: tile_rescore.kernel_off() is None)
    off, _ = _run(ds, tmp_path / "off", "cosine", "off")
    ti, st = _run(ds, tmp_path / "t", "cosine", "tiles")
    assert st["tiles_selfcheck_failures"] == 1 and st["tiles_fallbacks"] == 1, st
    assert "disagreed with the reference" in (tile_rescore.kernel_off() or "")
    assert st["tiles_reference"] > 0, st
    for name in off:
        assert_same_results(ti[name], off[name], abs_tol=_tol(ds, "cosine"),
                            label=name)


def test_a_huge_tile_maximum_forces_its_whole_row_live(monkeypatch):
    """Defence in depth: every finite scaled tile maximum is held to the
    magnitude cap, not only the row max. A tile far below the row max but
    outside the bound's hypotheses forces the row (all its tiles) live."""
    real = twopass.approx_rowmax

    def doctored(Qh, Ch, cs, out_dtype, with_tiles=False):
        out, fused, part, w = real(Qh, Ch, cs, out_dtype, with_tiles=True)
        part = part.clone()
        part[0, 3] = -1e6                        # tile 0 of query 3: absurd
        return out, fused, part, w

    monkeypatch.setattr(twopass, "approx_rowmax", doctored)
    g = torch.Generator().manual_seed(12)
    Q = torch.randn(8, BOUND_DIM, generator=g)
    C = torch.randn(300, BOUND_DIM, generator=g)
    cs, rs = _scales(C, Q, "cosine")
    info = {}
    upper = twopass.upper_bounds(Q, C, cs, rs, None, "cosine", tiles=info)
    assert math.isinf(float(upper[3])) and bool(torch.isposinf(info["upper"][:, 3]).all())
    assert bool(torch.isfinite(upper[[0, 1, 2, 4]]).all())


def test_an_oom_building_tile_bounds_leaves_the_per_query_bound(monkeypatch):
    def oom(*a, **kw):
        raise torch.OutOfMemoryError("CUDA out of memory (test)")

    g = torch.Generator().manual_seed(13)
    Q = torch.randn(8, BOUND_DIM, generator=g)
    C = torch.randn(300, BOUND_DIM, generator=g)
    cs, rs = _scales(C, Q, "cosine")
    want = twopass.upper_bounds(Q, C, cs, rs, None, "cosine")
    monkeypatch.setattr(twopass, "_tile_scaled", oom)
    info = {}
    got = twopass.upper_bounds(Q, C, cs, rs, None, "cosine", tiles=info)
    assert info == {} and torch.equal(got, want)
    assert twopass.stats()["tiles_bound_oom"] == 1


def test_an_oom_building_the_tile_mask_decides_the_slice_per_query(
    ds, tmp_path, small, monkeypatch
):
    def oom(*a, **kw):
        raise torch.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(compute, "_tile_mask", oom)
    off, _ = _run(ds, tmp_path / "off", "cosine", "off")
    ti, st = _run(ds, tmp_path / "t", "cosine", "tiles")
    assert st["tiles_bound_oom"] > 0 and st["slices_tiles"] == 0, st
    assert st["slices_twopass"] > 0, st
    for name in off:
        assert ti[name].equals(off[name]), name


def _cfg_prov(ds, out, mode):
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=BATCH,
                            two_pass_rescore=mode),
        searches=[SearchSpec(name="cos", metric="cosine", k=K),
                  SearchSpec(name="euc", metric="euclidean", k=K)])


@pytest.mark.parametrize("modes,want", [
    (("tiles", "tiles"), b"tiles"), (("tiles", "slice"), b"mixed"),
    (("slice", "slice"), None)])
def test_a_merge_carries_the_partials_rescore_stamp(ds, tmp_path, small,
                                                    monkeypatch, modes, want):
    """The merged artifact's stamp comes from its PARTIALS ("tiles", or
    "mixed" when they disagree), never from the merge process's own config or
    environment; and only dense cosine/dot searches are stamped."""
    from nova_bf.merge import run_merge
    from nova_bf.results import RESCORE_KEY

    out = tmp_path / "o"
    out.mkdir()
    for rank, mode in enumerate(modes):
        twopass.reset()
        run_compute(_cfg_prov(ds, out, mode), num_jobs=2, job_rank=rank)
    # The merge is told the opposite on both channels; it must not matter.
    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE",
                       "slice" if want == b"tiles" else "tiles")
    merged = run_merge(_cfg_prov(ds, out, "slice" if want else "tiles"))
    meta = {n: pq.read_schema(p).metadata or {} for n, p in merged.items()}
    assert meta["cos"].get(RESCORE_KEY) == want
    assert RESCORE_KEY not in meta["euc"], "euclidean never takes the two-pass"


def test_the_two_cutoffs_warn_independently(monkeypatch, caplog):
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "abc")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "xyz")
    with caplog.at_level("WARNING", logger="nova_bf.twopass"):
        twopass.threshold("tiles")
        twopass.threshold("slice")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("NOVA_BF_TWOPASS_TILE_THRESHOLD" in m for m in msgs)
    assert any("NOVA_BF_TWOPASS_THRESHOLD=" in m for m in msgs)


def test_the_probe_width_follows_the_fused_config_that_would_run(monkeypatch):
    assert twopass.default_tile_width("cuda:0") == twopass.FUSE_CONFIG[1]
    monkeypatch.setenv("NOVA_BF_FUSE_CONFIG", "64,256,32,8,3,8")
    assert twopass.default_tile_width("cuda:0") == 256
    monkeypatch.setitem(twopass._FUSE_KERNEL_INFO, "config", [64, 64, 32, 8, 3, 4])
    assert twopass.default_tile_width("cuda:0") == 64        # what ran wins
    assert twopass.default_tile_width("cpu") == twopass.CPU_TILE


def test_the_comparator_rejects_a_repeated_id():
    t = {"query_id": ["q0"], "hit_ids": [["a", "a"]], "hit_scores": [[1.0, 1.0]]}
    u = {"query_id": ["q0"], "hit_ids": [["a", "b"]], "hit_scores": [[1.0, 1.0]]}
    with pytest.raises(AssertionError, match="repeats"):
        assert_same_results(t, u, abs_tol=1e-6)
