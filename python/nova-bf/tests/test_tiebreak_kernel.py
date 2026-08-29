"""The CUDA kernel path, checked against the portable one.

Every test here SKIPS without a GPU, so none of it runs in a CPU-only CI — the
kernel's coverage is exactly as good as the hardware the suite is run on. Run it
on a GPU box before trusting `topk_triton`.

The portable `pack_topk` body is the oracle throughout: the two paths must
select the SAME SET of winners and pair each key with its own index. If they
ever diverge, a ground truth computed on a CUDA box differs from one computed
without — the machine-dependent answer this whole feature exists to eliminate.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import nova_bf.topk_triton as tk
from nova_bf.tiebreak import pack, pack_topk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or tk._cutfill is None,
    reason=f"needs CUDA + triton (kernel built: {tk._cutfill is not None}, "
           f"reason: {tk._UNAVAILABLE})",
)

DEV = "cuda"


def _portable(scores, ordinal, k):
    """The oracle: pack everything, then select. What runs when the gate says no."""
    return torch.topk(pack(scores, ordinal), k, dim=1, sorted=False)[1]


def _assert_matches_oracle(sc, od, k, label=""):
    """Both the SET of winners and the packed KEY must match the portable path
    exactly, for every kernel variant."""
    ref_idx = _portable(sc, od, k)
    ref_key = pack(sc, od)
    key, idx = tk.topk(sc, od, k)
    assert _same_set(idx, ref_idx), f"{label}: wrong winners"
    assert torch.equal(key, ref_key.gather(1, idx)), (
        f"{label}: key does not match its own index"
    )


def _same_set(a, b):
    return torch.equal(torch.sort(a.long(), dim=1).values, torch.sort(b.long(), dim=1).values)


def _scores(regime, n_q, n_cols, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if regime == "continuous":
        return torch.randn(n_q, n_cols, generator=g, device=DEV)
    if regime == "all-tied":
        return torch.full((n_q, n_cols), 0.25, device=DEV)
    if regime == "few-values":                       # dense exact ties
        return torch.randint(0, 8, (n_q, n_cols), generator=g, device=DEV).float()
    if regime == "inf-heavy":                        # sparse no-overlap / cell_mask
        s = torch.randn(n_q, n_cols, generator=g, device=DEV)
        return s.masked_fill_(torch.rand(n_q, n_cols, generator=g, device=DEV) < 0.7,
                              float("-inf"))
    if regime == "signed-zero":                      # euclidean self-hits
        s = torch.zeros(n_q, n_cols, device=DEV)
        s[:, ::2] = -0.0
        return s
    raise AssertionError(regime)


REGIMES = ["continuous", "few-values", "all-tied", "inf-heavy", "signed-zero"]


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("mode", ["ordinal", "id"])
def test_the_kernel_selects_what_the_portable_path_selects(regime, mode):
    """Both tie regimes, both tie-break modes. `id` uses a SHUFFLED permutation:
    if the ordinal descent were broken, positional order would still look right
    under 'ordinal' and only fail here."""
    n_q, n_cols, k = 512, 4096, 300
    sc = _scores(regime, n_q, n_cols).contiguous()
    od = (torch.arange(n_cols, dtype=torch.int64, device=DEV) if mode == "ordinal"
          else torch.randperm(n_cols, device=DEV).to(torch.int64) * 7919)
    assert tk.available(sc, od, k), "these inputs should take the kernel"
    _assert_matches_oracle(sc, od, k, f"{regime}/{mode}")


@pytest.mark.parametrize("k", [1, 2, 4095, 4096])
def test_edge_values_of_k(k):
    n_q, n_cols = 64, 4096
    sc = _scores("few-values", n_q, n_cols).contiguous()
    od = torch.arange(n_cols, dtype=torch.int64, device=DEV)
    _assert_matches_oracle(sc, od, k, f"k={k}")


@pytest.mark.parametrize("n_cols", [2, 3, 17, 1000, 1023, 1024, 4096])
def test_non_power_of_two_widths(n_cols):
    """BLOCK is rounded up to a power of two, so every lane past n_cols is
    padding — the case where a pad lane could steal a winner slot."""
    k = max(1, n_cols // 3)
    sc = _scores("few-values", 32, n_cols).contiguous()
    od = torch.randperm(n_cols, device=DEV).to(torch.int64)
    _assert_matches_oracle(sc, od, k, f"n_cols={n_cols}")


def test_a_row_that_is_entirely_negative_infinity():
    """Every candidate is -inf, so the cut sits at the bottom of the key space —
    the condition under which padding lanes could compete with real ones."""
    n_cols, k = 4096, 100
    sc = torch.full((16, n_cols), float("-inf"), device=DEV)
    od = torch.arange(n_cols, dtype=torch.int64, device=DEV)
    _assert_matches_oracle(sc, od, k, "all -inf")


def test_pack_topk_routes_through_the_kernel_and_matches(monkeypatch):
    """End to end through the real entry point: the key handed back must be the
    key OF the returned index, exactly as the portable path would build it."""
    n_q, n_cols, k = 256, 4096, 200
    sc = _scores("few-values", n_q, n_cols).contiguous()
    od = torch.randperm(n_cols, device=DEV).to(torch.int64)

    key_gpu, idx_gpu = pack_topk(sc, od, k)
    assert torch.equal(key_gpu, pack(sc, od).gather(1, idx_gpu))

    monkeypatch.setattr(tk, "available", lambda *a: False)
    key_cpu, idx_cpu = pack_topk(sc, od, k)
    assert _same_set(idx_gpu, idx_cpu), "kernel and portable path must agree"


def test_nan_scores_rank_the_same_in_both_paths():
    """NaN is dropped downstream by the `> -inf` gate, but the two paths must
    still agree on WHICH candidates survive, or the artifact would depend on
    whether the kernel ran."""
    n_cols, k = 1024, 100
    g = torch.Generator(device=DEV).manual_seed(3)
    sc = torch.randn(32, n_cols, generator=g, device=DEV)
    sc[torch.rand(32, n_cols, generator=g, device=DEV) < 0.1] = float("nan")
    sc = sc.contiguous()
    od = torch.arange(n_cols, dtype=torch.int64, device=DEV)
    _assert_matches_oracle(sc, od, k, "nan")


# --------------------------------------------------------------------------
# score matrices from the REAL scoring paths
# --------------------------------------------------------------------------
#
# The regimes above are synthetic. These build score matrices with the actual
# DenseCorpusBatch / SparseCorpusBatch / MultiVectorCorpusBatch, so the kernel
# meets the distributions those paths really produce -- in particular sparse's
# structural `-inf` for non-overlapping rows and multivector's `-inf` for
# zero-token docs, which no synthetic mask reproduces faithfully.


def _dense_scores(n_q, n_rows, dim=32, metric="cosine", seed=0):
    from nova_bf.compute import DenseCorpusBatch

    g = torch.Generator().manual_seed(seed)
    C = torch.randn(n_rows, dim, generator=g).numpy()
    Q = torch.randn(n_q, dim, generator=g).to(DEV)
    batch = DenseCorpusBatch(C)
    sl = batch.transfer(0, n_rows, DEV)
    qn = Q.norm(dim=1).clamp_min(1e-12)
    return sl.score(Q, metric, qn)


def _sparse_scores(n_q, n_rows, vocab=64, nnz=4, seed=0):
    """Sparse rows sharing no term with a query score -inf -- structurally, not
    by masking. Deliberately sparse enough that many rows miss."""
    from nova_bf.compute import SparseCorpusBatch

    rng = np.random.default_rng(seed)
    offs = np.arange(0, n_rows * nnz + 1, nnz, dtype=np.int64)
    idx = rng.integers(0, vocab, n_rows * nnz).astype(np.int64)
    val = rng.random(n_rows * nnz).astype(np.float32)
    norms = np.sqrt(np.add.reduceat(val**2, offs[:-1])).astype(np.float32)
    batch = SparseCorpusBatch(offs, idx, val, norms, list(range(vocab)), True)
    sl = batch.transfer(0, n_rows, DEV)
    Q = torch.zeros(n_q, vocab, device=DEV)
    for q in range(n_q):
        cols = rng.choice(vocab, size=nnz, replace=False)
        Q[q, cols] = torch.from_numpy(rng.random(nnz).astype(np.float32)).to(DEV)
    return sl.score(Q, "dot", Q.norm(dim=1).clamp_min(1e-12))


def _multivector_scores(n_q, n_rows, dim=16, toks=3, seed=0):
    from nova_bf.compute import MultiVectorCorpusBatch, MultiVectorQuery

    g = torch.Generator().manual_seed(seed)
    doc_offsets = np.arange(0, n_rows * toks + 1, toks, dtype=np.int64)
    flat = torch.randn(n_rows * toks, dim, generator=g).numpy()
    batch = MultiVectorCorpusBatch(doc_offsets, flat)
    sl = batch.transfer(0, n_rows, DEV)
    q_off_cpu = np.arange(0, n_q * toks + 1, toks, dtype=np.int64)
    q_off = torch.from_numpy(q_off_cpu).to(DEV)
    q_flat = torch.randn(n_q * toks, dim, generator=g).to(DEV)
    q = MultiVectorQuery(
        flat=q_flat, offsets=q_off, offsets_cpu=q_off_cpu, n_q=n_q, query_block=None
    )
    return sl.score(q, "dot")


@pytest.mark.parametrize("metric", ["cosine", "dot", "euclidean"])
def test_real_dense_score_matrices(metric):
    """euclidean matters most here: it ends in `.sqrt_().neg_()`, so a self-hit
    produces -0.0 — the value that broke the kernel until the fold was added."""
    sc = _dense_scores(256, 2048, metric=metric).contiguous()
    od = torch.randperm(2048, device=DEV).to(torch.int64)
    _assert_matches_oracle(sc, od, 200, f"real dense/{metric}")


@pytest.mark.parametrize("nnz", [1, 2, 8])
def test_real_sparse_score_matrices(nnz):
    """Sparse fills non-overlapping cells with -inf structurally, so at low nnz
    most of the matrix is -inf and the cut lands there."""
    sc = _sparse_scores(128, 2048, nnz=nnz).contiguous()
    od = torch.randperm(2048, device=DEV).to(torch.int64)
    n_inf = int((sc == float("-inf")).sum())
    assert n_inf > 0, "this fixture is meant to produce -inf cells"
    _assert_matches_oracle(sc, od, 200, f"real sparse/nnz={nnz}")


def test_real_multivector_score_matrices():
    sc = _multivector_scores(128, 1024).contiguous()
    od = torch.randperm(1024, device=DEV).to(torch.int64)
    _assert_matches_oracle(sc, od, 100, "real multivector")


# --------------------------------------------------------------------------
# fuzz
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_fuzz_against_the_portable_path(seed):
    """Random shapes, random score alphabets, random ordinals. The alphabet is
    deliberately tiny on some seeds so exact ties are dense, and includes the
    values that have bitten this code before: -0.0, +/-inf, denormals."""
    r = np.random.default_rng(seed)
    n_q = int(r.integers(1, 40))
    n_cols = int(r.integers(2, 600))
    k = int(r.integers(1, n_cols + 1))
    specials = [
        0.0, -0.0, 1.0, -1.0, np.inf, -np.inf,
        1e-45, -1e-45,                      # subnormals
        3.4e38, -3.4e38,                    # +/- FLT_MAX
        np.float32(np.nan),                 # positive NaN
        np.frombuffer(np.uint32(0xFFC00000).tobytes(), np.float32)[0],  # NEGATIVE NaN
    ]
    n_vals = int(r.integers(1, 12))
    pool = np.array(r.choice(specials + list(r.standard_normal(8)), n_vals), np.float32)
    sc = torch.from_numpy(r.choice(pool, size=(n_q, n_cols)).astype(np.float32)).to(DEV)
    od = torch.from_numpy(r.permutation(n_cols).astype(np.int64)).to(DEV)
    if bool(r.integers(0, 2)):                       # sometimes non-dense ordinals
        od = od * int(r.integers(1, 10**6))
    _assert_matches_oracle(sc.contiguous(), od, k, f"fuzz seed={seed}")


def test_ordinals_near_the_32_bit_ceiling():
    """MAX_ROWS_PER_WORKER is 2**32-1. The kernel ranks ordinals for descent 2,
    so magnitude should not matter — but the packed key it emits uses the RAW
    ordinal, and that is where a width mistake would surface."""
    from nova_bf.tiebreak import MAX_ROWS_PER_WORKER

    n_cols, k = 512, 100
    sc = torch.zeros(8, n_cols, device=DEV)          # everything ties
    od = torch.arange(n_cols, dtype=torch.int64, device=DEV)
    od = MAX_ROWS_PER_WORKER - 1 - od                # straddles 2**31
    _assert_matches_oracle(sc.contiguous(), od, k, "ordinals near 2**32")


def test_the_gate_rejects_everything_the_kernel_assumes():
    """`available()` is the contract boundary: a wrong True is a silent
    correctness bug, a wrong False only costs speed. Each of these would break
    a different assumption inside the kernel or its wrapper."""
    s = torch.randn(8, 64, device=DEV)
    o = torch.arange(64, dtype=torch.int64, device=DEV)
    assert tk.available(s, o, 8), "the happy path must still be taken"
    assert not tk.available(s.double(), o, 8), "non-float32"
    assert not tk.available(torch.randn(64, device=DEV), o, 8), "1-D scores"
    assert not tk.available(s, o.to(torch.int32), 8), "non-int64 ordinal"
    assert not tk.available(s, o.cpu(), 8), "ordinal on another device"
    assert not tk.available(s, o.unsqueeze(0).expand(8, 64), 8), "2-D (per-cell) ordinal"
    assert not tk.available(s[:, ::2], o[::2], 8), "non-contiguous scores"
    assert not tk.available(torch.randn(0, 64, device=DEV), o, 8), "n_q == 0"
    assert not tk.available(s, o, 0), "k == 0"
    assert not tk.available(s, o, 65), "k > n_cols"


def test_ordinals_at_both_ends_of_the_permitted_range():
    """The packed key puts the ordinal in the low 32 bits as `0xFFFFFFFF - ord`,
    so 0 and 0xFFFFFFFF are the exact edges of what `topk` accepts."""
    from nova_bf.tiebreak import MAX_ROWS_PER_WORKER

    n_cols, k = 256, 64
    sc = torch.zeros(8, n_cols, device=DEV).contiguous()   # all tied -> ordinal decides
    for lbl, od in (
        ("low edge", torch.arange(n_cols, dtype=torch.int64, device=DEV)),
        ("high edge", MAX_ROWS_PER_WORKER - torch.arange(n_cols, dtype=torch.int64, device=DEV)),
    ):
        assert int(od.min()) >= 0 and int(od.max()) <= MAX_ROWS_PER_WORKER
        _assert_matches_oracle(sc, od, k, lbl)


# --------------------------------------------------------------------------
# does the REAL pipeline actually reach the kernel?
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_run_compute_actually_uses_the_kernel(tmp_path, tiebreak, monkeypatch):
    """`available()` guards a PERFORMANCE path, so a gate that wrongly rejects
    is invisible — the run just silently gets 4x slower. This asserts the real
    `run_compute` reaches the kernel in BOTH tie-break modes, and that the
    ordinals it hands over satisfy every condition the gate checks.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    import nova_bf.topk_triton as tkm
    from nova_bf.compute import run_compute
    from nova_bf.config import (
        BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig,
        QueriesConfig, SearchSpec,
    )

    seen = {"calls": 0, "declined": 0}
    real_avail, real_topk = tkm.available, tkm.topk

    def spy_available(scores, ordinal, k):
        ok = real_avail(scores, ordinal, k)
        if not ok:
            seen["declined"] += 1
            # surface WHY, so a regression names itself instead of just being slow
            seen["why"] = dict(
                cuda=scores.is_cuda, dtype=str(scores.dtype),
                contig=scores.is_contiguous(), sdim=scores.ndim,
                odim=ordinal.ndim, odtype=str(ordinal.dtype),
                odev=str(ordinal.device), sdev=str(scores.device),
                n=tuple(scores.shape), k=k,
            )
        return ok

    def spy_topk(*a, **kw):
        seen["calls"] += 1
        return real_topk(*a, **kw)

    monkeypatch.setattr(tkm, "available", spy_available)
    monkeypatch.setattr(tkm, "topk", spy_topk)

    cdir = tmp_path / "c"
    cdir.mkdir()
    rng = np.random.default_rng(0)
    for f in range(2):
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array(
                    rng.standard_normal((600, 8)).astype(np.float32).tolist(),
                    pa.list_(pa.float32()),
                ),
                "sid": pa.array([f"f{f}_r{i}" for i in range(600)]),
            }),
            str(cdir / f"f{f}.parquet"),
        )
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(
                rng.standard_normal((32, 8)).astype(np.float32).tolist(),
                pa.list_(pa.float32()),
            ),
            "qid": pa.array([f"q{i}" for i in range(32)]),
        }),
        str(tmp_path / "q.parquet"),
    )
    out = tmp_path / f"o-{tiebreak}"
    out.mkdir()
    run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(out)),
        # batch wider than k so the pre-top-K (and therefore the kernel) fires
        params=ParamsConfig(io_workers=1, tiebreak=tiebreak, dense_batch_size=512),
        searches=[SearchSpec(name="t", k=64, metric="dot")],
    ))
    assert seen["calls"] > 0, (
        f"tiebreak={tiebreak}: the kernel was never reached. "
        f"declined={seen['declined']} first_reason={seen.get('why')}"
    )
    assert seen["declined"] == 0, (
        f"tiebreak={tiebreak}: gate declined {seen['declined']} call(s): {seen.get('why')}"
    )
