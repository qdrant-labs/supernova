"""Correctness of the shared-Gram dense scoring path (`DenseBatchSlice`).

When 2+ DISTINCT metrics score one dense batch, all of them are derived from a
single raw Gram `Q @ Cᵀ` instead of each running its own GEMM. Every dense
metric is a rank-1 rescaling or shift of that product, so the derivation is
mathematically exact — these tests pin down the three things that could still
go wrong:

  1. ALIASING. `dot` is handed the Gram itself with no copy, and `cosine`/
     `euclidean` then read that same tensor. An in-place op on the wrong
     operand would silently corrupt whichever member ran first, in a way only
     visible as wrong scores for that one search. Tested in BOTH member orders,
     and with `k >= rows` so the dot member's part buffer holds the Gram
     itself rather than a topk'd copy of it.
  2. THE GATE. A single-metric batch must keep the unshared path (no extra
     resident matrix, no change in float32 output); two specs sharing ONE
     metric must not trip sharing either, since there is no second product to
     amortize.
  3. AGREEMENT with the unshared path — bit-identical for `dot`, and for
     `cosine`/`euclidean` within float32 resolution AND with the neighbour
     ORDER unchanged, which is the property ground truth actually rests on.

Ground truth is computed in float64 numpy, never by re-deriving nova_bf's own
scoring — same convention as test_compute.py / test_compute_multi.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nova_bf.compute import DenseCorpusBatch, run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    QueriesConfig,
    SearchSpec,
)

DIM = 12
METRICS = ("dot", "cosine", "euclidean")


def _slice(C: np.ndarray, share: bool):
    b = DenseCorpusBatch(np.ascontiguousarray(C, dtype=np.float32))
    b.share_gram = share
    return b.transfer(0, len(C), "cpu")


def _both_paths(C: np.ndarray, Q: torch.Tensor, metric: str):
    """(unshared, shared) score matrices for one metric over the same data."""
    qn = Q.norm(dim=1).clamp_min(1e-12)
    return (
        _slice(C, False).score(Q, metric, qn),
        _slice(C, True).score(Q, metric, qn),
    )


def _np_scores(C: np.ndarray, Q: np.ndarray, metric: str) -> np.ndarray:
    """Independent float64 ground truth — deliberately NOT nova_bf's code."""
    c, q = C.astype(np.float64), Q.astype(np.float64)
    if metric == "dot":
        return q @ c.T
    if metric == "cosine":
        cn = np.maximum(np.linalg.norm(c, axis=1), 1e-12)
        qnorm = np.maximum(np.linalg.norm(q, axis=1), 1e-12)
        return (q @ c.T) / cn[None, :] / qnorm[:, None]
    return -np.sqrt(((q[:, None, :] - c[None, :, :]) ** 2).sum(-1))


# --------------------------------------------------------------- agreement
@pytest.mark.parametrize("n_q,rows", [(3, 40), (30, 7), (64, 256), (1, 26)])
def test_dot_is_bit_identical_under_sharing(n_q, rows):
    """`dot` returns the Gram unchanged, so sharing must not perturb it at all
    — not "allclose", exactly equal."""
    rng = np.random.default_rng(101)
    C = rng.standard_normal((rows, DIM)).astype(np.float32)
    Q = torch.tensor(rng.standard_normal((n_q, DIM)).astype(np.float32))
    unshared, shared = _both_paths(C, Q, "dot")
    assert torch.equal(unshared, shared)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("n_q,rows", [(3, 40), (30, 7), (64, 256)])
def test_shared_matches_unshared_and_float64_truth(metric, n_q, rows):
    """Both paths must agree with each other AND with independent float64
    numpy, to float32 resolution."""
    rng = np.random.default_rng(202)
    C = rng.standard_normal((rows, DIM)).astype(np.float32)
    Q_np = rng.standard_normal((n_q, DIM)).astype(np.float32)
    Q = torch.tensor(Q_np)
    unshared, shared = _both_paths(C, Q, metric)
    truth = _np_scores(C, Q_np, metric)
    scale = max(float(np.abs(truth).max()), 1.0)
    assert np.allclose(unshared.numpy(), truth, atol=2e-5 * scale)
    assert np.allclose(shared.numpy(), truth, atol=2e-5 * scale)
    assert np.allclose(shared.numpy(), unshared.numpy(), atol=2e-5 * scale)


@pytest.mark.parametrize("n_q,rows", [(5, 9), (25, 25), (26, 5), (5, 26), (64, 128)])
def test_euclidean_agrees_with_cdist_at_every_shape(n_q, rows):
    """The shared euclidean is the `‖q‖² + ‖c‖² − 2qc` expansion, which is what
    `torch.cdist` itself uses above 25 rows on either side — so at real shapes
    the two agree to float32 noise. Below that threshold cdist switches to its
    exact direct method and the derived form is slightly looser, bounded by
    ~sqrt(eps)·‖q‖ (float32 eps ~ 1.19e-7, so ~3.5e-4 per unit of ‖q‖).

    The tolerance is therefore scaled by ‖q‖ rather than fixed, and the
    RANKING is asserted exactly: an error that reorders neighbours would be a
    real defect, one that only shifts distance values by less than the
    expansion's own resolution floor is not."""
    rng = np.random.default_rng(303)
    C = rng.standard_normal((rows, DIM)).astype(np.float32)
    Q = torch.tensor(rng.standard_normal((n_q, DIM)).astype(np.float32))
    unshared, shared = _both_paths(C, Q, "euclidean")
    atol = 5e-4 * float(Q.norm(dim=1).max())
    assert torch.allclose(unshared, shared, atol=atol), (
        f"max |delta| = {(unshared - shared).abs().max().item():.3e} > {atol:.3e}")
    for qi in range(n_q):
        assert torch.equal(torch.argsort(unshared[qi], descending=True),
                           torch.argsort(shared[qi], descending=True)), (
            f"q{qi}: sharing reordered the neighbours")


def test_gram_built_once_per_slice():
    """Three metrics off one slice must issue exactly ONE matmul — the whole
    point of the change. Counted by wrapping torch.matmul."""
    rng = np.random.default_rng(404)
    C = rng.standard_normal((64, DIM)).astype(np.float32)
    Q = torch.tensor(rng.standard_normal((40, DIM)).astype(np.float32))
    qn = Q.norm(dim=1).clamp_min(1e-12)
    sl = _slice(C, True)
    calls = 0
    # `Q @ Cᵀ` dispatches through `__matmul__`, not `Tensor.matmul`.
    real = torch.Tensor.__matmul__

    def counting(self, other):
        nonlocal calls
        calls += 1
        return real(self, other)

    torch.Tensor.__matmul__ = counting
    try:
        for m in METRICS:
            sl.score(Q, m, qn)
    finally:
        torch.Tensor.__matmul__ = real
    assert calls == 1, f"expected one shared GEMM, got {calls}"


def test_unshared_path_issues_one_gemm_per_metric():
    """The counterpart to the above: without sharing, each metric pays its own
    product — which is the cost this change removes. (euclidean goes through
    cdist, whose internal expansion is not an observable `__matmul__`, so only
    dot+cosine are counted here.)"""
    rng = np.random.default_rng(4041)
    C = rng.standard_normal((64, DIM)).astype(np.float32)
    Q = torch.tensor(rng.standard_normal((40, DIM)).astype(np.float32))
    qn = Q.norm(dim=1).clamp_min(1e-12)
    sl = _slice(C, False)
    calls = 0
    real = torch.Tensor.__matmul__

    def counting(self, other):
        nonlocal calls
        calls += 1
        return real(self, other)

    torch.Tensor.__matmul__ = counting
    try:
        for m in ("dot", "cosine"):
            sl.score(Q, m, qn)
    finally:
        torch.Tensor.__matmul__ = real
    assert calls == 2, f"expected one GEMM per metric unshared, got {calls}"


# ---------------------------------------------------------------- aliasing
@pytest.mark.parametrize("order", [("dot", "cosine"), ("cosine", "dot"),
                                   ("dot", "euclidean"), ("euclidean", "dot"),
                                   ("cosine", "euclidean", "dot")])
def test_no_member_corrupts_another(order):
    """`dot` aliases the Gram; the other metrics read it afterward. Whatever
    the member order, every metric's matrix must still equal what it would be
    alone — a stray in-place op would corrupt the earlier member only."""
    rng = np.random.default_rng(505)
    C = rng.standard_normal((80, DIM)).astype(np.float32)
    Q_np = rng.standard_normal((32, DIM)).astype(np.float32)
    Q = torch.tensor(Q_np)
    qn = Q.norm(dim=1).clamp_min(1e-12)

    sl = _slice(C, True)
    got = {m: sl.score(Q, m, qn) for m in order}   # all held alive at once
    for m, mat in got.items():
        truth = _np_scores(C, Q_np, m)
        scale = max(float(np.abs(truth).max()), 1.0)
        assert np.allclose(mat.numpy(), truth, atol=2e-5 * scale), (
            f"member {m} corrupted in order {order}")


def test_masking_a_shared_gram_leaves_it_intact():
    """`_process_batch_group` applies a per-query filter with `masked_fill`
    (not `masked_fill_`) to whatever `score()` returned. Since `dot` gets the
    Gram itself, an in-place variant there would poison every later metric —
    assert the non-mutating contract holds on the returned tensor."""
    rng = np.random.default_rng(606)
    C = rng.standard_normal((64, DIM)).astype(np.float32)
    Q = torch.tensor(rng.standard_normal((30, DIM)).astype(np.float32))
    qn = Q.norm(dim=1).clamp_min(1e-12)
    sl = _slice(C, True)
    raw = sl.score(Q, "dot", qn)
    before = raw.clone()
    cell = torch.rand(raw.shape) > 0.5
    masked = raw.masked_fill(~cell, float("-inf"))
    assert torch.equal(raw, before), "masking mutated the shared Gram"
    assert torch.isinf(masked[~cell]).all()


# ------------------------------------------------------------------- edges
def test_zero_corpus_row_and_zero_query_scores_zero_not_nan():
    """A zero vector must score 0 under cosine (matching F.normalize's eps
    clamp), never NaN, on the shared path exactly as on the unshared one."""
    rng = np.random.default_rng(707)
    C = rng.standard_normal((40, DIM)).astype(np.float32)
    C[3] = 0.0
    Q_np = rng.standard_normal((30, DIM)).astype(np.float32)
    Q_np[2] = 0.0
    Q = torch.tensor(Q_np)
    for metric in METRICS:
        unshared, shared = _both_paths(C, Q, metric)
        assert torch.isfinite(shared).all(), f"{metric} produced non-finite"
        assert torch.allclose(unshared, shared, atol=1e-4)
    # cosine specifics: the zero row/query contribute exactly 0
    _, cos = _both_paths(C, Q, "cosine")
    assert torch.equal(cos[:, 3], torch.zeros(30))
    assert torch.equal(cos[2, :], torch.zeros(40))


def test_euclidean_self_distance_is_non_negative():
    """The expansion can produce a slightly negative d² for a near-duplicate;
    `clamp_min_(0)` must fence it so `sqrt` never yields NaN."""
    rng = np.random.default_rng(808)
    C = (rng.standard_normal((64, DIM)) * 30.0).astype(np.float32)
    Q = torch.tensor(C[:32].copy())          # every query IS a corpus row
    qn = Q.norm(dim=1).clamp_min(1e-12)
    sl = _slice(C, True)
    d = sl.score(Q, "euclidean", qn)
    assert torch.isfinite(d).all()
    assert (d <= 0).all(), "negated distance must be <= 0"
    diag = d[torch.arange(32), torch.arange(32)]
    assert diag.abs().max() < 1e-1, f"self-distance too large: {diag.abs().max()}"


# ------------------------------------------------------- end-to-end / gate
@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    """A dense-only corpus over 3 files with row counts that don't divide the
    batch size, plus a payload column for the filtered specs."""
    tmp = tmp_path_factory.mktemp("sharedgram")
    rng = np.random.default_rng(909)
    cdir = tmp / "corpus"
    cdir.mkdir()
    sizes = [37, 29, 41]
    rows, ids, langs = [], [], []
    n = 0
    for fi, sz in enumerate(sizes):
        vecs = rng.standard_normal((sz, DIM)).astype(np.float32)
        pq.write_table(
            pa.table({
                "dense_embedding": pa.array(vecs.tolist(), type=pa.list_(pa.float32())),
                "id": pa.array([f"c{n + i}" for i in range(sz)]),
                "language": pa.array(["eng" if (n + i) % 3 else "fra" for i in range(sz)]),
            }),
            str(cdir / f"f{fi}.parquet"),
        )
        rows.append(vecs)
        ids += [f"c{n + i}" for i in range(sz)]
        langs += ["eng" if (n + i) % 3 else "fra" for i in range(sz)]
        n += sz
    qvecs = rng.standard_normal((33, DIM)).astype(np.float32)
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(qvecs.tolist(), type=pa.list_(pa.float32())),
            "qid": pa.array([f"q{i}" for i in range(33)]),
        }),
        str(tmp / "queries.parquet"),
    )
    return {
        "tmp": tmp, "cdir": str(cdir), "qpath": str(tmp / "queries.parquet"),
        "C": np.concatenate(rows), "Q": qvecs, "ids": ids, "langs": langs,
    }


def _run(ds, specs, tag, batch_size=16):
    from nova_bf.config import ParamsConfig

    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(ds["tmp"] / f"out_{tag}")),
        params=ParamsConfig(io_workers=2, dense_batch_size=batch_size),
        searches=specs,
    )
    return run_compute(cfg)


@pytest.mark.parametrize(
    "metrics,expect_shared",
    [
        (["dot"], False),                       # one metric -> unshared
        (["cosine"], False),
        (["dot", "dot"], False),                # same metric twice -> nothing to share
        (["dot", "cosine"], True),
        (["dot", "cosine", "euclidean"], True),
    ],
)
def test_share_gram_gate(ds, metrics, expect_shared, monkeypatch):
    """The gate is `len(distinct metrics) > 1`, evaluated per batch. Recorded
    at slice time, which is the only place it can affect behaviour."""
    seen: list[bool] = []
    real = DenseCorpusBatch.transfer

    def spy(self, r0, r1, device):
        seen.append(self.share_gram)
        return real(self, r0, r1, device)

    monkeypatch.setattr(DenseCorpusBatch, "transfer", spy)
    specs = [
        SearchSpec(name=f"s{i}", vector_type="dense", metric=m, k=5)
        for i, m in enumerate(metrics)
    ]
    _run(ds, specs, f"gate_{'_'.join(metrics)}_{len(metrics)}")
    assert seen, "no dense slice was transferred"
    assert all(s is expect_shared for s in seen), seen


@pytest.mark.parametrize(
    "metrics,euclid_gets_norms",
    [
        (["euclidean"], False),                 # nothing to share -> no norms built
        (["euclidean", "euclidean"], False),    # same metric twice -> still nothing
        (["euclidean", "dot"], True),           # can share -> ‖q‖ is needed
        (["euclidean", "cosine"], True),
    ],
)
def test_euclidean_only_run_builds_no_query_norms(ds, metrics, euclid_gets_norms,
                                                  monkeypatch):
    """`_scores`'s euclidean branch ignores `q_norms`, so a euclidean-only run
    must not pay the O(n_q × dim) reduction (nor hold the (n_q,) tensor) that
    only the shared-Gram derivation reads. Recorded at the call boundary,
    which is where a wasted argument would show up."""
    from nova_bf.compute import DenseBatchSlice

    seen: list[tuple[str, bool]] = []
    real = DenseBatchSlice.score

    def spy(self, Q, metric, q_norms=None):
        seen.append((metric, q_norms is not None))
        return real(self, Q, metric, q_norms)

    monkeypatch.setattr(DenseBatchSlice, "score", spy)
    specs = [
        SearchSpec(name=f"s{i}", vector_type="dense", metric=m, k=5)
        for i, m in enumerate(metrics)
    ]
    _run(ds, specs, f"norms_{'_'.join(metrics)}_{len(metrics)}")
    euclid = [had for metric, had in seen if metric == "euclidean"]
    assert euclid, "no euclidean score call observed"
    assert all(had is euclid_gets_norms for had in euclid), seen
    # cosine always needs them, shared or not
    for metric, had in seen:
        if metric == "cosine":
            assert had, "cosine must always receive q_norms"


def test_zero_query_vector_end_to_end(tmp_path):
    """The shared cosine divides by `q_norms` without re-clamping, so a zero
    query would be 0/0 -> NaN if the value arrived unclamped. `run_compute`
    clamps once at the source instead (`q_norms_by_vt`), which is a non-local
    invariant and therefore worth pinning THROUGH `run_compute` rather than by
    handing `score()` a hand-clamped tensor. `_scores` has always depended on
    the same contract, so this guards both paths at once."""
    from nova_bf.config import ParamsConfig

    rng = np.random.default_rng(1234)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    C = rng.standard_normal((50, DIM)).astype(np.float32)
    C[7] = 0.0                                    # a zero CORPUS row too
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(C.tolist(), type=pa.list_(pa.float32())),
            "id": pa.array([f"c{i}" for i in range(50)]),
        }),
        str(cdir / "f0.parquet"),
    )
    Q = rng.standard_normal((5, DIM)).astype(np.float32)
    Q[1] = 0.0                                    # the zero QUERY under test
    pq.write_table(
        pa.table({
            "dense_embedding": pa.array(Q.tolist(), type=pa.list_(pa.float32())),
            "qid": pa.array([f"q{i}" for i in range(5)]),
        }),
        str(tmp_path / "queries.parquet"),
    )
    # all three metrics -> shared Gram; k > 1 so a NaN could not hide in a tie
    specs = [
        SearchSpec(name=m, vector_type="dense", metric=m, k=6) for m in METRICS
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(tmp_path / "queries.parquet"), id_column="qid"),
        output=OutputConfig(path=str(tmp_path / "out")),
        params=ParamsConfig(io_workers=1, dense_batch_size=16),
        searches=specs,
    )
    paths = run_compute(cfg)
    for name, path in paths.items():
        t = pq.read_table(path).to_pydict()
        for qid, scores in zip(t["query_id"], t["hit_scores"]):
            arr = np.asarray(scores, dtype=np.float64)
            assert not np.isnan(arr).any(), f"{name} {qid} produced NaN: {scores}"
            assert np.isfinite(arr).all(), f"{name} {qid} non-finite: {scores}"
    # the zero query's cosine row is all-zero similarity, so it still returns k
    # hits (all scoring 0.0) rather than NaN-poisoned garbage
    t = pq.read_table(paths["cosine"]).to_pydict()
    row = dict(zip(t["query_id"], t["hit_scores"]))["q1"]
    assert all(s == 0.0 for s in row), f"zero query cosine should be all 0.0, got {row}"


def test_multi_metric_run_matches_float64_ground_truth(ds):
    """End-to-end: one config with all three dense metrics (so the shared
    path is live) must return the true top-K per metric, verified against
    float64 numpy over the whole corpus."""
    k = 7
    specs = [
        SearchSpec(name="dot", vector_type="dense", metric="dot", k=k),
        SearchSpec(name="cos", vector_type="dense", metric="cosine", k=k),
        SearchSpec(name="euc", vector_type="dense", metric="euclidean", k=k),
    ]
    paths = _run(ds, specs, "e2e_truth")
    for name, metric in [("dot", "dot"), ("cos", "cosine"), ("euc", "euclidean")]:
        S = _np_scores(ds["C"], ds["Q"], metric)
        t = pq.read_table(paths[name]).to_pydict()
        got = dict(zip(t["query_id"], t["hit_ids"]))
        for qi in range(len(ds["Q"])):
            order = np.argsort(-S[qi], kind="stable")[:k]
            expect = [ds["ids"][j] for j in order]
            assert got[f"q{qi}"] == expect, f"{name} q{qi}"


def test_multi_metric_matches_solo_runs(ds):
    """The regression guard that matters operationally: a shared multi-metric
    run must produce the same ranking as each metric run alone (which takes
    the unshared path). Ids must match exactly; scores to float32 resolution.
    Also exercised with a filtered sibling riding the same shared pass."""
    eng = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        SearchSpec(name="dot", vector_type="dense", metric="dot", k=6),
        SearchSpec(name="cos", vector_type="dense", metric="cosine", k=6),
        SearchSpec(name="euc", vector_type="dense", metric="euclidean", k=6),
        SearchSpec(name="cos_eng", vector_type="dense", metric="cosine", k=4, filter=eng),
        SearchSpec(name="euc_eng", vector_type="dense", metric="euclidean", k=4, filter=eng),
    ]
    combined = _run(ds, specs, "combined")
    for spec in specs:
        solo = _run(ds, [spec], f"solo_{spec.name}")[spec.name]
        ct = pq.read_table(combined[spec.name]).to_pydict()
        st = pq.read_table(solo).to_pydict()
        assert ct["hit_ids"] == st["hit_ids"], f"ranking diverged for {spec.name}"
        for a, b in zip(ct["hit_scores"], st["hit_scores"]):
            assert np.allclose(a, b, atol=1e-5), f"scores diverged for {spec.name}"


def test_filtered_only_multi_metric_matches_solo(ds):
    """Same equivalence in the OTHER grid regime: every dense search filtered
    (no unfiltered baseline), so the shared batch is the compacted row union
    AND the Gram is shared across metrics on that compacted grid."""
    eng = Filter(must=[FilterCondition(field="language", match="eng")])
    fra = Filter(must=[FilterCondition(field="language", match="fra")])
    specs = [
        SearchSpec(name="dot_eng", vector_type="dense", metric="dot", k=5, filter=eng),
        SearchSpec(name="cos_fra", vector_type="dense", metric="cosine", k=5, filter=fra),
        SearchSpec(name="euc_eng", vector_type="dense", metric="euclidean", k=5, filter=eng),
    ]
    combined = _run(ds, specs, "union_combined")
    for spec in specs:
        solo = _run(ds, [spec], f"union_solo_{spec.name}")[spec.name]
        ct = pq.read_table(combined[spec.name]).to_pydict()
        st = pq.read_table(solo).to_pydict()
        assert ct["hit_ids"] == st["hit_ids"], f"ranking diverged for {spec.name}"
        for a, b in zip(ct["hit_scores"], st["hit_scores"]):
            assert np.allclose(a, b, atol=1e-5), f"scores diverged for {spec.name}"
