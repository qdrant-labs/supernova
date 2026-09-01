"""Two optimizations of the GPU score path, and the invariants each must hold.

  1. A GPU-eligible per-query filter builds its mask at ITS OWN query height,
     not the queries file's (`_narrow_gpu_leaf_state`).
  2. cosine's per-query-norm divide is fused into the packer's read rather
     than run as a separate pass over the per-slice score matrix. The scores written out must be the
     same; only tie behaviour may differ (see `topk_triton._cutfill`), so that is
     tested on data with no ties.

Each is checked for the ANSWER first and the mechanism second - a fast path that
quietly stops engaging is a silent perf regression, and one that engages when it
should not is a silent correctness bug.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import compute as compute_mod
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig, QueriesConfig,
    SearchSpec,
)

DIM = 8


def _build(tmp_path, n_corpus=60, n_filt=6, n_plain=14, seed=0):
    """A corpus plus a queries file whose rows split between an unfiltered
    search and one with a GPU-eligible (`match_from_query`) filter."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "c"
    cdir.mkdir(exist_ok=True)
    cvecs = rng.normal(size=(n_corpus, DIM)).astype(np.float32)
    pq.write_table(pa.table({
        "dense_embedding": pa.array(list(cvecs.tolist()), pa.list_(pa.float32())),
        "sid": pa.array([f"c{i}" for i in range(n_corpus)]),
        "lang": pa.array([["eng", "fra", "deu"][i % 3] for i in range(n_corpus)]),
        "score": pa.array(rng.integers(0, 100, n_corpus).tolist()),
    }), str(cdir / "f0.parquet"))

    n = n_filt + n_plain
    qvecs = rng.normal(size=(n, DIM)).astype(np.float32)
    pq.write_table(pa.table({
        "dense_embedding": pa.array(list(qvecs.tolist()), pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(n)]),
        "qset": pa.array(["filt"] * n_filt + ["plain"] * n_plain),
        "want": pa.array([["eng", "fra"][i % 2] for i in range(n)]),
    }), str(tmp_path / "q.parquet"))
    return cdir, tmp_path / "q.parquet"


def _cfg(tmp_path, cdir, qpath, out_name, *, metric="cosine", batch=8, **params):
    out = tmp_path / out_name
    out.mkdir(exist_ok=True)
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qpath), id_column="qid",
                              payload_fields=["qset", "want"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=batch, **params),
        searches=[
            SearchSpec(name="plain", k=5, metric=metric,
                       rows={"column": "qset", "isin": ["plain"]}),
            SearchSpec(name="filt", k=5, metric=metric,
                       rows={"column": "qset", "isin": ["filt"]},
                       filter={"must": [{"field": "lang",
                                         "match_from_query": "want"}]}),
        ],
    )


def _read(paths):
    out = {}
    for name, p in paths.items():
        t = pq.read_table(p).to_pydict()
        out[name] = (t["query_id"], t["hit_ids"], t["hit_scores"])
    return out


# ---------------------------------------------------------------------------
# 1. GPU-eligible filter mask height
# ---------------------------------------------------------------------------

def test_gpu_filter_mask_is_built_at_the_filters_own_height(tmp_path, monkeypatch):
    """The filtered search owns 6 of 20 query rows, so its per-slice mask must
    be 6 tall. At the production shape this is 5,000 rather than 110,000 - every
    leaf comparison and combining `and` runs over the mask, so the height is the
    cost."""
    cdir, qpath = _build(tmp_path)
    seen = []
    real = compute_mod._gpu_evaluate
    monkeypatch.setattr(compute_mod, "_gpu_evaluate",
                        lambda *a, **k: seen.append(real(*a, **k).shape) or real(*a, **k))
    run_compute(_cfg(tmp_path, cdir, qpath, "o1"))
    assert seen, "the GPU-eligible filter never ran"
    assert {h[0] for h in seen} == {6}, f"expected height 6, got {sorted({h[0] for h in seen})}"


def test_narrowing_does_not_change_the_answer(tmp_path, monkeypatch):
    """Narrowing renumbers the mask's rows, and a mask read at the wrong height
    masks the WRONG QUERIES without raising. Pin the answer against a run whose
    filter owns every query row, where narrowing is a no-op."""
    cdir, qpath = _build(tmp_path)
    a = _read(run_compute(_cfg(tmp_path, cdir, qpath, "oa")))
    # same corpus/queries, but the filtered spec has no `rows` subset, so its
    # height is the whole file either way
    out = tmp_path / "ob"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(qpath), id_column="qid",
                              payload_fields=["qset", "want"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=8),
        searches=[SearchSpec(name="filt", k=5, metric="cosine",
                             filter={"must": [{"field": "lang",
                                               "match_from_query": "want"}]})],
    )
    b = _read(run_compute(cfg))
    # the 6 "filt" queries must get the same hits in both runs
    qa, ia, sa = a["filt"]
    qb, ib, sb = b["filt"]
    pos = {q: j for j, q in enumerate(qb)}
    for j, q in enumerate(qa):
        assert ia[j] == ib[pos[q]], f"{q}: narrowed run disagrees"
        np.testing.assert_allclose(sa[j], sb[pos[q]], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. cosine's per-query divide, fused into the packer
# ---------------------------------------------------------------------------
# The divide used to be a standalone pass over the whole (n_q, batch) score
# matrix. It now happens inside `pack`/`pack_topk` as the scores are read, which
# is free on a kernel that already reads every element. The invariant that makes
# it safe is that the divide happens BEFORE the key is built, so the packed key
# still encodes the score that is reported - if it were applied afterwards,
# candidates equal once divided would be ordered by their raw values and the
# "lowest ordinal wins among equal scores" rule would no longer hold.


def test_fused_divide_matches_dividing_first():
    """`pack(s, o, n)` must equal `pack(s / n, o)` bit-for-bit.

    Weak on purpose, and worth saying so: on the portable path `pack` literally
    computes `scores / scale[:, None]`, so this is close to comparing code with
    itself. It guards the plumbing (that the scale is applied at all, to the
    right axis, before the key) and nothing about the KERNEL. The test that can
    actually fail is the CUDA one below, which forces the portable path with
    NOVA_BF_NO_TOPK_KERNEL and demands the two divides agree bit-for-bit."""
    import torch
    from nova_bf.tiebreak import pack, pack_topk
    rng = np.random.default_rng(4)
    for _ in range(20):
        sc = torch.tensor(rng.normal(size=(7, 33)), dtype=torch.float32)
        nrm = torch.tensor(rng.uniform(0.3, 30.0, size=7), dtype=torch.float32)
        o = torch.arange(33, dtype=torch.int64)
        assert torch.equal(pack(sc, o, nrm), pack(sc / nrm[:, None], o))
        ka, _ = pack_topk(sc, o, 5, nrm)
        kb, _ = pack_topk(sc / nrm[:, None], o, 5)
        assert torch.equal(ka.sort(1).values, kb.sort(1).values)


def test_fused_divide_preserves_the_ordinal_tie_rule():
    """The case the fusion exists to protect. Two candidates whose RAW scores
    differ by one ULP but which collide to the same f32 once divided: the
    reported scores are equal, so the LOWER ORDINAL must win. Selecting on the
    raw value would keep the other one."""
    import torch
    from nova_bf.tiebreak import pack, unpack_score
    n = np.float32(21.973186)
    rA, rB = np.float32(0.9357219), np.float32(0.9357218)   # A > B
    assert rA != rB and np.float32(rA / n) == np.float32(rB / n)
    sc = torch.tensor([[float(rA), float(rB)]], dtype=torch.float32)
    ords = torch.tensor([500, 200], dtype=torch.int64)      # B has the LOWER
    keys = pack(sc, ords, torch.tensor([float(n)], dtype=torch.float32))[0]
    winner = int(torch.argmax(keys))
    assert winner == 1, "expected the lower ordinal (B) to win the tie"
    a, b = unpack_score(keys.view(1, 2))[0]
    assert float(a) == float(b), "both must report the same divided score"


def test_no_leftover_divide_on_the_output(tmp_path):
    """The scores written must be the cosine values, with the divide applied
    exactly once - not zero times (raw) and not twice."""
    cdir, qpath = _build(tmp_path, seed=31)
    got = _read(run_compute(_cfg(tmp_path, cdir, qpath, "fz")))
    for name in got:
        for row in got[name][2]:
            for v in row:
                assert -1.0001 <= v <= 1.0001, f"{name}: {v} is not a cosine"


# ---------------------------------------------------------------------------
# Configurations an adversarial review found untested
# ---------------------------------------------------------------------------
# Every other fixture in-tree gives each filter exactly one spec whose rows ARE
# the filter's union, so the mask's local row numbering coincides with the
# file's and `spec_qrows` is an identity slice. That coincidence hides the whole
# bug class the narrowing introduces: indexing the narrowed mask by file rows,
# or swapping `spec_qsel` with `spec_qrows`, is invisible. These build the cases
# where the two genuinely differ.

def _oracle_topk(qvecs, cvecs, keep, k):
    """Independent f64 cosine top-k for one query: `keep` is a boolean over
    corpus rows. Returns the kept corpus indices best-first."""
    q = qvecs.astype(np.float64)
    c = cvecs.astype(np.float64)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    s = (q @ cn.T) / np.linalg.norm(q)
    s = np.where(keep, s, -np.inf)
    order = np.argsort(-s, kind="stable")
    return [int(i) for i in order[:k] if np.isfinite(s[i])]


def test_two_specs_share_one_filter_with_interleaved_row_subsets(tmp_path):
    """Two specs sharing ONE filter but owning alternating query rows. The
    filter's union is every row, so each spec's `spec_qrows` is a real GATHER
    into the mask rather than a slice -- the case where mask-local and
    file-local numbering diverge."""
    rng = np.random.default_rng(11)
    n_c, n_q, k = 40, 12, 4      # n_q must match the 12-entry "half" split below
    cvecs = rng.normal(size=(n_c, DIM)).astype(np.float32)
    langs = [["eng", "fra", "deu"][i % 3] for i in range(n_c)]
    cdir = tmp_path / "c"
    cdir.mkdir()
    pq.write_table(pa.table({
        "dense_embedding": pa.array(cvecs.tolist(), pa.list_(pa.float32())),
        "sid": pa.array([f"c{i}" for i in range(n_c)]),
        "lang": pa.array(langs),
    }), str(cdir / "f0.parquet"))

    qvecs = rng.normal(size=(n_q, DIM)).astype(np.float32)
    want = [["eng", "fra"][i % 2] for i in range(n_q)]
    pq.write_table(pa.table({
        "dense_embedding": pa.array(qvecs.tolist(), pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(n_q)]),
        # Three-way split, chosen so that `spec_qsel` and `spec_qrows` are
        # DIFFERENT index vectors -- the only configuration in which mixing
        # them up is observable:
        #   "p"  rows 0-3   an unfiltered spec, present only to make the
        #                   vector_type's row union WIDER than the filter's
        #   "a"  rows 4,6,8   |  both share one filter, so that filter's union
        #   "b"  rows 5,7,9   |  is {4..9} -- a strict subset of the file, so
        #                        it does not collapse back to "all rows"
        #   rows 10,11 belong to nothing.
        # spec_qsel[A] indexes {4,6,8} in the vt union {0..9}   -> [4,6,8]
        # spec_qrows[A] indexes {4,6,8} in the filter union {4..9} -> [0,2,4]
        "half": pa.array(["p", "p", "p", "p", "a", "b", "a", "b", "a", "b",
                          "z", "z"]),
        "want": pa.array(want),
    }), str(tmp_path / "q.parquet"))

    out = tmp_path / "o"
    out.mkdir()
    filt = {"must": [{"field": "lang", "match_from_query": "want"}]}
    res = run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid",
                              payload_fields=["half", "want"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=8),
        searches=[
            # widens the dense vector_type's row union past the filter's
            SearchSpec(name="P", k=k, metric="cosine",
                       rows={"column": "half", "isin": ["p"]}),
            SearchSpec(name="A", k=k, metric="cosine", filter=filt,
                       rows={"column": "half", "isin": ["a"]}),
            SearchSpec(name="B", k=k, metric="cosine", filter=filt,
                       rows={"column": "half", "isin": ["b"]}),
        ],
    ))
    for name, want_rows in (("A", {4, 6, 8}), ("B", {5, 7, 9})):
        t = pq.read_table(res[name]).to_pydict()
        assert {int(q[1:]) for q in t["query_id"]} == want_rows
        for row, qid in enumerate(t["query_id"]):
            qi = int(qid[1:])
            keep = np.array([lg == want[qi] for lg in langs])
            expect = [f"c{i}" for i in _oracle_topk(qvecs[qi], cvecs, keep, k)]
            assert t["hit_ids"][row] == expect, (
                f"{name} {qid}: filter applied to the WRONG query row"
            )


def test_narrowing_handles_matchany_and_range_leaves(tmp_path):
    """`_narrow_gpu_leaf_state` has three shapes to slice: scalar codes, a
    (n_q, n_distinct) MatchAny membership matrix, and a dict of range bounds.
    The other tests only cover the scalar one."""
    rng = np.random.default_rng(5)
    n_c, n_q, k = 30, 8, 3
    cvecs = rng.normal(size=(n_c, DIM)).astype(np.float32)
    langs = [["eng", "fra", "deu"][i % 3] for i in range(n_c)]
    views = list(range(n_c))
    cdir = tmp_path / "c"
    cdir.mkdir()
    pq.write_table(pa.table({
        "dense_embedding": pa.array(cvecs.tolist(), pa.list_(pa.float32())),
        "sid": pa.array([f"c{i}" for i in range(n_c)]),
        "lang": pa.array(langs),
        "views": pa.array(views),
    }), str(cdir / "f0.parquet"))

    qvecs = rng.normal(size=(n_q, DIM)).astype(np.float32)
    anylist = [["eng", "deu"] if i % 2 == 0 else ["fra"] for i in range(n_q)]
    lo = [5 * (i % 4) for i in range(n_q)]
    pq.write_table(pa.table({
        "dense_embedding": pa.array(qvecs.tolist(), pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(n_q)]),
        "qset": pa.array(["use"] * 4 + ["other"] * 4),
        "langs": pa.array(anylist, pa.list_(pa.string())),
        "lo": pa.array(lo),
    }), str(tmp_path / "q.parquet"))

    out = tmp_path / "o"
    out.mkdir()
    res = run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="sid"),
        queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid",
                              payload_fields=["qset", "langs", "lo"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=8),
        searches=[
            SearchSpec(name="pad", k=k, metric="cosine",
                       rows={"column": "qset", "isin": ["other"]}),
            SearchSpec(name="any", k=k, metric="cosine",
                       rows={"column": "qset", "isin": ["use"]},
                       filter={"must": [{"field": "lang",
                                         "match_from_query": "langs"}]}),
            SearchSpec(name="rng", k=k, metric="cosine",
                       rows={"column": "qset", "isin": ["use"]},
                       filter={"must": [{"field": "views",
                                         "range_from_query": {"gte": "lo"}}]}),
        ],
    ))
    t = pq.read_table(res["any"]).to_pydict()
    for row, qid in enumerate(t["query_id"]):
        qi = int(qid[1:])
        keep = np.array([lg in anylist[qi] for lg in langs])
        assert t["hit_ids"][row] == [f"c{i}" for i in _oracle_topk(qvecs[qi], cvecs, keep, k)]
    t = pq.read_table(res["rng"]).to_pydict()
    for row, qid in enumerate(t["query_id"]):
        qi = int(qid[1:])
        keep = np.array([v >= lo[qi] for v in views])
        assert t["hit_ids"][row] == [f"c{i}" for i in _oracle_topk(qvecs[qi], cvecs, keep, k)]


def test_sparse_cosine_is_untouched_by_the_fusion(tmp_path):
    """Sparse cosine's divisor is the corpus ROW norm - per COLUMN, so it does
    reorder and can never be folded into a per-query-row scale. `spec_cos_scale`
    must stay None for it, leaving sparse scoring exactly as it was."""
    rng = np.random.default_rng(9)
    cdir = tmp_path / "c"
    cdir.mkdir()
    rows = [([0, 2, 5], [1.0, 2.0, 3.0]), ([1, 2], [4.0, 1.0]),
            ([0, 1, 5], [2.0, 2.0, 1.0]), ([3, 4], [5.0, 1.0])]
    def _sp(path, rs, **cols):
        data = {"sparse_embedding": pa.array(
            [{"indices": i, "values": v} for i, v in rs],
            type=pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                            pa.field("values", pa.list_(pa.float32()))]))}
        data.update({k: pa.array(v) for k, v in cols.items()})
        pq.write_table(pa.table(data), str(path))
    _sp(cdir / "f0.parquet", rows, sid=[f"c{i}" for i in range(len(rows))])
    _sp(tmp_path / "q.parquet", [([0, 2], [1.0, 1.0]), ([1, 5], [2.0, 1.0])],
        qid=["q0", "q1"])

    def run(tag):
        out = tmp_path / tag
        out.mkdir()
        return pq.read_table(run_compute(BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding",
                                id_column="sid"),
            queries=QueriesConfig(path=str(tmp_path / "q.parquet"),
                                  sparse_column="sparse_embedding", id_column="qid"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1),
            searches=[SearchSpec(name="s", k=3, metric="cosine",
                                 vector_type="sparse")],
        ))["s"]).to_pydict()

    t = run("sp")
    # real cosine values, so the corpus-norm divide still happened exactly once
    for row in t["hit_scores"]:
        for v in row:
            assert -1.0001 <= v <= 1.0001, f"sparse cosine produced {v}"
    assert any(len(r) for r in t["hit_ids"]), "sparse search returned nothing"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_fused_kernel_divide_matches_the_portable_divide_bit_for_bit():
    """THE test for the fused divide.

    The Triton kernel does `s / tl.load(NRM + row)`; the portable path does a
    torch divide. If Triton lowered that to `div.approx.f32` (fast math) instead
    of a correctly-rounded `div.rn.f32`, the two would disagree in the last bit
    and GPU and CPU runs would silently produce different keys — and therefore
    different winners at any tie boundary. Nothing else in the suite compares
    the two divides directly, because only this path performs one inside a
    kernel.

    Forcing the portable path via NOVA_BF_NO_TOPK_KERNEL is what makes this an
    honest comparison rather than the kernel checked against itself.
    """
    import os
    import torch
    from nova_bf.tiebreak import pack, pack_topk

    rng = np.random.default_rng(17)
    for n_q, n_cols, k in [(64, 512, 100), (7, 4096, 1000), (1000, 2048, 500)]:
        raw = rng.normal(size=(n_q, n_cols))
        # Values `div.full.f32` gets wrong where `div.rn.f32` does not: results
        # small enough to land in the subnormal range (div.full flushes those to
        # zero), plus -0.0 and the -inf the padding and masked cells carry.
        raw[0, :4] = [1e-38, -1e-38, 0.0, -0.0]
        raw[min(1, n_q - 1), :2] = [-np.inf, np.inf]
        sc = torch.tensor(raw, dtype=torch.float32, device="cuda").contiguous()
        nrm = rng.uniform(0.05, 50.0, size=n_q)
        nrm[0] = 3.0e37                      # drives row 0's quotients subnormal
        nrm = torch.tensor(nrm, dtype=torch.float32, device="cuda").contiguous()
        o = torch.arange(n_cols, dtype=torch.int64, device="cuda")

        kk, ik = pack_topk(sc, o, k, nrm)                      # Triton kernel
        prev = os.environ.get("NOVA_BF_NO_TOPK_KERNEL")
        os.environ["NOVA_BF_NO_TOPK_KERNEL"] = "1"
        try:
            kp, ip = pack_topk(sc, o, k, nrm)                  # portable
        finally:
            if prev is None:
                os.environ.pop("NOVA_BF_NO_TOPK_KERNEL", None)
            else:
                os.environ["NOVA_BF_NO_TOPK_KERNEL"] = prev

        assert torch.equal(kk.sort(1).values, kp.sort(1).values), (
            f"kernel and portable keys differ at {(n_q, n_cols, k)} — the "
            "Triton divide is not bit-identical to torch's"
        )
        # and both must equal packing scores that were divided up front
        ref = pack(sc / nrm[:, None], o)
        want = torch.topk(ref, k=k, dim=1, sorted=False).values
        assert torch.equal(kk.sort(1).values, want.sort(1).values)
