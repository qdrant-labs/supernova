"""Correctness tests for the multivector (ColBERT / late-interaction MaxSim)
brute-force compute path.

Mirrors test_compute_sparse.py's fixture/ground-truth pattern, but over
``list<list<float32>>`` columns — the same on-disk schema nova-embed writes
(see nova_embed.storage.writer.MULTIVECTOR_EMBEDDING_TYPE) and nova-load reads.

Ground truth is computed independently, in plain numpy, from the tiny synthetic
corpus/queries — a literal double loop over query tokens (max over doc tokens,
summed) — NOT by re-deriving nova_bf's tiled/segment-reduction logic, so these
tests catch divergence between the two. A zero-token (null or empty) doc is a
non-candidate: it never appears in the ground-truth ranking, exactly as Qdrant's
inverted MaxSim would never surface a doc with no vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # the compute phase needs torch (install nova-bf[dev])
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf.compute import run_compute
from nova_bf.config import load_config
from nova_bf.io import multivector_to_ragged

DIM = 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mv_array(docs: list[np.ndarray], nulls: set[int] = frozenset()) -> pa.Array:
    """A list of per-doc ``(n_tokens, DIM)`` arrays -> a ``list<list<float32>>``
    Arrow array. A doc index in ``nulls`` becomes a NULL outer entry (spans no
    tokens); an empty ``(0, DIM)`` array becomes a non-null empty inner list.
    Both are zero-token docs."""
    tok_counts = [0 if i in nulls else len(d) for i, d in enumerate(docs)]
    total = sum(tok_counts)
    flat = (
        np.concatenate([d.reshape(-1) for i, d in enumerate(docs) if i not in nulls and len(d)])
        if any(tok_counts) else np.empty(0, np.float32)
    )
    inner = pa.ListArray.from_arrays(
        pa.array(np.arange(0, total * DIM + 1, DIM, dtype=np.int32)),
        pa.array(flat.astype(np.float32), type=pa.float32()),
    )
    outer_off = np.concatenate([[0], np.cumsum(tok_counts)]).astype(np.int32)
    mask = pa.array([i in nulls for i in range(len(docs))]) if nulls else None
    return pa.ListArray.from_arrays(pa.array(outer_off), inner, mask=mask)


def _ref_maxsim(q: np.ndarray, d: np.ndarray, metric: str) -> float:
    """Reference MaxSim for one (query, doc) token-set pair — the definition,
    written out literally. A zero-token doc OR a zero-token query is a
    non-candidate (-inf): a query with no tokens has nothing to score, so it
    retrieves nothing (matching nova-bf and the sparse zero-support gate).
    Cosine normalizes with the same 1e-12 floor `torch.nn.functional.normalize`
    uses, so a zero-magnitude token becomes a zero vector (contributes 0), not a
    0/0 NaN."""
    if len(d) == 0 or len(q) == 0:
        return float("-inf")
    if metric == "cosine":
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return float(sum(max(float(qt @ dt) for dt in d) for qt in q))


def _ref_topk(qdocs, cdocs, metric, k, keep=None):
    """Independent per-query top-k ids+scores. `keep` (optional) is a set of
    admissible corpus ids (a filter)."""
    out = {}
    for qi, q in enumerate(qdocs):
        scores = []
        for di, d in enumerate(cdocs):
            if keep is not None and di not in keep:
                scores.append(float("-inf"))
            else:
                scores.append(_ref_maxsim(q, d, metric))
        scores = np.array(scores)
        order = [int(o) for o in np.argsort(-scores, kind="stable") if scores[o] > -np.inf][:k]
        out[qi] = {di: scores[di] for di in order}
    return out


def _run(tmp, cdir, qpath, *, metric="dot", k=5, bs=None, qb=None,
         filter_yaml="", id_cols=True):
    out = tmp / "out"
    out.mkdir(exist_ok=True)
    params = ["io_workers: 2"]
    if bs is not None:
        params.append(f"multivector_batch_size: {bs}")
    if qb is not None:
        params.append(f"multivector_query_block: {qb}")
    cfg_text = f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  {"id_column: id" if id_cols else ""}
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  {"id_column: qid" if id_cols else ""}
output:
  path: {out}
params: {{{", ".join(params)}}}
searches:
  - name: mv
    k: {k}
    metric: {metric}
    vector_type: multivector
{filter_yaml}
"""
    p = tmp / "cfg.yaml"
    p.write_text(cfg_text)
    t = pq.read_table(run_compute(load_config(str(p)))["mv"]).to_pydict()
    # Keyed by query ROW position (output rows are in query-file order), so it
    # works whether hit/query ids are row-index strings (id_cols) or opaque
    # make_point_id UUIDs. `cast` int-decodes hit ids only when they're indices.
    cast = int if id_cols else (lambda x: x)
    return {
        qi: {cast(i): s for i, s in zip(hi, hs)}
        for qi, (hi, hs) in enumerate(zip(t["hit_ids"], t["hit_scores"]))
    }


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    rng = np.random.default_rng(7)
    tmp = tmp_path_factory.mktemp("bf_mv")
    cdir = tmp / "corpus"
    cdir.mkdir()
    # 3 files with row counts that don't divide typical batch sizes; a mix of
    # token counts, plus deliberate null and empty (zero-token) docs.
    sizes = [6, 5, 4]
    cdocs, langs = [], []
    g = 0
    # which global doc indices are null / empty
    null_ids = {2, 9}
    empty_ids = {7}
    for fi, n in enumerate(sizes):
        docs, file_nulls = [], set()
        for r in range(n):
            gid = g + r
            if gid in null_ids or gid in empty_ids:
                docs.append(np.zeros((0, DIM), np.float32))
                if gid in null_ids:
                    file_nulls.add(r)
            else:
                docs.append(rng.standard_normal((int(rng.integers(1, 6)), DIM)).astype(np.float32))
        arr = _mv_array(docs, nulls=file_nulls)
        ids = [str(g + r) for r in range(n)]
        lang = ["eng" if (g + r) % 2 == 0 else "fra" for r in range(n)]
        pq.write_table(
            pa.table({"multivector_embedding": arr,
                      "id": pa.array(ids), "lang": pa.array(lang)}),
            str(cdir / f"c{fi}.parquet"),
        )
        cdocs.extend(docs)
        langs.extend(lang)
        g += n

    # queries: a few with 1..4 tokens
    qdocs = [rng.standard_normal((int(rng.integers(1, 5)), DIM)).astype(np.float32) for _ in range(6)]
    pq.write_table(
        pa.table({"multivector_embedding": _mv_array(qdocs),
                  "qid": pa.array([str(i) for i in range(len(qdocs))])}),
        str(tmp / "q.parquet"),
    )
    return {
        "tmp": tmp, "cdir": str(cdir), "qpath": str(tmp / "q.parquet"),
        "cdocs": cdocs, "qdocs": qdocs, "langs": langs,
        "null_ids": null_ids, "empty_ids": empty_ids,
    }


# ---------------------------------------------------------------------------
# 1. exact segment-reduction pinning on a hand-computed case
# ---------------------------------------------------------------------------
def test_handcomputed_maxsim_exact():
    """A tiny fully-worked example: pin the segment-max (over doc tokens) then
    segment-sum (over query tokens) against hand arithmetic, not just the
    numpy reference."""
    import torch
    from nova_bf.compute import MultiVectorCorpusBatch, MultiVectorQuery

    # doc0: two tokens; query: two tokens; DIM=2, plain integers -> exact
    d = np.array([[1.0, 0.0], [0.0, 2.0]], np.float32)          # doc0 tokens
    q = np.array([[1.0, 1.0], [2.0, 0.0]], np.float32)          # query tokens
    # dot products:
    #   q0.d0 = 1, q0.d1 = 2  -> max 2
    #   q1.d0 = 2, q1.d1 = 0  -> max 2
    # MaxSim = 2 + 2 = 4
    batch = MultiVectorCorpusBatch(np.array([0, 2], np.int64), d, None)
    Q = MultiVectorQuery(torch.tensor(q), torch.tensor(np.array([0, 2], np.int64)), 1, None)
    sc = batch.transfer(0, 1, "cpu").score(Q, "dot")
    assert sc.shape == (1, 1)
    assert abs(float(sc[0, 0]) - 4.0) < 1e-6


@pytest.mark.parametrize("metric", ["dot", "cosine"])
@pytest.mark.parametrize("bs,qb", [(None, None), (1, 1), (3, 2), (2, 4), (1000, 1000)])
def test_tiling_invariance(ds, metric, bs, qb):
    """Same ranking + scores regardless of corpus-row tile (`bs`) and
    query-axis tile (`qb`), including tile=1 and tile>data — the whole point of
    hiding tiling inside `.score()`."""
    got = _run(ds["tmp"], ds["cdir"], ds["qpath"], metric=metric, k=8, bs=bs, qb=qb)
    ref = _ref_topk(ds["qdocs"], ds["cdocs"], metric, 8)
    for qi in ref:
        assert list(got[qi].keys()) == list(ref[qi].keys()), (
            f"{metric} bs={bs} qb={qb} q{qi} ids: got {list(got[qi])} ref {list(ref[qi])}")
        for di in ref[qi]:
            assert abs(got[qi][di] - ref[qi][di]) < 1e-3


def test_null_and_empty_docs_are_noncandidates(ds):
    """Every null / zero-token doc must be absent from every query's results —
    with k >= corpus size so the ONLY reason to be missing is being a
    non-candidate."""
    n_corpus = len(ds["cdocs"])
    got = _run(ds["tmp"], ds["cdir"], ds["qpath"], metric="dot", k=n_corpus + 5)
    bad = ds["null_ids"] | ds["empty_ids"]
    for qi, hits in got.items():
        assert not (set(hits) & bad), f"q{qi} surfaced non-candidate docs {set(hits) & bad}"
        # everything else IS a candidate, so exactly the candidates appear
        assert set(hits) == set(range(n_corpus)) - bad


def test_uniform_filter_parity(ds):
    """A uniform `must match lang=eng` filter selects the same candidate set the
    reference does, with identical MaxSim scores."""
    keep = {i for i, l in enumerate(ds["langs"]) if l == "eng"} \
        - (ds["null_ids"] | ds["empty_ids"])
    filter_yaml = (
        "    filter:\n"
        "      must:\n"
        "        - field: lang\n"
        "          match: eng\n"
    )
    got = _run(ds["tmp"], ds["cdir"], ds["qpath"], metric="dot", k=8, filter_yaml=filter_yaml)
    ref = _ref_topk(ds["qdocs"], ds["cdocs"], "dot", 8,
                    keep={i for i, l in enumerate(ds["langs"]) if l == "eng"})
    for qi in ref:
        assert list(got[qi].keys()) == list(ref[qi].keys())
        for di in ref[qi]:
            assert abs(got[qi][di] - ref[qi][di]) < 1e-3
        assert set(got[qi]) <= keep


def test_make_point_id_path(ds):
    """Without an id_column, hits resolve via make_point_id(file_key, row) — the
    ranking (by resolved order within each query) must still match; we compare
    the SCORE multiset per query since ids are now opaque strings."""
    got = _run(ds["tmp"], ds["cdir"], ds["qpath"], metric="dot", k=8, id_cols=False)
    ref = _ref_topk(ds["qdocs"], ds["cdocs"], "dot", 8)
    for qi in ref:
        gs = sorted(got[qi].values())
        rs = sorted(ref[qi].values())
        assert len(gs) == len(rs)
        assert np.allclose(gs, rs, atol=1e-3)


def test_token_budget_autoderives_tiles(ds):
    """`multivector_token_budget` fills in both tile knobs and still produces the
    exact same ranking as the untiled run."""
    out = ds["tmp"] / "out"
    out.mkdir(exist_ok=True)
    cfg_text = f"""
corpus: {{path: {ds["cdir"]}, multivector_column: multivector_embedding, id_column: id}}
queries: {{path: {ds["qpath"]}, multivector_column: multivector_embedding, id_column: qid}}
output: {{path: {out}}}
params: {{io_workers: 2, multivector_token_budget: 64}}
searches:
  - {{name: mv, k: 8, metric: dot, vector_type: multivector}}
"""
    p = ds["tmp"] / "cfg_tb.yaml"
    p.write_text(cfg_text)
    t = pq.read_table(run_compute(load_config(str(p)))["mv"]).to_pydict()
    got = {int(q): {int(i): s for i, s in zip(hi, hs)}
           for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}
    ref = _ref_topk(ds["qdocs"], ds["cdocs"], "dot", 8)
    for qi in ref:
        assert list(got[qi].keys()) == list(ref[qi].keys())


# ---------------------------------------------------------------------------
# decoder unit tests
# ---------------------------------------------------------------------------
def test_cosine_is_scale_invariant(tmp_path):
    """nova-bf's cosine MaxSim is the mathematically exact, scale-invariant
    cosine: scaling any doc's (or query's) token magnitudes leaves the cosine
    ranking and scores unchanged (normalization floor 1e-12). This pins the
    intentional semantics that diverge from Qdrant's low-norm guard only for
    sub-~1e-3-norm vectors (see docs/brute-force/multivector-maxsim.md) — a
    regime real embeddings never reach."""
    rng = np.random.default_rng(21)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    cdocs = [rng.standard_normal((int(rng.integers(1, 5)), DIM)).astype(np.float32) for _ in range(15)]
    qdocs = [rng.standard_normal((3, DIM)).astype(np.float32) for _ in range(4)]

    def run_with_scale(scale):
        scaled = [d * scale for d in cdocs]
        pq.write_table(pa.table({"multivector_embedding": _mv_array(scaled),
                                 "id": pa.array([str(i) for i in range(len(scaled))])}),
                       str(cdir / "c0.parquet"))
        qpath = tmp_path / "q.parquet"
        pq.write_table(pa.table({"multivector_embedding": _mv_array(qdocs),
                                 "qid": pa.array([str(i) for i in range(len(qdocs))])}), str(qpath))
        return _run(tmp_path, str(cdir), str(qpath), metric="cosine", k=10)

    base = run_with_scale(1.0)
    for scale in (0.01, 100.0):
        got = run_with_scale(scale)
        for qi in base:
            assert list(got[qi].keys()) == list(base[qi].keys()), f"scale={scale} q{qi} ranking drift"
            for di in base[qi]:
                assert abs(got[qi][di] - base[qi][di]) < 1e-3, f"scale={scale} q{qi} d{di} score drift"


def test_allow_tf32_flag_roundtrips_and_is_correct(ds):
    """`params.allow_tf32` is accepted and a run with it set still produces the
    exact ranking. On this CPU box it's a no-op (torch's TF32 flag is CUDA-only),
    so results must be identical to the default run — this pins the plumbing;
    the actual TF32 speedup + ranking-preservation + Qdrant parity were measured
    live on an A10G (see docs/brute-force/multivector-maxsim.md)."""
    base = _run(ds["tmp"], ds["cdir"], ds["qpath"], metric="dot", k=8)
    out = ds["tmp"] / "out_tf32"
    out.mkdir(exist_ok=True)
    cfg_text = f"""
corpus: {{path: {ds["cdir"]}, multivector_column: multivector_embedding, id_column: id}}
queries: {{path: {ds["qpath"]}, multivector_column: multivector_embedding, id_column: qid}}
output: {{path: {out}}}
params: {{io_workers: 2, allow_tf32: true}}
searches:
  - {{name: mv, k: 8, metric: dot, vector_type: multivector}}
"""
    p = ds["tmp"] / "cfg_tf32.yaml"
    p.write_text(cfg_text)
    t = pq.read_table(run_compute(load_config(str(p)))["mv"]).to_pydict()
    got = {int(q): [int(i) for i in hi] for q, hi in zip(t["query_id"], t["hit_ids"])}
    for qi in base:
        assert got[qi] == list(base[qi].keys())


def test_decoder_null_empty_and_slice():
    D = 3
    inner = pa.ListArray.from_arrays(
        pa.array([0, 3, 6, 9], type=pa.int32()),
        pa.array(np.arange(9, dtype=np.float32), type=pa.float32()),
    )
    # doc0: 2 tok, doc1: null, doc2: 1 tok, doc3: empty
    outer = pa.ListArray.from_arrays(
        pa.array([0, 2, 2, 3, 3], type=pa.int32()), inner,
        mask=pa.array([False, True, False, False]),
    )
    off, flat = multivector_to_ragged(pa.chunked_array([outer]))
    assert off.tolist() == [0, 2, 2, 3, 3]
    assert flat.shape == (3, D)
    # a sliced array (buffer offset != 0) still decodes correctly
    off2, flat2 = multivector_to_ragged(pa.chunked_array([outer.slice(2, 2)]))
    assert off2.tolist() == [0, 1, 1]                # doc2 (1 tok), doc3 (empty)
    assert np.array_equal(flat2, np.array([[6.0, 7.0, 8.0]], np.float32))


def test_zero_token_query_is_tiling_invariant_noncandidate(tmp_path):
    """A zero-token (empty/null) query has nothing to score → it must retrieve
    NOTHING (all -inf), identically regardless of how `query_block` tiles the
    query axis. Regression for the bug where a zero-token query sharing a block
    with a non-empty one scored 0.0 (candidate) while a whole zero-token block
    scored -inf — making the GT depend on a pure performance knob."""
    rng = np.random.default_rng(3)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    cdocs = [rng.standard_normal((int(rng.integers(1, 5)), DIM)).astype(np.float32) for _ in range(12)]
    pq.write_table(pa.table({"multivector_embedding": _mv_array(cdocs),
                             "id": pa.array([str(i) for i in range(12)])}),
                   str(cdir / "c0.parquet"))
    # queries: q0 non-empty, q1 ZERO-token, q2 non-empty, q3 zero-token
    qdocs = [rng.standard_normal((3, DIM)).astype(np.float32),
             np.zeros((0, DIM), np.float32),
             rng.standard_normal((2, DIM)).astype(np.float32),
             np.zeros((0, DIM), np.float32)]
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({"multivector_embedding": _mv_array(qdocs),
                             "qid": pa.array([str(i) for i in range(4)])}), str(qpath))

    runs = {qb: _run(tmp_path, str(cdir), str(qpath), metric="dot", k=12, qb=qb)
            for qb in (None, 1, 2, 3, 4)}
    # zero-token queries q1, q3 -> NO hits, in every tiling
    for qb, got in runs.items():
        assert got[1] == {}, f"qb={qb}: zero-token q1 should retrieve nothing, got {got[1]}"
        assert got[3] == {}, f"qb={qb}: zero-token q3 should retrieve nothing, got {got[3]}"
    # non-empty queries identical across all tilings
    base = runs[None]
    for qb, got in runs.items():
        for qi in (0, 2):
            assert list(got[qi].keys()) == list(base[qi].keys()), f"qb={qb} q{qi} ids drift"


def test_all_empty_corpus_shard_coalesced(tmp_path):
    """A corpus shard that is ENTIRELY zero-token docs must not crash a
    coalesced run (uniform filter + multivector_batch_size set), and its docs
    must be non-candidates. Regression for the `_concat_multivector_batches`
    width-0 concat crash."""
    rng = np.random.default_rng(5)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # file 0: real docs (cat=a/b); file 1: ALL null/empty docs; file 2: real docs
    def write(fi, docs, nulls, cats, base):
        pq.write_table(pa.table({
            "multivector_embedding": _mv_array(docs, nulls=nulls),
            "id": pa.array([str(base + i) for i in range(len(docs))]),
            "cat": pa.array(cats),
        }), str(cdir / f"c{fi}.parquet"))
    f0 = [rng.standard_normal((int(rng.integers(1, 5)), DIM)).astype(np.float32) for _ in range(5)]
    write(0, f0, set(), ["a" if i % 2 else "b" for i in range(5)], 0)
    f1 = [np.zeros((0, DIM), np.float32) for _ in range(4)]     # ALL empty (2 null, 2 empty-list)
    write(1, f1, {0, 1}, ["a"] * 4, 5)
    f2 = [rng.standard_normal((int(rng.integers(1, 5)), DIM)).astype(np.float32) for _ in range(6)]
    write(2, f2, set(), ["a" if i % 2 else "b" for i in range(6)], 9)

    qdocs = [rng.standard_normal((3, DIM)).astype(np.float32) for _ in range(4)]
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({"multivector_embedding": _mv_array(qdocs),
                             "qid": pa.array([str(i) for i in range(4)])}), str(qpath))

    filter_yaml = ("    filter:\n      must:\n        - field: cat\n          match: a\n")
    # multivector_batch_size set -> coalescing path; k >= corpus so a candidate is
    # missing ONLY if it's a non-candidate or filtered out.
    got = _run(tmp_path, str(cdir), str(qpath), metric="dot", k=100, bs=8, filter_yaml=filter_yaml)
    cdocs = f0 + f1 + f2
    cats = (["a" if i % 2 else "b" for i in range(5)] + ["a"] * 4
            + ["a" if i % 2 else "b" for i in range(6)])
    empty_ids = {5, 6, 7, 8}
    for qi in range(4):
        assert not (set(got[qi]) & empty_ids), f"q{qi} surfaced empty-shard docs"
        keep = {i for i, c in enumerate(cats) if c == "a"} - empty_ids
        assert set(got[qi]) == keep, f"q{qi} candidate set mismatch"
        ref = _ref_topk(qdocs, cdocs, "dot", 100,
                        keep={i for i, c in enumerate(cats) if c == "a"})
        for di in ref[qi]:
            assert abs(got[qi][di] - ref[qi][di]) < 1e-3


def test_coalescing_uniform_filter_parity(ds):
    """Same as the uniform-filter test but WITH multivector_batch_size set, so
    the cross-file coalescing path (`_flush_coalesce_group` +
    `_concat_multivector_batches`) actually executes."""
    filter_yaml = ("    filter:\n      must:\n        - field: lang\n          match: eng\n")
    got = _run(ds["tmp"], ds["cdir"], ds["qpath"], metric="dot", k=8, bs=3, filter_yaml=filter_yaml)
    ref = _ref_topk(ds["qdocs"], ds["cdocs"], "dot", 8,
                    keep={i for i, l in enumerate(ds["langs"]) if l == "eng"})
    for qi in ref:
        assert list(got[qi].keys()) == list(ref[qi].keys())
        for di in ref[qi]:
            assert abs(got[qi][di] - ref[qi][di]) < 1e-3


def test_multivector_and_dense_shared_run(tmp_path):
    """A dense spec and a multivector spec in ONE run share corpus IO/decode —
    each must still produce its own independent, correct ranking."""
    rng = np.random.default_rng(11)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    M = 40
    cdocs = [rng.standard_normal((int(rng.integers(1, 5)), DIM)).astype(np.float32) for _ in range(M)]
    dense_c = rng.standard_normal((M, DIM)).astype(np.float32)
    pq.write_table(pa.table({
        "multivector_embedding": _mv_array(cdocs),
        "dense_embedding": pa.array(dense_c.tolist(), type=pa.list_(pa.float32())),
        "id": pa.array([str(i) for i in range(M)]),
    }), str(cdir / "c0.parquet"))
    NQ = 5
    qmv = [rng.standard_normal((3, DIM)).astype(np.float32) for _ in range(NQ)]
    dense_q = rng.standard_normal((NQ, DIM)).astype(np.float32)
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "multivector_embedding": _mv_array(qmv),
        "dense_embedding": pa.array(dense_q.tolist(), type=pa.list_(pa.float32())),
        "qid": pa.array([str(i) for i in range(NQ)]),
    }), str(qpath))
    out = tmp_path / "out"
    out.mkdir()
    cfg_text = f"""
corpus: {{path: {cdir}, multivector_column: multivector_embedding, dense_column: dense_embedding, id_column: id}}
queries: {{path: {qpath}, multivector_column: multivector_embedding, dense_column: dense_embedding, id_column: qid}}
output: {{path: {out}}}
params: {{io_workers: 2, multivector_query_block: 2}}
searches:
  - {{name: mv, k: 10, metric: dot, vector_type: multivector}}
  - {{name: dn, k: 10, metric: dot, vector_type: dense}}
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg_text)
    res = run_compute(load_config(str(p)))
    # multivector spec
    tm = pq.read_table(res["mv"]).to_pydict()
    gm = {int(q): {int(i): s for i, s in zip(hi, hs)}
          for q, hi, hs in zip(tm["query_id"], tm["hit_ids"], tm["hit_scores"])}
    rm = _ref_topk(qmv, cdocs, "dot", 10)
    for qi in rm:
        assert list(gm[qi].keys()) == list(rm[qi].keys())
        for di in rm[qi]:
            assert abs(gm[qi][di] - rm[qi][di]) < 1e-3
    # dense spec: independent dot-product ranking
    td = pq.read_table(res["dn"]).to_pydict()
    gd = {int(q): [int(i) for i in hi] for q, hi in zip(td["query_id"], td["hit_ids"])}
    for qi in range(NQ):
        sc = dense_c @ dense_q[qi]
        ref = [int(o) for o in np.argsort(-sc, kind="stable")[:10]]
        assert gd[qi] == ref, f"dense q{qi} ranking mismatch"


def test_per_query_filter_multivector(tmp_path):
    """A per-query filter (`match_from_query`) with a multivector search — the
    per-query cell-mask path — selects each query's own admissible docs."""
    rng = np.random.default_rng(13)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    M = 30
    cdocs = [rng.standard_normal((int(rng.integers(1, 5)), DIM)).astype(np.float32) for _ in range(M)]
    tenant = [i % 3 for i in range(M)]
    pq.write_table(pa.table({
        "multivector_embedding": _mv_array(cdocs),
        "id": pa.array([str(i) for i in range(M)]),
        "tenant": pa.array(tenant),
    }), str(cdir / "c0.parquet"))
    NQ = 4
    qmv = [rng.standard_normal((3, DIM)).astype(np.float32) for _ in range(NQ)]
    want_tenant = [0, 1, 2, 0]
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "multivector_embedding": _mv_array(qmv),
        "qid": pa.array([str(i) for i in range(NQ)]),
        "want": pa.array(want_tenant),
    }), str(qpath))
    out = tmp_path / "out"
    out.mkdir()
    cfg_text = f"""
corpus: {{path: {cdir}, multivector_column: multivector_embedding, id_column: id}}
queries: {{path: {qpath}, multivector_column: multivector_embedding, id_column: qid}}
output: {{path: {out}}}
params: {{io_workers: 2}}
searches:
  - name: mv
    k: 100
    metric: dot
    vector_type: multivector
    filter:
      must:
        - field: tenant
          match_from_query: want
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg_text)
    t = pq.read_table(run_compute(load_config(str(p)))["mv"]).to_pydict()
    got = {int(q): {int(i): s for i, s in zip(hi, hs)}
           for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}
    for qi in range(NQ):
        keep = {i for i in range(M) if tenant[i] == want_tenant[qi]}
        assert set(got[qi]) == keep, f"q{qi} tenant filter: got {set(got[qi])} want {keep}"
        ref = _ref_topk(qmv, cdocs, "dot", 100, keep=keep)
        for di in ref[qi]:
            assert abs(got[qi][di] - ref[qi][di]) < 1e-3


def test_decoder_null_doc_with_physical_span_is_zero_token():
    """A NULL outer entry must be a zero-token doc even if Arrow gives its slot
    a non-empty physical offset span — the validity bitmap wins, not the
    offsets. Regression for trusting offsets over the null mask."""
    D = 2
    inner = pa.ListArray.from_arrays(
        pa.array([0, 2, 4, 6, 8], type=pa.int32()),
        pa.array(np.arange(8, dtype=np.float32), type=pa.float32()),
    )  # 4 tokens
    # doc0 = tokens[0:2], doc1 = NULL but offsets claim [2:4], doc2 = tokens[4:4]
    outer = pa.ListArray.from_arrays(
        pa.array([0, 2, 4, 4], type=pa.int32()), inner, mask=pa.array([False, True, False]))
    off, flat = multivector_to_ragged(pa.chunked_array([outer]))
    assert (off[2] - off[1]) == 0, "null doc must be zero-token regardless of its span"
    # the stray tokens the null slot physically carried are excluded
    assert flat.shape[0] == 2 and off.tolist() == [0, 2, 2, 2]
    assert np.array_equal(flat, np.arange(4, dtype=np.float32).reshape(2, 2))


def test_decoder_rejects_dense_and_null_token_values():
    dense = pa.array([[1.0, 2.0, 3.0]], type=pa.list_(pa.float32()))
    with pytest.raises(TypeError, match="DENSE|list of token"):
        multivector_to_ragged(pa.chunked_array([dense]))
    # a null float inside a token vector must not silently become NaN
    inner = pa.ListArray.from_arrays(pa.array([0, 2], type=pa.int32()),
                                     pa.array([1.0, None], type=pa.float32()))
    outer = pa.ListArray.from_arrays(pa.array([0, 1], type=pa.int32()), inner)
    with pytest.raises(ValueError, match="null"):
        multivector_to_ragged(pa.chunked_array([outer]))


def test_cross_file_dim_mismatch_raises(tmp_path):
    """Two query files with different token dims must fail with a clear message,
    not a generic numpy concat error."""
    def mv_dim(docs, d):  # dim-aware builder (the shared _mv_array hardcodes DIM)
        tc = [len(x) for x in docs]; total = sum(tc)
        flat = np.concatenate([x.reshape(-1) for x in docs]).astype(np.float32)
        inner = pa.ListArray.from_arrays(pa.array(np.arange(0, total * d + 1, d, dtype=np.int32)),
                                         pa.array(flat, type=pa.float32()))
        off = np.concatenate([[0], np.cumsum(tc)]).astype(np.int32)
        return pa.ListArray.from_arrays(pa.array(off), inner)
    rng = np.random.default_rng(0)
    cdir = tmp_path / "corpus"; cdir.mkdir()
    pq.write_table(pa.table({"multivector_embedding": mv_dim([rng.standard_normal((2, DIM)).astype(np.float32)], DIM),
                             "id": pa.array(["0"])}), str(cdir / "c0.parquet"))
    qdir = tmp_path / "q"; qdir.mkdir()
    pq.write_table(pa.table({"multivector_embedding": mv_dim([rng.standard_normal((2, DIM)).astype(np.float32)], DIM),
                             "qid": pa.array(["0"])}), str(qdir / "q0.parquet"))
    pq.write_table(pa.table({"multivector_embedding": mv_dim([rng.standard_normal((2, DIM + 1)).astype(np.float32)], DIM + 1),
                             "qid": pa.array(["1"])}), str(qdir / "q1.parquet"))
    out = tmp_path / "out"; out.mkdir()
    cfg = f"""
corpus: {{path: {cdir}, multivector_column: multivector_embedding, id_column: id}}
queries: {{path: {qdir}, multivector_column: multivector_embedding, id_column: qid}}
output: {{path: {out}}}
params: {{io_workers: 1}}
searches: [{{name: mv, k: 1, metric: dot, vector_type: multivector}}]
"""
    p = tmp_path / "cfg.yaml"; p.write_text(cfg)
    with pytest.raises(ValueError, match="token dim mismatch"):
        run_compute(load_config(str(p)))


def test_decoder_all_null_and_empty_column():
    D = 4
    inner = pa.ListArray.from_arrays(pa.array([0], type=pa.int32()),
                                     pa.array([], type=pa.float32()))
    alln = pa.ListArray.from_arrays(pa.array([0, 0, 0], type=pa.int32()), inner,
                                    mask=pa.array([True, True]))
    off, flat = multivector_to_ragged(pa.chunked_array([alln]))
    assert off.tolist() == [0, 0, 0]
    assert flat.shape[0] == 0
    empty = pa.array([], type=pa.list_(pa.list_(pa.float32())))
    off2, flat2 = multivector_to_ragged(pa.chunked_array([empty]))
    assert off2.tolist() == [0]
    assert flat2.shape[0] == 0
