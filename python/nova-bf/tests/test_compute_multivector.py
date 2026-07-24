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
    written out literally. Zero-token doc -> non-candidate (-inf)."""
    if len(d) == 0 or len(q) == 0:
        return float("-inf") if len(d) == 0 else 0.0
    if metric == "cosine":
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
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
