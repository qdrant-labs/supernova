"""Correctness tests for `BruteForceConfig.searches` — running several
independent top-K searches (dense/sparse x filtered/unfiltered) against the
SAME corpus in one `run_compute`/`run_merge` call, sharing corpus file IO and
per-vector-type decode across specs, with specs sharing the same
vector_type+filter further sharing filter evaluation, row compaction, and GPU
transfer/CSR build (see compute.py's `SpecGroup`) — only each search's own
scoring and top-K stay independent.

Ground truth for each spec is computed independently in plain numpy (not by
re-deriving nova_bf's own scoring), mirroring test_compute.py/
test_compute_sparse.py's pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # the compute phase needs torch (install nova-bf[dev])
import pyarrow as pa
import pyarrow.parquet as pq

import nova_bf.compute as compute_mod
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
from nova_bf.io import Store
from nova_bf.merge import run_merge

DIM, VOCAB, NNZ = 8, 12, 8
SIZES = [5, 7, 4]  # 3 corpus files → 16 vectors; row counts that don't divide the batch


def _random_sparse_row(rng, vocab=VOCAB, nnz=NNZ):
    idx = rng.choice(vocab, size=nnz, replace=False)
    val = rng.standard_normal(nnz).astype(np.float32)
    return idx.tolist(), val.tolist()


def _write_combined(path, dense_rows, sparse_rows, **columns):
    """A corpus/query parquet carrying BOTH dense_embedding and
    sparse_embedding columns side by side, plus arbitrary payload columns —
    same shape `nova bf` expects when a run's `searches` mixes vector_types."""
    data = {
        "dense_embedding": pa.array(np.asarray(dense_rows).tolist(), type=pa.list_(pa.float32())),
        "sparse_embedding": pa.array(
            [{"indices": idx, "values": val} for idx, val in sparse_rows],
            type=pa.struct([
                pa.field("indices", pa.list_(pa.uint32())),
                pa.field("values", pa.list_(pa.float32())),
            ]),
        ),
    }
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


def _densify_sparse(rows, vocab=VOCAB):
    out = np.zeros((len(rows), vocab), dtype=np.float64)
    for i, (idx, val) in enumerate(rows):
        out[i, idx] = val
    return out


@pytest.fixture
def ds(tmp_path):
    """A tiny corpus (both dense+sparse columns, a filterable `language`
    column) + query set, with independent ground truth for either vector_type,
    optionally restricted to an `allowed` global-index subset (filter tests)."""
    rng = np.random.default_rng(0)
    cdir = tmp_path / "corpus"
    cdir.mkdir()

    dense_parts, sparse_rows, ids_by_g, lang_by_g, g = [], [], [], [], 0
    for fi, n in enumerate(SIZES):
        dense_rows = rng.standard_normal((n, DIM)).astype(np.float32)
        s_rows = [_random_sparse_row(rng) for _ in range(n)]
        ids = [f"c{g + r}" for r in range(n)]
        lang = ["eng" if (g + r) % 2 == 0 else "fra" for r in range(n)]
        _write_combined(cdir / f"f{fi}.parquet", dense_rows, s_rows, id=ids, language=lang)
        dense_parts.append(dense_rows)
        sparse_rows += s_rows
        ids_by_g += ids
        lang_by_g += lang
        g += n
    dense_corpus = np.concatenate(dense_parts)

    n_q = 4
    q_dense = rng.standard_normal((n_q, DIM)).astype(np.float32)
    q_sparse = [_random_sparse_row(rng) for _ in range(n_q)]
    qids = [f"q{i}" for i in range(n_q)]
    qpath = tmp_path / "queries.parquet"
    _write_combined(qpath, q_dense, q_sparse, qid=qids)

    q_sparse_dense = _densify_sparse(q_sparse)
    corpus_sparse_dense = _densify_sparse(sparse_rows)

    def ground_truth(vector_type, k, allowed=None):
        rows = allowed if allowed is not None else list(range(len(ids_by_g)))
        if vector_type == "dense":
            s = q_dense @ dense_corpus[rows].T
        else:
            s = q_sparse_dense @ corpus_sparse_dense[rows].T
        expected = {}
        for i, qid in enumerate(qids):
            order = np.argsort(-s[i])[:k]
            expected[qid] = [ids_by_g[rows[j]] for j in order]
        return expected

    return {
        "cdir": str(cdir),
        "qpath": str(qpath),
        "tmp": tmp_path,
        "qids": qids,
        "ids_by_g": ids_by_g,
        "lang_by_g": lang_by_g,
        "ground_truth": ground_truth,
    }


def _out(ds, name):
    p = ds["tmp"] / name
    p.mkdir(exist_ok=True)
    return str(p)


def test_multi_spec_matches_independent_ground_truth(ds):
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        SearchSpec(name="dense_all", vector_type="dense", metric="dot", k=3),
        SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=2, filter=eng_filter),
        SearchSpec(name="sparse_all", vector_type="sparse", metric="dot", k=3, corpus_batch_size=2),
        SearchSpec(name="sparse_eng", vector_type="sparse", metric="dot", k=2, filter=eng_filter),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "multi_out")),
        searches=specs,
    )
    paths = run_compute(cfg)
    assert set(paths) == {"dense_all", "dense_eng", "sparse_all", "sparse_eng"}

    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    expectations = {
        "dense_all": ds["ground_truth"]("dense", 3),
        "dense_eng": ds["ground_truth"]("dense", 2, allowed=eng_globals),
        "sparse_all": ds["ground_truth"]("sparse", 3),
        "sparse_eng": ds["ground_truth"]("sparse", 2, allowed=eng_globals),
    }
    for name, expected in expectations.items():
        t = pq.read_table(paths[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"


def test_grouped_matches_ungrouped_per_search(ds):
    """The core regression guard for SpecGroup: running several searches
    together (several of which share the same vector_type+filter, so they
    land in the same SpecGroup and share GPU transfer/CSR build) must produce
    results BIT-IDENTICAL to running each of those same searches alone, one
    per `run_compute` call. Covers dense+sparse, and — for dense — all three
    metrics (cosine/dot/euclidean) sharing one group, since dense cosine/dot/
    euclidean all read the SAME shared, un-normalized `Cb` (see `_scores`)."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        # group: (dense, filter=None) — 3 members, all 3 metrics
        SearchSpec(name="dense_dot", vector_type="dense", metric="dot", k=3),
        SearchSpec(name="dense_cos", vector_type="dense", metric="cosine", k=2),
        SearchSpec(name="dense_euclid", vector_type="dense", metric="euclidean", k=2),
        # group: (dense, eng_filter) — 1 member (distinct group from the above)
        SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=2, filter=eng_filter),
        # group: (sparse, filter=None) — 2 members, mixed metrics (the
        # restructured path — see test_three_member_group_mixed_metrics for
        # a 3-member version with independently-verified ground truth)
        SearchSpec(name="sparse_dot", vector_type="sparse", metric="dot", k=3),
        SearchSpec(name="sparse_cos", vector_type="sparse", metric="cosine", k=2),
    ]

    combined_cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "grouped_out")),
        searches=specs,
    )
    combined_paths = run_compute(combined_cfg)

    for spec in specs:
        solo_cfg = BruteForceConfig(
            corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
            queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
            output=OutputConfig(path=_out(ds, f"solo_{spec.name}_out")),
            searches=[spec],
        )
        solo_path = run_compute(solo_cfg)[spec.name]

        combined_t = pq.read_table(combined_paths[spec.name]).to_pydict()
        solo_t = pq.read_table(solo_path).to_pydict()
        assert combined_t["hit_ids"] == solo_t["hit_ids"], f"search={spec.name}"
        for combined_scores, solo_scores in zip(combined_t["hit_scores"], solo_t["hit_scores"]):
            assert np.allclose(combined_scores, solo_scores, atol=1e-5), f"search={spec.name}"


def test_three_member_group_mixed_metrics(tmp_path):
    """One SpecGroup (same vector_type=sparse, both unfiltered) with THREE
    members: two `cosine`, one `dot` — directly targets the two riskiest new
    bug classes from grouping sparse searches: double-normalization (a cosine
    member's score divided by row_norms twice) and cross-member contamination
    (one member reading another's scores). Verified against independently
    hand-computed dot products and cosine similarities, not nova_bf's own
    scoring — c1 and c2 are deliberately given the identical cosine similarity
    (same angle to the query, different magnitude) so a correct
    implementation must still recover their very different DOT scores,
    exactly like the two-spec version of this regression test."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # c0 = {1: 3.0, 2: 4.0}, norm 5.0 | c1 = {1: 1.0}, norm 1.0 | c2 = {2: 2.0}, norm 2.0
    corpus_rows = [([1, 2], [3.0, 4.0]), ([1], [1.0]), ([2], [2.0])]
    dummy_dense = np.random.default_rng(4).standard_normal((3, DIM)).astype(np.float32)
    _write_combined(cdir / "f0.parquet", dummy_dense, corpus_rows, id=["c0", "c1", "c2"])

    # q0 = {1: 1.0, 2: 1.0}, norm sqrt(2)
    query_rows = [([1, 2], [1.0, 1.0])]
    qpath = tmp_path / "queries.parquet"
    _write_combined(qpath, np.random.default_rng(5).standard_normal((1, DIM)).astype(np.float32),
                     query_rows, qid=["q0"])

    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(name="sparse_cos_a", vector_type="sparse", metric="cosine", k=3),
            SearchSpec(name="sparse_dot", vector_type="sparse", metric="dot", k=3),
            SearchSpec(name="sparse_cos_b", vector_type="sparse", metric="cosine", k=3),
        ],
    )
    paths = run_compute(cfg)

    dot = dict(zip(*[pq.read_table(paths["sparse_dot"]).to_pydict()[c][0] for c in ("hit_ids", "hit_scores")]))
    # dot(q0, c0) = 1*3+1*4 = 7; dot(q0, c1) = 1*1 = 1; dot(q0, c2) = 1*2 = 2
    assert dot["c0"] == pytest.approx(7.0, abs=1e-4)
    assert dot["c1"] == pytest.approx(1.0, abs=1e-4)
    assert dot["c2"] == pytest.approx(2.0, abs=1e-4)

    sqrt2 = np.sqrt(2.0)
    expected_cos = {"c0": 7.0 / (sqrt2 * 5.0), "c1": 1.0 / (sqrt2 * 1.0), "c2": 2.0 / (sqrt2 * 2.0)}
    # c1 and c2 share the identical cosine similarity (1/sqrt(2)) despite very
    # different dot products — proves the dot spec above isn't just silently
    # reusing (or half-reusing) a cosine member's normalized result.
    assert expected_cos["c1"] == pytest.approx(expected_cos["c2"], abs=1e-9)
    for name in ("sparse_cos_a", "sparse_cos_b"):
        cos = dict(zip(*[pq.read_table(paths[name]).to_pydict()[c][0] for c in ("hit_ids", "hit_scores")]))
        for cid, expected in expected_cos.items():
            assert cos[cid] == pytest.approx(expected, abs=1e-4), f"search={name} id={cid}"


def test_group_specs_batch_size_floors_at_group_max_k():
    """Unit test for `_merge_batch_size`/`_group_specs`: a group's resolved
    batch size must never fall below the largest `k` among its members, even
    when a different, lower-k member explicitly requests a smaller
    `corpus_batch_size` — mirrors the existing single-spec floor (a batch
    smaller than k can't fill the top-K and gives no memory benefit) at group
    granularity."""
    specs = [
        SearchSpec(name="big_k_wholefile", vector_type="dense", metric="dot", k=100),
        SearchSpec(name="small_k_tiny_batch", vector_type="dense", metric="dot", k=5, corpus_batch_size=10),
    ]
    # spec_batch mirrors what run_compute's setup loop resolves per spec BEFORE
    # grouping (each spec's own corpus_batch_size, already floored to its own
    # k) — neither is below its OWN k here, so this is each spec's
    # corpus_batch_size verbatim.
    spec_batch = [None, 10]
    groups = compute_mod._group_specs(specs, spec_batch)
    assert len(groups) == 1  # same vector_type, both unfiltered -> one group
    assert groups[0].batch_size == 100  # min(10) floored at max(k)=100, NOT 10


def test_group_batch_size_floor_end_to_end_still_correct(ds):
    """End-to-end companion to the unit test above: a member requesting a
    tiny `corpus_batch_size` sharing a group with a higher-`k` member must
    still produce correct, ground-truth-matching results for BOTH members —
    the floor changes only memory/perf, never correctness."""
    specs = [
        SearchSpec(name="low_k_tiny_batch", vector_type="dense", metric="dot", k=2, corpus_batch_size=1),
        SearchSpec(name="high_k", vector_type="dense", metric="dot", k=5),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "batch_floor_out")),
        searches=specs,
    )
    paths = run_compute(cfg)
    expectations = {
        "low_k_tiny_batch": ds["ground_truth"]("dense", 2),
        "high_k": ds["ground_truth"]("dense", 5),
    }
    for name, expected in expectations.items():
        t = pq.read_table(paths[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"


def test_sparse_dot_spec_not_corrupted_by_coscheduled_cosine_spec(tmp_path):
    """Regression test: `need_sparse_norms` (compute.py) gates whether a
    file's sparse rows get their L2 norm computed at all, but that's a
    RUN-WIDE decision (true if ANY spec in `searches` is sparse+cosine) — it
    must never leak into a co-scheduled sparse+dot spec's scoring. Two corpus
    rows share the exact same direction but different magnitude, so a dot
    spec accidentally receiving corpus-side cosine normalization collapses
    both rows to an identical (wrong) score instead of ranking them by their
    true, very different dot products — exactly the failure this guards."""
    rng = np.random.default_rng(3)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # same direction [1, 2] -> [v, v], different magnitude -> true dot products
    # 2.0 and 20.0 are nothing alike, but cosine-normalizing the corpus rows
    # collapses both to the identical value ~1.414 (query dotted with the same
    # unit vector either way).
    corpus_rows = [([1, 2], [1.0, 1.0]), ([1, 2], [10.0, 10.0])]
    dummy_dense = rng.standard_normal((2, DIM)).astype(np.float32)
    _write_combined(cdir / "f0.parquet", dummy_dense, corpus_rows, id=["c0", "c1"])

    query_rows = [([1, 2], [1.0, 1.0])]
    qpath = tmp_path / "queries.parquet"
    _write_combined(qpath, rng.standard_normal((1, DIM)).astype(np.float32), query_rows, qid=["q0"])

    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(name="sparse_dot", vector_type="sparse", metric="dot", k=2),
            SearchSpec(name="sparse_cos", vector_type="sparse", metric="cosine", k=2),
        ],
    )
    paths = run_compute(cfg)

    dot_scores = dict(zip(*[pq.read_table(paths["sparse_dot"]).to_pydict()[c][0] for c in ("hit_ids", "hit_scores")]))
    cos_scores = dict(zip(*[pq.read_table(paths["sparse_cos"]).to_pydict()[c][0] for c in ("hit_ids", "hit_scores")]))

    # true dot products: query=[1,1] on tokens [1,2] -> c0: 1*1+1*1=2.0, c1: 1*10+1*10=20.0
    assert dot_scores["c0"] == pytest.approx(2.0, abs=1e-4)
    assert dot_scores["c1"] == pytest.approx(20.0, abs=1e-4)
    # both corpus rows are the same unit direction as the query -> cosine ~1.0 for both
    assert cos_scores["c0"] == pytest.approx(1.0, abs=1e-4)
    assert cos_scores["c1"] == pytest.approx(1.0, abs=1e-4)


def test_mismatched_dense_and_sparse_query_loads_are_rejected(ds, monkeypatch):
    """Regression test for the query cross-check's failure path: if a run
    needs both vector_types and `load_queries`/`load_queries_sparse` ever
    disagree on query identity/order (not just count) for the same query
    store, `run_compute` must raise rather than silently misattributing one
    load's ids/payload to the other's vectors."""
    orig_load_sparse = compute_mod.load_queries_sparse

    def reordered_load_queries_sparse(store, qcfg):
        Q_np, vocab, q_ids, payload = orig_load_sparse(store, qcfg)
        assert len(q_ids) > 1  # sanity: reordering must actually change something
        return Q_np, vocab, list(reversed(q_ids)), payload

    monkeypatch.setattr(compute_mod, "load_queries_sparse", reordered_load_queries_sparse)

    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "mismatch_out")),
        searches=[
            SearchSpec(name="dense_all", vector_type="dense", metric="dot", k=3),
            SearchSpec(name="sparse_all", vector_type="sparse", metric="dot", k=3),
        ],
    )
    with pytest.raises(RuntimeError, match="don't match"):
        run_compute(cfg)


def test_shared_io_and_decode_once_per_file_not_per_spec(ds, monkeypatch):
    """2 dense specs + 1 sparse spec must still read each corpus file's
    columns ONCE and decode each vector_type ONCE per corpus file — not once
    per spec sharing that vector_type. `load_queries`/`load_queries_sparse`
    each make exactly one additional call apiece (there's only one query
    file), accounted for explicitly below."""
    calls = {"read_columns": 0, "dense_to_2d": 0, "sparse_to_coo_parts": 0}

    orig_read = Store.read_columns

    def counting_read(self, path, columns):
        calls["read_columns"] += 1
        return orig_read(self, path, columns)

    orig_dense = compute_mod.dense_to_2d

    def counting_dense(col):
        calls["dense_to_2d"] += 1
        return orig_dense(col)

    orig_sparse = compute_mod.sparse_to_coo_parts

    def counting_sparse(col):
        calls["sparse_to_coo_parts"] += 1
        return orig_sparse(col)

    monkeypatch.setattr(Store, "read_columns", counting_read)
    monkeypatch.setattr(compute_mod, "dense_to_2d", counting_dense)
    monkeypatch.setattr(compute_mod, "sparse_to_coo_parts", counting_sparse)

    specs = [
        SearchSpec(name="d1", vector_type="dense", metric="dot", k=2),
        SearchSpec(name="d2", vector_type="dense", metric="cosine", k=3),
        SearchSpec(name="s1", vector_type="sparse", metric="dot", k=2),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"]),
        queries=QueriesConfig(path=ds["qpath"]),
        output=OutputConfig(path=_out(ds, "shared_io_out")),
        searches=specs,
    )
    run_compute(cfg)

    n_files = len(SIZES)
    # +2: load_queries and load_queries_sparse each call read_columns once for
    # the single query file (both vector_types are needed here).
    assert calls["read_columns"] == n_files + 2
    # +1: load_queries's one call for the query file. The corpus side must be
    # exactly n_files (shared by BOTH dense specs), not n_files * 2.
    assert calls["dense_to_2d"] == n_files + 1
    assert calls["sparse_to_coo_parts"] == n_files + 1


def test_corpus_ids_resolved_even_when_a_different_specs_filter_drops_the_file(tmp_path):
    """A spec whose filter drops an ENTIRE file must not prevent a different,
    unfiltered spec that needs `corpus.id_column` from resolving ids for rows
    in that SAME file — corpus_ids[gidx] must be recorded unconditionally per
    file, not gated on any one spec's post-filter row count (compute.py)."""
    rng = np.random.default_rng(2)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n = 4
    dense_rows = rng.standard_normal((n, DIM)).astype(np.float32)
    sparse_rows = [_random_sparse_row(rng) for _ in range(n)]
    ids = [f"c{i}" for i in range(n)]
    lang = ["fra"] * n  # every row is "fra" — an "eng" filter drops the whole file
    _write_combined(cdir / "f0.parquet", dense_rows, sparse_rows, id=ids, language=lang)

    q_dense = rng.standard_normal((2, DIM)).astype(np.float32)
    q_sparse = [_random_sparse_row(rng) for _ in range(2)]
    qids = ["q0", "q1"]
    qpath = tmp_path / "queries.parquet"
    _write_combined(qpath, q_dense, q_sparse, qid=qids)

    specs = [
        SearchSpec(
            name="filtered_eng", vector_type="dense", metric="dot", k=2,
            filter=Filter(must=[FilterCondition(field="language", match="eng")]),
        ),
        SearchSpec(name="unfiltered", vector_type="dense", metric="dot", k=2),
    ]
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=specs,
    )
    paths = run_compute(cfg)

    filtered_t = pq.read_table(paths["filtered_eng"]).to_pydict()
    for hi in filtered_t["hit_ids"]:
        assert hi == []  # every row dropped by the filter

    unfiltered_t = pq.read_table(paths["unfiltered"]).to_pydict()
    for hi in unfiltered_t["hit_ids"]:
        assert len(hi) == 2  # k=2, both slots filled
        assert all(i in ids for i in hi)  # ids resolved from the SAME file, correctly


def test_sharded_compute_and_merge_multi_spec(ds):
    specs = [
        SearchSpec(name="dense_all", vector_type="dense", metric="dot", k=3),
        SearchSpec(name="sparse_all", vector_type="sparse", metric="dot", k=3),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "sharded_out")),
        searches=specs,
        params=ParamsConfig(io_workers=2),
    )
    num_jobs = 2
    for rank in range(num_jobs):
        run_compute(cfg, num_jobs=num_jobs, job_rank=rank)

    merged = run_merge(cfg)
    assert set(merged) == {"dense_all", "sparse_all"}

    expectations = {
        "dense_all": ds["ground_truth"]("dense", 3),
        "sparse_all": ds["ground_truth"]("sparse", 3),
    }
    for name, expected in expectations.items():
        t = pq.read_table(merged[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"
