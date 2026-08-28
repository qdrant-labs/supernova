"""Correctness tests for `SearchSpec.rows` — per-search QUERY-row subsets.

Without a subset every spec covers every row of the queries file, so a run
that unions several query sets into one file has to neutralize each spec's
foreign rows with a match-nothing sentinel (`zzznomatchzzz`). The foreign rows
are still allocated, scored and top-K'd; `rows` lets a spec decline to see
them.

The load-bearing test here is `test_rows_subset_equals_sentinel_pattern`: the
same searches run BOTH ways must produce identical hits for the rows each spec
owns. Everything else checks a specific mechanism (subset-height sparse
matrix, non-contiguous gather, per-query filter mask alignment, merge).

Two fixture shapes, and the difference matters:

  * `union_ds` — the query sets between them cover EVERY row, which is the
    shape of the real combined config. `_resolve_spec_rows` then finds the
    vector_type's row union to be the whole file and drops the subset for
    matrix-building purposes, so the query and score matrices stay full height
    and every spec's selector is a contiguous slice. Only the post-scoring
    slice is under test.
  * `interleaved_ds` — the sets interleave AND some rows belong to no search,
    so the union is a strict subset, the matrices really are shorter, and every
    selector is gapped. This is the only fixture that exercises the gather
    branch of `_row_selector` and the reduced-height loader path.

Reduced-height runs are compared with `pytest.approx`, not exact equality, and
that is deliberate: shortening the query matrix changes the scoring matmul's
shape and therefore its f32 accumulation order, so scores move by ~1 ULP. Hit
ids are compared exactly. See `RowSelector` and docs/brute-force/overview.md.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

import nova_bf.compute as compute_mod
from nova_bf.compute import (
    _local_positions,
    _resolve_spec_rows,
    _row_selector,
    load_queries_sparse,
    run_compute,
)
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.io import Store
from nova_bf.merge import run_merge

DIM, VOCAB, NNZ = 8, 12, 6
SENTINEL = "zzznomatchzzz"


def _sparse_row(rng, vocab=VOCAB, nnz=NNZ):
    idx = np.sort(rng.choice(vocab, size=nnz, replace=False))
    return idx.tolist(), rng.standard_normal(nnz).astype(np.float32).tolist()


def _write(path, dense, sparse_rows, **columns):
    data = {
        "dense_embedding": pa.array(np.asarray(dense).tolist(), type=pa.list_(pa.float32())),
        "sparse_embedding": pa.array(
            [{"indices": i, "values": v} for i, v in sparse_rows],
            type=pa.struct([
                pa.field("indices", pa.list_(pa.uint32())),
                pa.field("values", pa.list_(pa.float32())),
            ]),
        ),
    }
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture
def union_ds(tmp_path):
    """A queries file that UNIONS two query sets, exactly like the combined
    MS MARCO file: `query_set` says which spec owns a row, and each spec's
    per-query filter column carries a match-nothing sentinel on the rows it
    does not own (so the same file works with or without `rows`)."""
    rng = np.random.default_rng(7)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n_c = 10
    corpus_dense = rng.standard_normal((n_c, DIM)).astype(np.float32)
    corpus_sparse = [_sparse_row(rng) for _ in range(n_c)]
    tenants = ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"]
    cids = [f"c{i}" for i in range(n_c)]
    _write(cdir / "f0.parquet", corpus_dense, corpus_sparse, id=cids, tenant_id=tenants)

    # 6 queries: rows 0-2 are set "one", rows 3-5 are set "two".
    query_set = ["one", "one", "one", "two", "two", "two"]
    want_one = ["A", "B", "C", SENTINEL, SENTINEL, SENTINEL]
    want_two = [SENTINEL, SENTINEL, SENTINEL, "A", "B", "C"]
    n_q = len(query_set)
    q_dense = rng.standard_normal((n_q, DIM)).astype(np.float32)
    q_sparse = [_sparse_row(rng) for _ in range(n_q)]
    qpath = tmp_path / "queries.parquet"
    _write(
        qpath, q_dense, q_sparse,
        qid=[f"q{i}" for i in range(n_q)],
        query_set=query_set, want_one=want_one, want_two=want_two,
    )
    return {
        "cdir": str(cdir), "qpath": str(qpath), "tmp": tmp_path,
        "cids": cids, "tenants": tenants, "corpus_dense": corpus_dense,
        "q_dense": q_dense, "query_set": query_set,
        "qids": [f"q{i}" for i in range(n_q)],
    }


def _cfg(ds, name, searches):
    out = ds["tmp"] / name
    out.mkdir(exist_ok=True)
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid", payload_fields=["query_set"]),
        output=OutputConfig(path=str(out)),
        searches=searches,
    )


def _rows_of(path):
    t = pq.read_table(path).to_pydict()
    return {q: (h, s) for q, h, s in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}


# --------------------------------------------------------------------------
# the load-bearing one
# --------------------------------------------------------------------------
def test_rows_subset_equals_sentinel_pattern(union_ds):
    """`rows` must be a pure optimization: for the rows a spec owns, the hits
    are identical to what the sentinel pattern produces today."""
    ds = union_ds
    f_one = Filter(must=[FilterCondition(field="tenant_id", match_from_query="want_one")])
    f_two = Filter(must=[FilterCondition(field="tenant_id", match_from_query="want_two")])

    sentinel_paths = run_compute(_cfg(ds, "sentinel_out", [
        SearchSpec(name="one", vector_type="dense", metric="dot", k=5, filter=f_one),
        SearchSpec(name="two", vector_type="dense", metric="dot", k=5, filter=f_two),
    ]))
    subset_paths = run_compute(_cfg(ds, "subset_out", [
        SearchSpec(name="one", vector_type="dense", metric="dot", k=5, filter=f_one,
                   rows={"column": "query_set", "isin": ["one"]}),
        SearchSpec(name="two", vector_type="dense", metric="dot", k=5, filter=f_two,
                   rows={"column": "query_set", "isin": ["two"]}),
    ]))

    for name, owned in (("one", "one"), ("two", "two")):
        sent = _rows_of(sentinel_paths[name])
        sub = _rows_of(subset_paths[name])
        want = {q for q, qs in zip(ds["qids"], ds["query_set"]) if qs == owned}
        assert set(sub) == want, f"{name}: subset run must cover exactly its own rows"
        # the sentinel run really did neutralize the foreign rows
        assert all(not sent[q][0] for q in set(sent) - want), f"{name}: sentinel leaked hits"
        assert any(sent[q][0] for q in want), f"{name}: fixture produced no hits at all"
        for q in want:
            assert sub[q][0] == sent[q][0], f"{name}/{q} hit_ids differ"
            assert sub[q][1] == pytest.approx(sent[q][1]), f"{name}/{q} hit_scores differ"


# --------------------------------------------------------------------------
# mechanisms
# --------------------------------------------------------------------------
def test_rows_subset_matches_independent_ground_truth(union_ds):
    """Unfiltered specs, subset only — checked against plain numpy."""
    ds = union_ds
    paths = run_compute(_cfg(ds, "gt_out", [
        SearchSpec(name="one", vector_type="dense", metric="dot", k=4,
                   rows={"column": "query_set", "isin": ["one"]}),
    ]))
    got = _rows_of(paths["one"])
    for qi, (qid, qs) in enumerate(zip(ds["qids"], ds["query_set"])):
        if qs != "one":
            assert qid not in got
            continue
        scores = ds["q_dense"][qi] @ ds["corpus_dense"].T
        expected = [ds["cids"][j] for j in np.argsort(-scores)[:4]]
        assert got[qid][0] == expected, f"query={qid}"


def test_non_contiguous_file_rows_still_read_the_right_queries(union_ds):
    """Rows 0,2,4 are not a contiguous run of FILE rows. This spec is alone, so
    its vector_type's matrix is exactly its own 3 rows and `spec_qsel` is still
    the contiguous slice(0, 3) — what this pins is that the file-row -> matrix-
    row mapping (`_local_positions`) is applied at all.

    The genuinely non-contiguous SCORE-MATRIX selector needs two specs whose
    rows interleave; see `test_interleaved_subsets_gather_the_score_matrix`.
    """
    ds = union_ds
    qpath = ds["tmp"] / "queries.parquet"
    t = pq.read_table(qpath).to_pydict()
    t["query_set"] = ["pick" if i % 2 == 0 else "skip" for i in range(len(t["qid"]))]
    pq.write_table(pa.table({k: pa.array(v, type=pq.read_table(qpath).schema.field(k).type)
                             for k, v in t.items()}), str(qpath))

    paths = run_compute(_cfg(ds, "gather_out", [
        SearchSpec(name="pick", vector_type="dense", metric="dot", k=3,
                   rows={"column": "query_set", "isin": ["pick"]}),
    ]))
    got = _rows_of(paths["pick"])
    assert set(got) == {"q0", "q2", "q4"}
    for qid in got:
        qi = int(qid[1:])
        scores = ds["q_dense"][qi] @ ds["corpus_dense"].T
        assert got[qid][0] == [ds["cids"][j] for j in np.argsort(-scores)[:3]]


def test_sparse_query_matrix_is_built_at_subset_height(union_ds):
    """The point of `rows` on the sparse path: the dense (n_q, vocab) query
    matrix must not carry rows no sparse search owns."""
    ds = union_ds
    store = Store(ds["qpath"])
    qcfg = QueriesConfig(path=ds["qpath"], id_column="qid")
    full_Q, full_vocab, ids, _, _ = load_queries_sparse(store, qcfg)
    sub_Q, sub_vocab, sub_ids, _, _ = load_queries_sparse(
        store, qcfg, rows=np.array([3, 4, 5], dtype=np.int64)
    )
    assert full_Q.shape[0] == 6
    assert sub_Q.shape[0] == 3, "subset must shrink the query matrix height"
    assert len(sub_vocab) <= len(full_vocab), "subset vocab is a subset"
    # ids/payload stay FULL length either way — run_compute's cross-vector_type
    # id check compares them, and per-query filter masks span the full axis.
    assert sub_ids == ids


def test_subset_rows_survive_the_merge(union_ds):
    """Sharded compute + merge must preserve the subset, not resurrect the
    file's other rows."""
    ds = union_ds
    cfg = _cfg(ds, "merge_out", [
        SearchSpec(name="one", vector_type="dense", metric="dot", k=4,
                   rows={"column": "query_set", "isin": ["one"]}),
        SearchSpec(name="two", vector_type="dense", metric="dot", k=4,
                   rows={"column": "query_set", "isin": ["two"]}),
    ])
    for rank in range(2):
        run_compute(cfg, num_jobs=2, job_rank=rank)
    merged = run_merge(cfg)
    assert set(_rows_of(merged["one"])) == {"q0", "q1", "q2"}
    assert set(_rows_of(merged["two"])) == {"q3", "q4", "q5"}


def test_mixed_subset_and_full_specs_share_one_matrix(union_ds):
    """A spec with no `rows` forces its vector_type's matrix to full height;
    a subset spec alongside it must still read the right rows out of it."""
    ds = union_ds
    paths = run_compute(_cfg(ds, "mixed_out", [
        SearchSpec(name="all", vector_type="dense", metric="dot", k=3),
        SearchSpec(name="two", vector_type="dense", metric="dot", k=3,
                   rows={"column": "query_set", "isin": ["two"]}),
    ]))
    every = _rows_of(paths["all"])
    subset = _rows_of(paths["two"])
    assert set(every) == set(ds["qids"])
    assert set(subset) == {"q3", "q4", "q5"}
    for qid in subset:
        assert subset[qid][0] == every[qid][0], f"{qid}: subset spec read the wrong row"


def test_payload_is_sliced_to_the_subset(union_ds):
    ds = union_ds
    paths = run_compute(_cfg(ds, "payload_out", [
        SearchSpec(name="two", vector_type="dense", metric="dot", k=2,
                   rows={"column": "query_set", "isin": ["two"]}),
    ]))
    t = pq.read_table(paths["two"]).to_pydict()
    assert t["query_id"] == ["q3", "q4", "q5"]
    assert t["query_set"] == ["two", "two", "two"], "payload must follow the subset"


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------
def test_selector_matching_nothing_is_an_error(union_ds):
    ds = union_ds
    with pytest.raises(ValueError, match="no row matching"):
        run_compute(_cfg(ds, "empty_out", [
            SearchSpec(name="nope", vector_type="dense", metric="dot", k=2,
                       rows={"column": "query_set", "isin": ["does_not_exist"]}),
        ]))


def test_unknown_selector_column_is_rejected(union_ds):
    ds = union_ds
    with pytest.raises(Exception) as e:
        run_compute(_cfg(ds, "badcol_out", [
            SearchSpec(name="nope", vector_type="dense", metric="dot", k=2,
                       rows={"column": "not_a_column", "isin": ["one"]}),
        ]))
    assert "not_a_column" in str(e.value)


# --------------------------------------------------------------------------
# the CPU/packed per-query filter path
# --------------------------------------------------------------------------
def test_rows_subset_with_text_filter_equals_sentinel_pattern(tmp_path):
    """`match_text_from_query` is NOT gpu-eligible (torch has no string tensor
    type), so its per-query mask takes the packed `keeps` path and is unpacked
    over the FULL query axis — a different height from this vector_type's
    query matrix once a subset shrinks it. That makes this the branch the
    `n_q_full` plumbing exists for, and the one `match_from_query` tests do
    NOT reach (they go through `_gpu_evaluate` instead).

    It is also the real filtered_text config: `field: text,
    match_text_from_query: keyword_phrase`.
    """
    rng = np.random.default_rng(3)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    texts = [
        "physics of motion", "dna replication basics", "physics and energy",
        "cooking with dna? no", "quantum physics primer", "bread baking guide",
    ]
    n_c = len(texts)
    _write(
        cdir / "f0.parquet",
        rng.standard_normal((n_c, DIM)).astype(np.float32),
        [_sparse_row(rng) for _ in range(n_c)],
        id=[f"c{i}" for i in range(n_c)], text=texts,
    )

    # rows 2-3 own the text search; rows 0-1 are foreign (sentinel = "").
    # The owned rows are deliberately NOT a prefix: if the per-query mask were
    # unpacked at this spec's subset height (2) instead of the full query axis
    # (4), indexing it by FILE row 2/3 would run off the end. A prefix subset
    # would silently paper over that.
    query_set = ["other", "other", "kw", "kw"]
    keyword = ["", "", "physics", "dna"]
    n_q = len(query_set)
    q_dense = rng.standard_normal((n_q, DIM)).astype(np.float32)
    qpath = tmp_path / "queries.parquet"
    _write(
        qpath, q_dense, [_sparse_row(rng) for _ in range(n_q)],
        qid=[f"q{i}" for i in range(n_q)],
        query_set=query_set, keyword_phrase=keyword,
    )

    text_filter = Filter(must=[
        FilterCondition(field="text", match_text_from_query="keyword_phrase")
    ])

    def run(name, rows):
        out = tmp_path / name
        out.mkdir()
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            searches=[SearchSpec(name="kw", vector_type="dense", metric="dot",
                                 k=5, filter=text_filter, rows=rows)],
        )
        return _rows_of(run_compute(cfg)["kw"])

    sent = run("text_sentinel", None)
    sub = run("text_subset", {"column": "query_set", "isin": ["kw"]})

    assert set(sub) == {"q2", "q3"}
    assert not sent["q0"][0] and not sent["q1"][0], "empty keyword must match nothing"
    assert sent["q2"][0] and sent["q3"][0], "fixture produced no hits to compare"
    for q in ("q2", "q3"):
        assert sub[q][0] == sent[q][0], f"{q}: subset changed hit_ids on the packed path"
        assert sub[q][1] == pytest.approx(sent[q][1]), f"{q}: subset changed hit_scores"

    # and the hits really are keyword-restricted
    assert all("physics" in texts[int(c[1:])] for c in sub["q2"][0])
    assert all("dna" in texts[int(c[1:])] for c in sub["q3"][0])


# --------------------------------------------------------------------------
# the gather selectors, and the reduced-height matrix
# --------------------------------------------------------------------------
def test_row_selector_contiguity_guard():
    """`_row_selector` may only collapse an index array to a `slice` when that
    array is one gap-free ascending run. A `slice(idx[0], idx[-1] + 1)` built
    from a gapped array silently addresses rows the spec does not own — and
    every subset in a queries file sorted by query set IS contiguous, so no
    end-to-end fixture built that way can tell the two apart. Hence a direct
    unit test on the guard.
    """
    import torch

    assert _row_selector(None, "cpu") is None
    # contiguous -> slice (a view, not a gather)
    assert _row_selector(np.array([0, 1, 2]), "cpu") == slice(0, 3)
    assert _row_selector(np.array([3, 4, 5]), "cpu") == slice(3, 6)
    assert _row_selector(np.array([7]), "cpu") == slice(7, 8)
    # gapped -> index tensor, NOT slice(first, last + 1)
    for idx in ([0, 2, 4], [1, 2, 6], [0, 1, 3]):
        sel = _row_selector(np.array(idx, dtype=np.int64), "cpu")
        assert isinstance(sel, torch.Tensor), f"{idx} must not collapse to a slice"
        assert sel.tolist() == idx
        assert sel.dtype == torch.int64


@pytest.fixture
def interleaved_ds(tmp_path):
    """Two query sets INTERLEAVED in the file, plus rows neither search owns.

    This is the shape that produces the selectors `union_ds` cannot:
      * rows no search owns  -> the vector_type's row union is a STRICT subset
        of the file, so the query matrix is genuinely shorter (6 of 8 rows)
      * interleaved sets     -> each spec's positions WITHIN that matrix are
        gapped, so `spec_qsel` is an index tensor, not a slice
      * gapped file rows     -> `spec_qrows` is an index tensor too, and it is
        actually USED here because both specs carry a per-query filter
    """
    rng = np.random.default_rng(19)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n_c = 24
    corpus_dense = rng.standard_normal((n_c, DIM)).astype(np.float32)
    corpus_sparse = [_sparse_row(rng) for _ in range(n_c)]
    tenants = [["A", "B", "C"][i % 3] for i in range(n_c)]
    texts = [f"alpha doc{i} " + ("charlie" if i % 2 else "delta") for i in range(n_c)]
    cids = [f"c{i}" for i in range(n_c)]
    _write(cdir / "f0.parquet", corpus_dense, corpus_sparse,
           id=cids, tenant_id=tenants, text=texts)

    #        row:   0       1      2      3       4      5      6      7
    query_set = ["none", "one", "one", "none", "two", "two", "one", "two"]
    want_one = [SENTINEL, "A", "B", SENTINEL, SENTINEL, SENTINEL, "C", SENTINEL]
    want_two = [SENTINEL, SENTINEL, SENTINEL, SENTINEL, "C", "A", SENTINEL, "B"]
    phrase = [SENTINEL, "charlie", "delta", SENTINEL, SENTINEL, SENTINEL, "charlie", SENTINEL]
    n_q = len(query_set)
    q_dense = rng.standard_normal((n_q, DIM)).astype(np.float32)
    qpath = tmp_path / "queries.parquet"
    _write(qpath, q_dense, [_sparse_row(rng) for _ in range(n_q)],
           qid=[f"q{i}" for i in range(n_q)], query_set=query_set,
           want_one=want_one, want_two=want_two, phrase=phrase)
    return {
        "cdir": str(cdir), "qpath": str(qpath), "tmp": tmp_path,
        "cids": cids, "corpus_dense": corpus_dense, "q_dense": q_dense,
        "query_set": query_set, "qids": [f"q{i}" for i in range(n_q)],
    }


def test_interleaved_fixture_really_produces_gapped_selectors(interleaved_ds):
    """Guard on the fixture itself: if a later edit sorted these rows by query
    set, every selector below would quietly become a contiguous slice and the
    two tests after this one would stop testing the gather path at all.
    """
    specs = [
        SearchSpec(name="one", vector_type="dense", metric="dot", k=4,
                   rows={"column": "query_set", "isin": ["one"]}),
        SearchSpec(name="two", vector_type="dense", metric="dot", k=4,
                   rows={"column": "query_set", "isin": ["two"]}),
    ]
    vals = {"query_set": np.array(interleaved_ds["query_set"])}
    spec_rows, vt_rows = _resolve_spec_rows(specs, vals, len(vals["query_set"]))

    assert spec_rows[0].tolist() == [1, 2, 6]
    assert spec_rows[1].tolist() == [4, 5, 7]
    # rows 0 and 3 belong to no search -> the union is a strict subset, so the
    # query matrix really is built shorter than the file (6 of 8).
    assert vt_rows["dense"].tolist() == [1, 2, 4, 5, 6, 7]

    # ...and each spec's positions within that 6-row matrix are gapped.
    for m in (0, 1):
        local = _local_positions(spec_rows[m], vt_rows["dense"])
        assert local[-1] - local[0] != len(local) - 1, "spec_qsel must be a gather"
        assert spec_rows[m][-1] - spec_rows[m][0] != len(spec_rows[m]) - 1, (
            "spec_qrows must be a gather"
        )
    assert _local_positions(spec_rows[0], vt_rows["dense"]).tolist() == [0, 1, 4]
    assert _local_positions(spec_rows[1], vt_rows["dense"]).tolist() == [2, 3, 5]


def test_interleaved_subsets_gather_the_score_matrix(interleaved_ds):
    """End-to-end over the gather selectors both fronts: `spec_qsel` gathers the
    SCORE matrix's query axis and `spec_qrows` gathers a FULL-axis per-query
    filter mask, in the same run, with gaps in both. `match_from_query` is
    gpu-eligible so the mask comes from `_gpu_evaluate`; the packed CPU path is
    covered by the `phrase` spec in the next test.
    """
    ds = interleaved_ds
    f_one = Filter(must=[FilterCondition(field="tenant_id", match_from_query="want_one")])
    f_two = Filter(must=[FilterCondition(field="tenant_id", match_from_query="want_two")])
    mk = lambda rows_one, rows_two: [
        SearchSpec(name="one", vector_type="dense", metric="dot", k=4,
                   filter=f_one, rows=rows_one),
        SearchSpec(name="two", vector_type="dense", metric="dot", k=4,
                   filter=f_two, rows=rows_two),
    ]
    sent = run_compute(_cfg(ds, "il_sentinel", mk(None, None)))
    sub = run_compute(_cfg(ds, "il_subset",
                           mk({"column": "query_set", "isin": ["one"]},
                              {"column": "query_set", "isin": ["two"]})))

    for name, want in (("one", {"q1", "q2", "q6"}), ("two", {"q4", "q5", "q7"})):
        s, b = _rows_of(sub[name]), _rows_of(sent[name])
        assert set(s) == want
        assert any(s[q][0] for q in want), f"{name}: fixture produced no hits"
        for q in want:
            assert s[q][0] == b[q][0], f"{name}/{q}: gather read the wrong query row"
            # scores are NOT bit-exact here on purpose: this run's query matrix
            # is 6 rows where the sentinel run's is 8, which changes the
            # matmul's shape and so its f32 accumulation order. See
            # RowSelector's docstring and docs/brute-force/overview.md.
            assert s[q][1] == pytest.approx(b[q][1], rel=1e-5), f"{name}/{q}"


def test_reduced_height_matrix_with_packed_and_gpu_masks(interleaved_ds):
    """A shortened query matrix (6 of 8 rows) driving BOTH per-query mask
    fronts at once: `match_text_from_query` unpacked from the packed FULL-axis
    `keeps` (this is what `n_q_full` exists for) and a gpu-eligible
    `match_from_query`, each spec gathering its own gapped rows out of the one
    shared score matrix.
    """
    ds = interleaved_ds
    f_text = Filter(must=[FilterCondition(field="text", match_text_from_query="phrase")])
    f_gpu = Filter(must=[FilterCondition(field="tenant_id", match_from_query="want_two")])
    mk = lambda r1, r2: [
        SearchSpec(name="tx", vector_type="dense", metric="cosine", k=4,
                   filter=f_text, rows=r1),
        SearchSpec(name="gp", vector_type="dense", metric="cosine", k=4,
                   filter=f_gpu, rows=r2),
    ]
    sent = run_compute(_cfg(ds, "rh_sentinel", mk(None, None)))
    sub = run_compute(_cfg(ds, "rh_subset",
                           mk({"column": "query_set", "isin": ["one"]},
                              {"column": "query_set", "isin": ["two"]})))
    for name, want in (("tx", {"q1", "q2", "q6"}), ("gp", {"q4", "q5", "q7"})):
        s, b = _rows_of(sub[name]), _rows_of(sent[name])
        assert set(s) == want
        assert any(s[q][0] for q in want), f"{name}: fixture produced no hits"
        for q in want:
            assert s[q][0] == b[q][0], f"{name}/{q}: wrong query row"
            assert s[q][1] == pytest.approx(b[q][1], rel=1e-5), f"{name}/{q}"


def test_sparse_subset_with_per_query_filter(interleaved_ds):
    """Sparse + `rows` + a per-query filter together: the sparse loader
    compacts BEFORE the vocab build, so this spec's query matrix is both
    shorter AND narrower than the file's — while the filter mask it gets
    masked with is still full height.
    """
    ds = interleaved_ds
    f_one = Filter(must=[FilterCondition(field="tenant_id", match_from_query="want_one")])
    mk = lambda rows: [SearchSpec(name="s1", vector_type="sparse", metric="dot",
                                  k=4, filter=f_one, rows=rows)]
    sent = run_compute(_cfg(ds, "sp_sentinel", mk(None)))
    sub = run_compute(_cfg(ds, "sp_subset", mk({"column": "query_set", "isin": ["one"]})))
    s, b = _rows_of(sub["s1"]), _rows_of(sent["s1"])
    assert set(s) == {"q1", "q2", "q6"}
    assert any(s[q][0] for q in s), "fixture produced no hits"
    for q in s:
        assert s[q][0] == b[q][0], f"{q}: sparse subset read the wrong query row"
        assert s[q][1] == pytest.approx(b[q][1], rel=1e-5), f"{q}"


def test_misspelled_rows_column_names_rows_not_a_filter(union_ds):
    """A typo'd `rows.column` rides the same validation as the per-query filter
    columns, so the error has to say which config key to go fix — pointing the
    reader at `match_from_query` sends them to the wrong place entirely.
    """
    ds = union_ds
    with pytest.raises(ValueError) as e:
        run_compute(_cfg(ds, "rowtypo_out", [
            SearchSpec(name="nope", vector_type="dense", metric="dot", k=2,
                       rows={"column": "query_sett", "isin": ["one"]}),
        ]))
    msg = str(e.value)
    assert "query_sett" in msg
    assert "rows.column" in msg
    assert "match_from_query" not in msg, "must not blame a per-query filter"


def test_row_count_drift_between_the_two_query_reads_is_fatal(union_ds, monkeypatch):
    """`rows` selectors are resolved from a SEPARATE, earlier pass over the
    queries store than the one that loads the vectors. The indices from that
    pass then address `Q`, `query_ids` and `payload` by file row, so the two
    reads agreeing on the row set is load-bearing. They do agree today (same
    store, same `list_parquets()` order, no truncation on either path) — this
    pins the guard so a future loader that filters or truncates rows fails
    loudly instead of subsetting the wrong queries.
    """
    ds = union_ds
    real = compute_mod._load_query_columns

    def short(store, qcfg, cols):
        out = real(store, qcfg, cols)
        return {c: v[:-1] for c, v in out.items()}  # one row shorter

    monkeypatch.setattr(compute_mod, "_load_query_columns", short)
    with pytest.raises(RuntimeError, match="row set changed between the two reads"):
        run_compute(_cfg(ds, "drift_out", [
            SearchSpec(name="one", vector_type="dense", metric="dot", k=2,
                       rows={"column": "query_set", "isin": ["one"]}),
        ]))
