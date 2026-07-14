"""Correctness tests for `BruteForceConfig.searches` — running several
independent top-K searches (dense/sparse x filtered/unfiltered) against the
SAME corpus in one `run_compute`/`run_merge` call, sharing corpus file IO and
per-vector-type decode across specs. Per vector_type, every search sharing
that vector_type shares ONE GPU batch grid via `_process_shared_batch` (see
compute.py's module docstring):

- If ANY search of that vector_type is unfiltered, the shared grid is the
  whole file, uncompacted — a filtered search's own `metric` never needs to
  match anyone else's, since scoring one more metric on an already-resident
  batch is cheap.
- Otherwise, the shared grid is the UNION of every distinct active filter's
  surviving rows (`_union_keep`), compacted/transferred/scored once, with
  each search then masking down further to its own filter's subset.

Only each search's own scoring and top-K stay independent either way.

Ground truth for each spec is computed independently in plain numpy (not by
re-deriving nova_bf's own scoring), mirroring test_compute.py/
test_compute_sparse.py's pattern.
"""

from __future__ import annotations

import logging

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
    RangeFromQuery,
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
        SearchSpec(name="sparse_all", vector_type="sparse", metric="dot", k=3),
        SearchSpec(name="sparse_eng", vector_type="sparse", metric="dot", k=2, filter=eng_filter),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "multi_out")),
        params=ParamsConfig(sparse_batch_size=2),
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
    """The core regression guard for shared-batch processing: running several
    searches together must produce results BIT-IDENTICAL to running each of
    those same searches alone, one per `run_compute` call. Covers dense+
    sparse, all three dense metrics (cosine/dot/euclidean) sharing the SAME
    unfiltered pass (see `_scores`), and a filtered dense search (`dense_eng`)
    riding that same pass alongside them."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        # dense: 3 unfiltered members (all 3 metrics) -> whole-file baseline
        SearchSpec(name="dense_dot", vector_type="dense", metric="dot", k=3),
        SearchSpec(name="dense_cos", vector_type="dense", metric="cosine", k=2),
        SearchSpec(name="dense_euclid", vector_type="dense", metric="euclidean", k=2),
        # dense: filtered, but an unfiltered dense sibling exists above -> also
        # rides the shared batch (masked down to its own filter), not
        # processed alone
        SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=2, filter=eng_filter),
        # sparse: 2 unfiltered members, mixed metrics — shared again (the
        # shared-metric-cache path; see test_three_member_group_mixed_metrics
        # for a 3-member version with independently-verified ground truth)
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
    """One shared batch (vector_type=sparse, all three unfiltered) with
    THREE members: two `cosine`, one `dot` — directly targets the two riskiest
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


def test_resolve_vt_batch_size_never_raises():
    """Unit test for `_resolve_vt_batch_size` — `k_floor` spans EVERY search
    of a vector_type, since they all now share one batch grid regardless of
    filter (see `_union_keep`). Raising a configured value here would let one
    search's large `k` silently blow past a DIFFERENT, unrelated search's own
    memory bound, so it's never overridden — only warned about — even when
    it's below `k_floor`. (Unlike the old per-identical-filter-group
    behavior this replaced: that raised freely, since only THAT group's own
    members shared the grid back then — see
    test_identical_filter_specs_no_longer_raise_batch_size.)"""
    assert compute_mod._resolve_vt_batch_size(10, k_floor=100, vt="dense") == 10
    assert compute_mod._resolve_vt_batch_size(500, k_floor=100, vt="dense") == 500
    assert compute_mod._resolve_vt_batch_size(None, k_floor=100, vt="dense") is None


def test_union_keep_ors_every_distinct_filter():
    """Unit test for `_union_keep`: the shared row-set for a vector_type with
    no unfiltered search is the OR of every distinct filter's own mask, not
    just one of them — this is what lets fully disjoint filters still share
    ONE transfer/score pass instead of one each."""
    keeps = {
        "a": np.array([True, False, False, True]),
        "b": np.array([False, True, False, False]),
    }
    union = compute_mod._union_keep(["a", "b"], keeps)
    assert union.tolist() == [True, True, False, True]
    # a single filter's union is just its own mask, copied (not aliased —
    # mutating the result must never corrupt `keeps`).
    solo = compute_mod._union_keep(["a"], keeps)
    solo[0] = False
    assert keeps["a"][0]


def test_no_baseline_distinct_filters_match_independent_ground_truth(ds):
    """The core regression guard for the union-of-filters mechanism
    (`_union_keep`): when NO search of a vector_type is unfiltered, but two
    searches have genuinely DIFFERENT (here, disjoint — eng vs fra) filters,
    each must still get its own correct top-K — the shared union batch must
    never let one search's filter leak into another's results. `lang_by_g`
    alternates eng/fra across the WHOLE corpus, so the union here covers
    every row, the same as an unfiltered baseline would — a good edge case
    for the union path specifically."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    fra_filter = Filter(must=[FilterCondition(field="language", match="fra")])
    specs = [
        SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=3, filter=eng_filter),
        SearchSpec(name="dense_fra", vector_type="dense", metric="dot", k=3, filter=fra_filter),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "no_baseline_union_out")),
        searches=specs,
    )
    paths = run_compute(cfg)

    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    fra_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "fra"]
    expectations = {
        "dense_eng": ds["ground_truth"]("dense", 3, allowed=eng_globals),
        "dense_fra": ds["ground_truth"]("dense", 3, allowed=fra_globals),
    }
    for name, expected in expectations.items():
        t = pq.read_table(paths[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"


def test_no_baseline_distinct_filters_match_solo_runs_bit_identical(ds):
    """Companion to `test_grouped_matches_ungrouped_per_search`, specifically
    for the no-baseline union path: two searches with different (eng/fra)
    filters and no unfiltered sibling must produce results BIT-IDENTICAL to
    running each alone — each solo run's own union is just its single
    filter, so this also proves the union path degenerates correctly to the
    single-filter case."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    fra_filter = Filter(must=[FilterCondition(field="language", match="fra")])
    specs = [
        SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=3, filter=eng_filter),
        SearchSpec(name="dense_fra", vector_type="dense", metric="dot", k=3, filter=fra_filter),
    ]
    combined_cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "no_baseline_union_combined")),
        searches=specs,
    )
    combined_paths = run_compute(combined_cfg)

    for spec in specs:
        solo_cfg = BruteForceConfig(
            corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
            queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
            output=OutputConfig(path=_out(ds, f"no_baseline_union_solo_{spec.name}")),
            searches=[spec],
        )
        solo_path = run_compute(solo_cfg)[spec.name]

        combined_t = pq.read_table(combined_paths[spec.name]).to_pydict()
        solo_t = pq.read_table(solo_path).to_pydict()
        assert combined_t["hit_ids"] == solo_t["hit_ids"], f"search={spec.name}"
        for combined_scores, solo_scores in zip(combined_t["hit_scores"], solo_t["hit_scores"]):
            assert np.allclose(combined_scores, solo_scores, atol=1e-5), f"search={spec.name}"


def test_no_baseline_log_reports_union_of_distinct_filters(ds, caplog):
    """Regression test for the log wording itself: with no unfiltered search
    sharing a vector_type, the log must report the union path (distinct
    filter count), not the old per-group wording."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    fra_filter = Filter(must=[FilterCondition(field="language", match="fra")])
    specs = [
        SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=3, filter=eng_filter),
        SearchSpec(name="dense_fra", vector_type="dense", metric="dot", k=3, filter=fra_filter),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "no_baseline_union_log_out")),
        searches=specs,
    )
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        run_compute(cfg)
    assert any(
        "no unfiltered search" in r.message and "union of 2 distinct filter(s)" in r.message
        for r in caplog.records
    )


def test_identical_filter_specs_no_longer_raise_batch_size(ds, caplog):
    """Deliberate behavior change from the old per-identical-filter-group
    path: specs sharing an IDENTICAL filter used to have their batch size
    raised to their own group's k_floor (safe when only THAT group shared
    the grid). Now every search of the vector_type shares ONE grid regardless
    of filter — even when they're all the same filter — so raising is no
    longer safe in general and `_resolve_vt_batch_size` never does it; a
    small configured batch just costs the larger-k search extra merge
    rounds instead of a wrong answer."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        SearchSpec(name="eng_smallk", vector_type="dense", metric="dot", k=2, filter=eng_filter),
        SearchSpec(name="eng_bigk", vector_type="dense", metric="dot", k=1000, filter=eng_filter),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "identical_filter_no_raise_out")),
        params=ParamsConfig(dense_batch_size=1),
        searches=specs,
    )
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        paths = run_compute(cfg)
    assert any("batch_size=1)" in r.message for r in caplog.records), \
        "batch size must stay at the configured 1, not be raised to k=1000 (old Path B behavior)"
    assert not any("raising to k" in r.message for r in caplog.records)

    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    expectations = {
        "eng_smallk": ds["ground_truth"]("dense", 2, allowed=eng_globals),
        "eng_bigk": ds["ground_truth"]("dense", 1000, allowed=eng_globals),
    }
    for name, expected in expectations.items():
        t = pq.read_table(paths[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"


def test_batch_size_floor_end_to_end_still_correct(ds):
    """End-to-end companion to the unit test above: a tiny `params.
    dense_batch_size` shared by a low-k and a high-k search (both unfiltered,
    so the configured value is kept rather than raised — see
    test_resolve_vt_batch_size_never_raises) must still produce correct,
    ground-truth-matching results for BOTH — an under-filled batch means
    more merge rounds, never a wrong answer."""
    specs = [
        SearchSpec(name="low_k", vector_type="dense", metric="dot", k=2),
        SearchSpec(name="high_k", vector_type="dense", metric="dot", k=5),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "batch_floor_out")),
        params=ParamsConfig(dense_batch_size=1),  # NOT raised to 5 — kept at 1
        searches=specs,
    )
    paths = run_compute(cfg)
    expectations = {
        "low_k": ds["ground_truth"]("dense", 2),
        "high_k": ds["ground_truth"]("dense", 5),
    }
    for name, expected in expectations.items():
        t = pq.read_table(paths[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"


def test_unrelated_large_k_filtered_spec_does_not_inflate_shared_batch(ds, caplog):
    """Regression test: an unfiltered spec's own configured `dense_batch_size`
    must not be silently widened just because a DIFFERENT, filtered spec
    sharing the same vector_type has a much larger `k` — that would defeat
    the memory bound the batch size exists to enforce. Both specs must still
    produce correct results; only the log should note the larger-k search
    needs extra merge rounds, not raise the resolved batch."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        SearchSpec(name="dense_all_smallk", vector_type="dense", metric="dot", k=2),
        SearchSpec(name="dense_eng_bigk", vector_type="dense", metric="dot", k=1000, filter=eng_filter),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "no_inflate_out")),
        params=ParamsConfig(dense_batch_size=1),
        searches=specs,
    )
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        paths = run_compute(cfg)
    assert any("batch_size=1)" in r.message for r in caplog.records), "batch size must stay at the configured 1, not be raised to k=1000"
    assert not any("raising to k" in r.message for r in caplog.records), "the shared batch grid must never be force-raised for one unrelated search's large k"

    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    expectations = {
        "dense_all_smallk": ds["ground_truth"]("dense", 2),
        "dense_eng_bigk": ds["ground_truth"]("dense", 1000, allowed=eng_globals),
    }
    for name, expected in expectations.items():
        t = pq.read_table(paths[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"


def test_explicit_empty_filter_still_shares_full_file_batch(ds, caplog):
    """Regression test: an explicit-but-empty `filter: {}` (`Filter()`) is
    semantically identical to no filter at all — `evaluate()` keeps every
    row either way — so a vector_type where every spec sets one must still
    take the zero-copy, whole-file shared batch grid, not the union-of-
    filters compaction path (which `has_baseline`'s `is None` check alone
    would have mistakenly ruled out, since `Filter()` is not `None`)."""
    specs = [
        SearchSpec(name="dense_a", vector_type="dense", metric="dot", k=3, filter=Filter()),
        SearchSpec(name="dense_b", vector_type="dense", metric="dot", k=3, filter=Filter()),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "empty_filter_out")),
        searches=specs,
    )
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        paths = run_compute(cfg)
    assert any("share one full-file batch pass" in r.message for r in caplog.records), \
        "an explicit-but-empty filter must still route to the whole-file shared batch, not union compaction"
    assert not any("no unfiltered search" in r.message for r in caplog.records)

    expected = ds["ground_truth"]("dense", 3)
    for name in ("dense_a", "dense_b"):
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


def test_filtered_spec_metric_not_used_by_baseline_still_shares_batch(ds):
    """Locks in the design decision that dropped the old "metrics must be a
    subset of the baseline's" precondition: `dense_eng` here is `cosine`, but
    the only unfiltered dense sibling (`dense_all`) is `dot` — under the old
    SpecGroup/borrower model this metric mismatch would have forced
    `dense_eng` into fully independent processing; now it still rides the
    shared batch (computing one more metric — `cosine` — on the
    already-resident batch is cheap), and must produce results BIT-IDENTICAL
    to running it alone."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        SearchSpec(name="dense_all", vector_type="dense", metric="dot", k=3),
        SearchSpec(name="dense_eng", vector_type="dense", metric="cosine", k=2, filter=eng_filter),
    ]
    combined_cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "mismatch_metric_combined")),
        searches=specs,
    )
    combined_paths = run_compute(combined_cfg)

    solo_cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "mismatch_metric_solo")),
        searches=[specs[1]],
    )
    solo_path = run_compute(solo_cfg)["dense_eng"]

    # Crucially, the solo run here has NO unfiltered dense sibling, so its
    # shared batch is the union of its own (single) filter — batch.compact()
    # runs, unlike the combined run's whole-file pass above. Matching
    # bit-for-bit across both is a stronger proof than comparing either one
    # to itself.
    combined_t = pq.read_table(combined_paths["dense_eng"]).to_pydict()
    solo_t = pq.read_table(solo_path).to_pydict()
    assert combined_t["hit_ids"] == solo_t["hit_ids"]
    for combined_scores, solo_scores in zip(combined_t["hit_scores"], solo_t["hit_scores"]):
        assert np.allclose(combined_scores, solo_scores, atol=1e-5)


def test_sparse_cosine_filtered_spec_riding_dot_baseline_not_double_normalized(tmp_path):
    """Sparse analog of the test above, with hand-computed expected scores
    (not nova_bf's own scoring, like `test_three_member_group_mixed_metrics`):
    the only unfiltered sparse sibling is `dot`, so the filtered `cosine`
    search must compute its OWN normalization on the masked-down columns of
    the shared (raw, unnormalized) score matrix — never double-normalized,
    never silently left un-normalized."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # c0 (eng) = {1: 3.0, 2: 4.0}, norm 5.0 | c1 (fra) = {1: 1.0}, norm 1.0
    corpus_rows = [([1, 2], [3.0, 4.0]), ([1], [1.0])]
    dummy_dense = np.random.default_rng(6).standard_normal((2, DIM)).astype(np.float32)
    _write_combined(
        cdir / "f0.parquet", dummy_dense, corpus_rows, id=["c0", "c1"], language=["eng", "fra"],
    )

    # q0 = {1: 1.0, 2: 1.0}, norm sqrt(2)
    query_rows = [([1, 2], [1.0, 1.0])]
    qpath = tmp_path / "queries.parquet"
    _write_combined(
        qpath, np.random.default_rng(7).standard_normal((1, DIM)).astype(np.float32), query_rows, qid=["q0"],
    )

    out = tmp_path / "out"
    out.mkdir()
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(name="sparse_dot_all", vector_type="sparse", metric="dot", k=2),
            SearchSpec(name="sparse_cos_eng", vector_type="sparse", metric="cosine", k=2, filter=eng_filter),
        ],
    )
    paths = run_compute(cfg)

    cos_scores = dict(zip(*[pq.read_table(paths["sparse_cos_eng"]).to_pydict()[c][0] for c in ("hit_ids", "hit_scores")]))
    # only c0 survives the "eng" filter; dot(q0, c0) = 1*3+1*4 = 7, norms sqrt(2) and 5.0
    sqrt2 = np.sqrt(2.0)
    assert list(cos_scores) == ["c0"]
    assert cos_scores["c0"] == pytest.approx(7.0 / (sqrt2 * 5.0), abs=1e-4)


def test_multi_batch_filtered_spec_riding_shared_batch_correct(ds):
    """A filtered dense search sharing the whole-file batch grid with an
    unfiltered sibling must still accumulate the correct top-K across
    MULTIPLE batch boundaries (`params.dense_batch_size` forces several
    batches per file) — not just when the whole file fits in one batch."""
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    specs = [
        SearchSpec(name="dense_all", vector_type="dense", metric="dot", k=3),
        SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=2, filter=eng_filter),
    ]
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=_out(ds, "multi_batch_shared")),
        params=ParamsConfig(dense_batch_size=2),  # SIZES = [5, 7, 4] -> multiple batches/file
        searches=specs,
    )
    paths = run_compute(cfg)

    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    expectations = {
        "dense_all": ds["ground_truth"]("dense", 3),
        "dense_eng": ds["ground_truth"]("dense", 2, allowed=eng_globals),
    }
    for name, expected in expectations.items():
        t = pq.read_table(paths[name]).to_pydict()
        got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
        for q in ds["qids"]:
            assert got[q] == expected[q], f"search={name} query={q}"


def test_filtered_spec_zero_surviving_rows_in_a_middle_batch(tmp_path):
    """A filtered search riding the whole-file shared batch grid must handle a
    batch where NONE of its rows survive its filter — not just an all-or-
    nothing whole-file case (see `test_corpus_ids_resolved_even_when_a_
    different_specs_filter_drops_the_file` for the whole-file version). Rows
    are laid out `eng, eng, fra, fra, eng, eng` with `dense_batch_size=2`, so
    the SECOND batch (rows 2-3) has zero survivors for the "eng" filter while
    the first and third batches have two each."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n = 6
    rng = np.random.default_rng(8)
    dense_rows = rng.standard_normal((n, DIM)).astype(np.float32)
    sparse_rows = [_random_sparse_row(rng) for _ in range(n)]
    ids = [f"c{i}" for i in range(n)]
    lang = ["eng", "eng", "fra", "fra", "eng", "eng"]
    _write_combined(cdir / "f0.parquet", dense_rows, sparse_rows, id=ids, language=lang)

    q_dense = rng.standard_normal((2, DIM)).astype(np.float32)
    q_sparse = [_random_sparse_row(rng) for _ in range(2)]
    qids = ["q0", "q1"]
    qpath = tmp_path / "queries.parquet"
    _write_combined(qpath, q_dense, q_sparse, qid=qids)

    eng_globals = [i for i, lg in enumerate(lang) if lg == "eng"]
    expected_s = q_dense @ dense_rows[eng_globals].T
    expected = {
        qids[i]: [ids[eng_globals[j]] for j in np.argsort(-expected_s[i])[:2]] for i in range(2)
    }

    out = tmp_path / "out"
    out.mkdir()
    eng_filter = Filter(must=[FilterCondition(field="language", match="eng")])
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(dense_batch_size=2),
        searches=[
            SearchSpec(name="dense_all", vector_type="dense", metric="dot", k=2),
            SearchSpec(name="dense_eng", vector_type="dense", metric="dot", k=2, filter=eng_filter),
        ],
    )
    paths = run_compute(cfg)
    t = pq.read_table(paths["dense_eng"]).to_pydict()
    got = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
    for q in qids:
        assert got[q] == expected[q]


def test_mismatched_dense_and_sparse_query_loads_are_rejected(ds, monkeypatch):
    """Regression test for the query cross-check's failure path: if a run
    needs both vector_types and `load_queries`/`load_queries_sparse` ever
    disagree on query identity/order (not just count) for the same query
    store, `run_compute` must raise rather than silently misattributing one
    load's ids/payload to the other's vectors."""
    orig_load_sparse = compute_mod.load_queries_sparse

    def reordered_load_queries_sparse(store, qcfg, filter_cols=()):
        Q_np, vocab, q_ids, payload, filter_vals = orig_load_sparse(store, qcfg, filter_cols)
        assert len(q_ids) > 1  # sanity: reordering must actually change something
        return Q_np, vocab, list(reversed(q_ids)), payload, filter_vals

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


# --- per-query filters (match_from_query / range_from_query / match_text_from_query) ---


def _tenant_dataset(tmp_path, corpus_tenants, query_tenants, dim=DIM, seed=10):
    """A small dense-only corpus/query set with a `tenant_id` payload column
    on the corpus side and a matching column on the queries side, for
    per-query filter tests. Returns (cfg kwargs pieces, dense arrays) so
    callers can compute independent ground truth."""
    rng = np.random.default_rng(seed)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n = len(corpus_tenants)
    dense = rng.standard_normal((n, dim)).astype(np.float32)
    ids = [f"c{i}" for i in range(n)]
    sparse_rows = [_random_sparse_row(rng) for _ in range(n)]
    _write_combined(cdir / "f0.parquet", dense, sparse_rows, id=ids, tenant_id=list(corpus_tenants))

    n_q = len(query_tenants)
    qdense = rng.standard_normal((n_q, dim)).astype(np.float32)
    q_sparse_rows = [_random_sparse_row(rng) for _ in range(n_q)]
    qpath = tmp_path / "queries.parquet"
    qids = [f"q{i}" for i in range(n_q)]
    _write_combined(qpath, qdense, q_sparse_rows, qid=qids, tenant_id=list(query_tenants))

    return {
        "cdir": str(cdir), "qpath": str(qpath), "ids": ids, "qids": qids,
        "dense": dense, "qdense": qdense,
    }


def test_per_query_match_from_query_matches_independent_ground_truth(tmp_path):
    ds = _tenant_dataset(
        tmp_path,
        corpus_tenants=["A", "B", "A", "C", "B", "A"],
        query_tenants=["A", "B", "C"],
    )
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(
            name="per_tenant", vector_type="dense", metric="dot", k=10,
            filter=Filter(must=[FilterCondition(field="tenant_id", match_from_query="tenant_id")]),
        )],
    )
    paths = run_compute(cfg)
    t = pq.read_table(paths["per_tenant"]).to_pydict()

    corpus_tenants = ["A", "B", "A", "C", "B", "A"]
    query_tenants = ["A", "B", "C"]
    for qid, hit_ids in zip(t["query_id"], t["hit_ids"]):
        qi = int(qid[1:])
        allowed = [i for i in range(len(ds["ids"])) if corpus_tenants[i] == query_tenants[qi]]
        scores = ds["qdense"][qi] @ ds["dense"][allowed].T
        order = np.argsort(-scores)[:10]
        expected = [ds["ids"][allowed[j]] for j in order]
        assert hit_ids == expected, f"query={qid}"


def test_per_query_range_from_query_matches_independent_ground_truth(tmp_path):
    rng = np.random.default_rng(11)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n = 12
    dense = rng.standard_normal((n, DIM)).astype(np.float32)
    cost = rng.uniform(1, 30, n).astype(np.float32)
    ids = [f"c{i}" for i in range(n)]
    sparse_rows = [_random_sparse_row(rng) for _ in range(n)]
    _write_combined(cdir / "f0.parquet", dense, sparse_rows, id=ids, cost=cost.tolist())

    n_q = 3
    qdense = rng.standard_normal((n_q, DIM)).astype(np.float32)
    q_sparse_rows = [_random_sparse_row(rng) for _ in range(n_q)]
    budgets = [8.0, 15.0, 30.0]
    qpath = tmp_path / "queries.parquet"
    qids = [f"q{i}" for i in range(n_q)]
    _write_combined(qpath, qdense, q_sparse_rows, qid=qids, max_budget=budgets)

    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(
            name="budget_capped", vector_type="dense", metric="dot", k=10,
            filter=Filter(must=[FilterCondition(
                field="cost", range_from_query=RangeFromQuery(lt="max_budget"),
            )]),
        )],
    )
    paths = run_compute(cfg)
    t = pq.read_table(paths["budget_capped"]).to_pydict()
    for qid, hit_ids in zip(t["query_id"], t["hit_ids"]):
        qi = int(qid[1:])
        allowed = [i for i in range(n) if cost[i] < budgets[qi]]
        scores = qdense[qi] @ dense[allowed].T
        order = np.argsort(-scores)[:10]
        expected = [ids[allowed[j]] for j in order]
        assert hit_ids == expected, f"query={qid}"


def test_per_query_match_text_from_query_matches_independent_ground_truth(tmp_path):
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    titles = [
        "wireless mouse", "gaming keyboard", "wireless keyboard combo",
        "mouse pad xl", "bluetooth mouse", "mechanical keyboard",
    ]
    n = len(titles)
    rng = np.random.default_rng(12)
    dense = rng.standard_normal((n, DIM)).astype(np.float32)
    ids = [f"c{i}" for i in range(n)]
    sparse_rows = [_random_sparse_row(rng) for _ in range(n)]
    _write_combined(cdir / "f0.parquet", dense, sparse_rows, id=ids, title=titles)

    n_q = 3
    qdense = rng.standard_normal((n_q, DIM)).astype(np.float32)
    q_sparse_rows = [_random_sparse_row(rng) for _ in range(n_q)]
    phrases = ["wireless mouse", "keyboard", "wireless mouse"]  # q0/q2 share a phrase
    qpath = tmp_path / "queries.parquet"
    qids = [f"q{i}" for i in range(n_q)]
    _write_combined(qpath, qdense, q_sparse_rows, qid=qids, phrase=phrases)

    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column="id"),
        queries=QueriesConfig(path=str(qpath), id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(
            name="by_phrase", vector_type="dense", metric="dot", k=10,
            filter=Filter(must=[FilterCondition(field="title", match_text_from_query="phrase")]),
        )],
    )
    paths = run_compute(cfg)
    t = pq.read_table(paths["by_phrase"]).to_pydict()

    import re

    def word_match(phrase, title):
        return all(re.search(rf"\b{re.escape(w)}\b", title, re.IGNORECASE) for w in phrase.split())

    hits = dict(zip(t["query_id"], t["hit_ids"]))
    for qi, phrase in enumerate(phrases):
        qid = f"q{qi}"
        allowed = [i for i in range(n) if word_match(phrase, titles[i])]
        scores = qdense[qi] @ dense[allowed].T
        order = np.argsort(-scores)[:10]
        expected = [ids[allowed[j]] for j in order]
        assert hits[qid] == expected, f"query={qid}"
    # q0 and q2 share an identical phrase -> identical candidate set
    assert set(hits["q0"]) == set(hits["q2"]) or hits["q0"] == hits["q2"]


def test_per_query_filter_sharing_vt_with_unfiltered_baseline(tmp_path):
    """A per-query-filtered spec riding the whole-file batch alongside an
    unfiltered sibling (has_baseline True because of the unfiltered spec)
    must still produce correct, independently-verifiable results."""
    ds = _tenant_dataset(tmp_path, corpus_tenants=["A", "B", "A", "C"], query_tenants=["A", "C"])
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(name="dense_all", vector_type="dense", metric="dot", k=10),
            SearchSpec(
                name="per_tenant", vector_type="dense", metric="dot", k=10,
                filter=Filter(must=[FilterCondition(field="tenant_id", match_from_query="tenant_id")]),
            ),
        ],
    )
    paths = run_compute(cfg)

    corpus_tenants = ["A", "B", "A", "C"]
    query_tenants = ["A", "C"]
    t_all = pq.read_table(paths["dense_all"]).to_pydict()
    for qi, hit_ids in enumerate(t_all["hit_ids"]):
        scores = ds["qdense"][qi] @ ds["dense"].T
        order = np.argsort(-scores)[:10]
        assert hit_ids == [ds["ids"][j] for j in order]

    t_pt = pq.read_table(paths["per_tenant"]).to_pydict()
    for qid, hit_ids in zip(t_pt["query_id"], t_pt["hit_ids"]):
        qi = int(qid[1:])
        allowed = [i for i in range(4) if corpus_tenants[i] == query_tenants[qi]]
        scores = ds["qdense"][qi] @ ds["dense"][allowed].T
        order = np.argsort(-scores)[:10]
        assert hit_ids == [ds["ids"][allowed[j]] for j in order]


def test_per_query_filter_sharing_vt_with_uniform_filter_no_baseline(tmp_path, caplog):
    """The trickiest routing interaction: a per-query-filtered spec and a
    UNIFORM-filtered spec share a vector_type with NO unfiltered spec at
    all. `has_baseline` must still become True (forced by the per-query
    spec alone) — confirmed via the log line — and BOTH specs must still
    produce correct results, the uniform one riding the whole-file base
    instead of getting its own union-compaction."""
    ds = _tenant_dataset(tmp_path, corpus_tenants=["A", "B", "A", "C"], query_tenants=["A", "C"])
    out = tmp_path / "out"
    out.mkdir()
    eng_like_filter = Filter(must=[FilterCondition(field="tenant_id", match=["A", "B"])])
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[
            SearchSpec(
                name="uniform_ab", vector_type="dense", metric="dot", k=10, filter=eng_like_filter,
            ),
            SearchSpec(
                name="per_tenant", vector_type="dense", metric="dot", k=10,
                filter=Filter(must=[FilterCondition(field="tenant_id", match_from_query="tenant_id")]),
            ),
        ],
    )
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        paths = run_compute(cfg)
    assert any("share one full-file batch pass" in r.message for r in caplog.records), \
        "a per-query filter alone must force has_baseline, even with no unfiltered sibling"
    assert not any("union of" in r.message for r in caplog.records)

    corpus_tenants = ["A", "B", "A", "C"]
    query_tenants = ["A", "C"]
    t_uniform = pq.read_table(paths["uniform_ab"]).to_pydict()
    ab_rows = [i for i in range(4) if corpus_tenants[i] in ("A", "B")]
    for qi, hit_ids in enumerate(t_uniform["hit_ids"]):
        scores = ds["qdense"][qi] @ ds["dense"][ab_rows].T
        order = np.argsort(-scores)[:10]
        assert hit_ids == [ds["ids"][ab_rows[j]] for j in order]

    t_pt = pq.read_table(paths["per_tenant"]).to_pydict()
    for qid, hit_ids in zip(t_pt["query_id"], t_pt["hit_ids"]):
        qi = int(qid[1:])
        allowed = [i for i in range(4) if corpus_tenants[i] == query_tenants[qi]]
        scores = ds["qdense"][qi] @ ds["dense"][allowed].T
        order = np.argsort(-scores)[:10]
        assert hit_ids == [ds["ids"][allowed[j]] for j in order]


def test_per_query_filter_sharded_compute_and_merge(tmp_path):
    ds = _tenant_dataset(
        tmp_path, corpus_tenants=["A", "B", "A", "C", "B", "A", "C", "B"], query_tenants=["A", "B", "C"],
    )
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(
            name="per_tenant", vector_type="dense", metric="dot", k=10,
            filter=Filter(must=[FilterCondition(field="tenant_id", match_from_query="tenant_id")]),
        )],
    )
    single_paths = run_compute(cfg, num_jobs=None)
    single_t = pq.read_table(single_paths["per_tenant"]).to_pydict()

    shard_out = tmp_path / "shard_out"
    shard_out.mkdir()
    shard_cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(shard_out)),
        searches=cfg.searches,
    )
    num_jobs = 2
    for rank in range(num_jobs):
        run_compute(shard_cfg, num_jobs=num_jobs, job_rank=rank)
    merged = run_merge(shard_cfg)
    merged_t = pq.read_table(merged["per_tenant"]).to_pydict()

    assert merged_t["query_id"] == single_t["query_id"]
    assert merged_t["hit_ids"] == single_t["hit_ids"]
    for a, b in zip(merged_t["hit_scores"], single_t["hit_scores"]):
        assert np.allclose(a, b, atol=1e-5)


def test_per_query_filter_missing_queries_column_raises_clear_error(tmp_path):
    ds = _tenant_dataset(tmp_path, corpus_tenants=["A", "B"], query_tenants=["A"])
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["qpath"], id_column="qid"),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(
            name="bad", vector_type="dense", metric="dot", k=5,
            filter=Filter(must=[FilterCondition(field="tenant_id", match_from_query="nonexistent_col")]),
        )],
    )
    with pytest.raises(ValueError, match="nonexistent_col"):
        run_compute(cfg)
