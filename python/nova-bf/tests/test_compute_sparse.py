"""Correctness tests for the sparse brute-force compute path.

Mirrors test_compute.py's fixture/ground-truth pattern, but over
struct<indices: list<uint32>, values: list<float32>> columns — the same
on-disk schema nova-embed writes and nova-load reads. Ground truth is computed
independently, by densifying the tiny synthetic corpus/queries over their full
observed vocabulary in plain numpy (not by re-deriving nova_bf's own
query-vocab-truncation logic), so these tests catch divergence between the two.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # the compute phase needs torch (install nova-bf[dev])
import pyarrow as pa
import pyarrow.parquet as pq

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
from nova_bf.ids import make_point_id
from nova_bf.io import Store

K = 3
SIZES = [5, 7, 4]  # 3 corpus files → 16 vectors; row counts that don't divide the batch
VOCAB = 12  # token ids used by queries; corpus also uses an out-of-vocab id (see below)
NNZ = 8  # fixed nnz per row: 2*NNZ > VOCAB guarantees any two rows share >=1 token
         # (pigeonhole on same-size subsets), so scores are essentially never
         # exact ties — real-valued dense-test vectors get this for free.
OOV_TOKEN = VOCAB + 5  # never appears in any query — must be dropped, not error


def _random_row(rng, vocab, nnz=NNZ):
    idx = rng.choice(vocab, size=nnz, replace=False)
    val = rng.standard_normal(nnz).astype(np.float32)
    return idx.tolist(), val.tolist()


def _write_sparse_vectors(path, rows, **columns):
    data = {
        "sparse_embedding": pa.array(
            [{"indices": idx, "values": val} for idx, val in rows],
            type=pa.struct([
                pa.field("indices", pa.list_(pa.uint32())),
                pa.field("values", pa.list_(pa.float32())),
            ]),
        )
    }
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


def _dense(rows, vocab):
    """Densify (idx, val) rows over [0, vocab) — token ids >= vocab (e.g. the
    corpus's deliberately out-of-query-vocab OOV_TOKEN) are dropped, mirroring
    nova_bf's own query-vocab truncation (see _build_query_vocab)."""
    out = np.zeros((len(rows), vocab), dtype=np.float64)
    for i, (idx, val) in enumerate(rows):
        idx = np.asarray(idx)
        val = np.asarray(val)
        keep = idx < vocab
        out[i, idx[keep]] = val[keep]
    return out


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("bf_sparse")
    cdir = tmp / "corpus"
    cdir.mkdir()

    corpus_rows, ids_by_g, loc_by_g, lang_by_g, g = [], [], [], [], 0
    per_file_rows, per_file_ids, per_file_lang = [], [], []
    for fi, n in enumerate(SIZES):
        rows = [_random_row(rng, VOCAB) for _ in range(n)]
        ids = [f"c{g + r}" for r in range(n)]
        lang = [("eng" if (g + r) % 2 == 0 else "fra") for r in range(n)]
        per_file_rows.append(rows)
        per_file_ids.append(ids)
        per_file_lang.append(lang)
        corpus_rows += rows
        ids_by_g += ids
        loc_by_g += [(fi, r) for r in range(n)]
        lang_by_g += lang
        g += n

    # inject an out-of-query-vocab token into corpus row 0 — must be dropped
    # silently (it can never match any query), not error.
    idx0, val0 = per_file_rows[0][0]
    per_file_rows[0][0] = (idx0 + [OOV_TOKEN], val0 + [7.0])
    corpus_rows[0] = per_file_rows[0][0]

    for fi in range(len(SIZES)):
        _write_sparse_vectors(
            cdir / f"f{fi}.parquet", per_file_rows[fi], id=per_file_ids[fi], language=per_file_lang[fi],
        )

    n_q = 4
    query_rows = [_random_row(rng, VOCAB) for _ in range(n_q)]
    qids = [f"q{i}" for i in range(n_q)]
    qpath = tmp / "queries.parquet"
    _write_sparse_vectors(qpath, query_rows, qid=qids)

    corpus_dense = _dense(corpus_rows, VOCAB)  # OOV token dropped: only [0, VOCAB) columns
    queries_dense = _dense(query_rows, VOCAB)
    # Cosine normalizes by each row's TRUE (untruncated) L2 norm — truncating a
    # corpus row to the query vocab never changes a dot product (the dropped
    # dimensions are all outside every query's support anyway), but it WOULD
    # shrink the row's norm if computed post-truncation, artificially inflating
    # its cosine score. So norms come from the raw (idx, val) rows, not from
    # the truncated dense array — matching nova_bf's own design (see
    # _sparse_batch_to_csr's row-norm scaling, applied before vocab dropping).
    corpus_norms = np.array([np.linalg.norm(val) for _, val in corpus_rows])
    query_norms = np.array([np.linalg.norm(val) for _, val in query_rows])

    def ground_truth(metric, allowed=None):
        """Top-K (global idx, score) per query, optionally restricted to `allowed`
        global corpus indices (for filter tests) — same shape as the dense
        test's `_filtered_ground_truth`."""
        rows = allowed if allowed is not None else list(range(len(corpus_rows)))
        c = corpus_dense[rows]
        q = queries_dense
        s = q @ c.T  # (n_q, len(rows)) — truncation-safe: dot product is unaffected
        if metric == "cosine":
            s = s / query_norms.clip(min=1e-12)[:, None] / corpus_norms[rows].clip(min=1e-12)[None, :]
        expected = {}
        for i, qid in enumerate(qids):
            order = np.argsort(-s[i])[:K]
            expected[qid] = [(rows[j], float(s[i, j])) for j in order]
        return expected

    return {
        "cdir": str(cdir),
        "qpath": str(qpath),
        "tmp": tmp,
        "qids": qids,
        "ids_by_g": ids_by_g,
        "loc_by_g": loc_by_g,
        "lang_by_g": lang_by_g,
        "ground_truth": ground_truth,
        "files": Store(str(cdir)).list_parquets(),
    }


def _run(ds, *, metric, batch, id_column, out_name, filt=None):
    out = ds["tmp"] / out_name
    out.mkdir(exist_ok=True)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], sparse_column="sparse_embedding", id_column=id_column),
        queries=QueriesConfig(path=ds["qpath"], sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, sparse_batch_size=batch),
        searches=[SearchSpec(name="test", k=K, metric=metric, vector_type="sparse", filter=filt)],
    )
    t = pq.read_table(run_compute(cfg)["test"]).to_pydict()
    return {q: list(zip(hi, hs)) for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}


@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_neighbors_match_bruteforce(ds, metric):
    res = _run(ds, metric=metric, batch=None, id_column=None, out_name=f"nn_{metric}")
    expected = ds["ground_truth"](metric)
    for q in ds["qids"]:
        got = sorted((s for _, s in res[q]), reverse=True)
        exp = sorted((s for _, s in expected[q]), reverse=True)
        assert np.allclose(got, exp, atol=1e-4)


@pytest.mark.parametrize("id_column", [None, "id"])
@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_tiling_is_invariant(ds, id_column, metric):
    """Batched scoring must give identical neighbors to whole-file scoring —
    cosine is included, not just dot, since the per-row norm scaling in
    _sparse_batch_to_csr is sliced per-batch and could drift out of sync with
    the file's row space under tiling if that indexing were ever wrong."""
    whole = _run(ds, metric=metric, batch=None, id_column=id_column, out_name=f"whole_{metric}_{id_column}")
    tiled = _run(ds, metric=metric, batch=K, id_column=id_column, out_name=f"tiled_{metric}_{id_column}")
    for q in ds["qids"]:
        ids_whole = [i for i, _ in whole[q]]
        ids_tiled = [i for i, _ in tiled[q]]
        assert ids_whole == ids_tiled
        sc_whole = [s for _, s in whole[q]]
        sc_tiled = [s for _, s in tiled[q]]
        assert np.allclose(sc_whole, sc_tiled, atol=1e-3)


def _run_multi(ds, *, order, out_name):
    """Like `_run` but with several sparse searches sharing one pass — one
    per metric in `order`, preserving that order in the config."""
    out = ds["tmp"] / out_name
    out.mkdir(exist_ok=True)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], sparse_column="sparse_embedding"),
        queries=QueriesConfig(path=ds["qpath"], sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2),
        searches=[
            SearchSpec(name=f"s_{m}", k=K, metric=m, vector_type="sparse") for m in order
        ],
    )
    paths = run_compute(cfg)
    res = {}
    for m in order:
        t = pq.read_table(paths[f"s_{m}"]).to_pydict()
        res[m] = {q: list(zip(hi, hs)) for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}
    return res


def test_metric_order_permutation_invariant(ds):
    """One raw spmm per slice is shared across metrics (`_masked_raw`), built
    by whichever metric scores FIRST — so spec order must not matter, and a
    co-scheduled search must equal its own solo run exactly."""
    a = _run_multi(ds, order=["dot", "cosine"], out_name="perm_dc")
    b = _run_multi(ds, order=["cosine", "dot"], out_name="perm_cd")
    assert a == b
    for metric in ("dot", "cosine"):
        solo = _run(ds, metric=metric, batch=None, id_column=None, out_name=f"perm_solo_{metric}")
        assert a[metric] == solo


def test_precoalesced_file_slices_are_valid_csr(tmp_path):
    """Files are remapped + coalesced ONCE in the reader (`_remap_sparse_file`)
    and batch slices are pure nnz-range views — a file whose rows carry
    duplicate AND unsorted tokens, split across 2-row slices, must match the
    whole-file run and still sum duplicates (row c0's token 3 appears twice:
    1.0 + 4.0)."""
    corpus = [[
        ([3, 0, 3], [1.0, 2.0, 4.0]),  # unsorted + duplicated token 3
        ([1], [1.0]),
        ([2, 2, 0], [0.5, 0.5, 1.0]),
        ([0, 1], [2.0, 3.0]),
        ([4, 3], [1.0, 1.5]),  # 1.5 not 1.0: keep scores tie-free — tie ORDER
        # across different batchings is deliberately unpromised (see the
        # parity-locked tie-break design), and this test is about CSR
        # validity, not tie order.
    ]]
    queries = [([0, 3], [1.0, 1.0])]
    (tmp_path / "whole").mkdir()
    whole = _run_tiny(tmp_path / "whole", corpus, queries, k=5)
    (tmp_path / "sliced").mkdir()
    sliced = _run_tiny(tmp_path / "sliced", corpus, queries, k=5, batch=2)
    assert whole == sliced
    scores = dict(whole["q0"])
    assert scores["c0"] == pytest.approx(2.0 + 1.0 + 4.0)  # dup token summed


def test_zero_gate_file_ok_predicate():
    """Unit contract of the cheap-gate guard: strictly-positive values with a
    safe min-product qualify; signed queries, signed corpus values,
    STORED-ZERO corpus entries (structural overlaps that dot as 0.0 — see
    the helper's docstring), and denormal-underflow products all fall back
    to the structural indicator path."""
    from nova_bf.compute import _zero_gate_file_ok

    pos = np.array([0.5, 2.0], np.float32)
    assert _zero_gate_file_ok(pos, True, 0.01)
    assert not _zero_gate_file_ok(pos, False, 0.01)  # signed query side
    assert not _zero_gate_file_ok(np.array([-0.1, 1.0], np.float32), True, 0.01)
    assert not _zero_gate_file_ok(np.array([0.0, 1.0], np.float32), True, 0.01)  # stored zero
    assert not _zero_gate_file_ok(np.array([1e-25], np.float32), True, 1e-25)  # underflow
    assert _zero_gate_file_ok(np.zeros(0, np.float32), True, 0.01)  # empty file
    assert _zero_gate_file_ok(pos, True, float("inf"))  # queries with no positives at all


@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_zero_gate_and_structural_gate_agree_on_positive_data(tmp_path, metric, monkeypatch):
    """On strictly-positive values the cheap `raw == 0` gate and the signed
    -data structural indicator must select IDENTICAL candidates: run the
    same corpus twice, once with the gate eligible and once force-disabled
    (impossible `_ZERO_GATE_MIN_PRODUCT` pushes every file structural).
    Corpus includes overlap-only, partial-overlap, and zero-overlap docs so
    both inclusion and exclusion are exercised."""
    import nova_bf.compute as compute_mod

    corpus = [
        [([0, 1], [1.0, 2.0]), ([2], [3.0])],
        [([8], [4.0]), ([0, 8, 9], [0.5, 5.0, 6.0])],
    ]
    queries = [([0, 2], [1.0, 1.0]), ([9], [2.0]), ([5], [1.0])]  # q2 overlaps nothing

    (tmp_path / "fast").mkdir()
    fast = _run_tiny(tmp_path / "fast", corpus, queries, metric=metric, k=4)
    monkeypatch.setattr(compute_mod, "_ZERO_GATE_MIN_PRODUCT", float("inf"))
    (tmp_path / "struct").mkdir()
    structural = _run_tiny(tmp_path / "struct", corpus, queries, metric=metric, k=4)
    assert fast == structural
    assert fast["q2"] == []  # zero-overlap query excluded under BOTH gates


@pytest.mark.parametrize("batch", [None, K])
def test_id_column_uses_raw_ids(ds, batch):
    res = _run(ds, metric="dot", batch=batch, id_column="id", out_name=f"idcol_{batch}")
    expected = ds["ground_truth"]("dot")
    for q in ds["qids"]:
        got = [i for i, _ in res[q]]
        want = [ds["ids_by_g"][g] for g, _ in expected[q]]
        assert got == want


def test_default_uses_make_point_id(ds):
    res = _run(ds, metric="dot", batch=None, id_column=None, out_name="hash")
    expected = ds["ground_truth"]("dot")
    files = ds["files"]
    for q in ds["qids"]:
        got = [i for i, _ in res[q]]
        want = []
        for g, _ in expected[q]:
            fi, row = ds["loc_by_g"][g]
            want.append(make_point_id(files[fi].key, row))
        assert got == want


def test_filter_match_restricts_candidates(ds):
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    res = _run(ds, metric="dot", batch=None, id_column="id", out_name="filter_lang", filt=filt)
    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    expected = ds["ground_truth"]("dot", allowed=eng_globals)
    for q in ds["qids"]:
        got = [i for i, _ in res[q]]
        assert got  # sanity: something survived the filter
        assert got == [ds["ids_by_g"][g] for g, _ in expected[q]]


@pytest.mark.parametrize("batch", [None, K])
def test_filter_preserves_row_numbers_under_batching(ds, batch):
    """A filter must resolve the same (correct) point ids whether or not
    params.sparse_batch_size tiles the file — guards against filtering
    renumbering rows instead of keeping their true file-row number."""
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    res = _run(ds, metric="dot", batch=batch, id_column=None, out_name=f"filter_defid_{batch}", filt=filt)
    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    expected = ds["ground_truth"]("dot", allowed=eng_globals)
    files = ds["files"]
    for q in ds["qids"]:
        want = []
        for g, _ in expected[q]:
            fi, row = ds["loc_by_g"][g]
            want.append(make_point_id(files[fi].key, row))
        assert [i for i, _ in res[q]] == want


def test_out_of_vocab_corpus_token_is_dropped_not_errored(ds):
    # Exercised implicitly by every test above (corpus row 0 carries OOV_TOKEN,
    # and ground_truth() is computed over the truncated [0, VOCAB) space, so a
    # mismatch here would mean nova_bf scored the OOV token instead of dropping
    # it). This test just asserts the run completes and returns full k hits.
    res = _run(ds, metric="dot", batch=None, id_column=None, out_name="oov_smoke")
    for q in ds["qids"]:
        assert len(res[q]) == K


def test_sparse_euclidean_rejected():
    with pytest.raises(ValueError, match="euclidean"):
        SearchSpec(name="s", vector_type="sparse", metric="euclidean")


def test_large_hashed_token_ids_do_not_blow_up_memory():
    """Regression test: real hashed sparse schemes (e.g. fastembed's BM25)
    scatter token ids across the full uint32 range. _build_query_vocab must
    stay sized by the number of DISTINCT tokens, not by the largest token id —
    a dense `remap[token_id]` array sized by max(id) would allocate ~17GB for
    just 2 distinct tokens here (confirmed against real BM25 output, whose
    token ids commonly exceed 2 billion)."""
    from nova_bf.compute import _build_query_vocab, _vocab_lookup

    huge_ids = np.array([2_145_091_943, 613_153_351], dtype=np.int64)
    vocab = _build_query_vocab(huge_ids)
    assert len(vocab) == 2
    assert vocab.nbytes < 1_000  # nowhere near the ~17GB a dense remap would need
    # exact round-trip: every id we built the vocab from must resolve back
    looked_up = _vocab_lookup(vocab, huge_ids)
    assert (vocab[looked_up] == huge_ids).all()
    # an id absent from the vocab must resolve to -1, not a wrong/adjacent slot
    assert _vocab_lookup(vocab, np.array([123], dtype=np.int64))[0] == -1


def test_duplicate_indices_within_a_row_are_summed(ds):
    """A row (query or corpus) whose sparse `indices` contains the same token
    id twice must have its values SUMMED, not have the later write silently
    clobber the earlier one — this matches how the corpus-side CSR tensor
    already sums repeated column indices within a row by construction."""
    tmp = ds["tmp"]
    cdir = tmp / "dup_corpus"
    cdir.mkdir(exist_ok=True)
    # one corpus row with token 3 appearing twice (values 2.0 and 5.0 -> sum 7.0)
    _write_sparse_vectors(cdir / "f0.parquet", [([3, 3, 1], [2.0, 5.0, 1.0])], id=["dup0"], language=["eng"])
    qpath = tmp / "dup_queries.parquet"
    # query also repeats token 3 (values 1.0 and 1.0 -> sum 2.0) and includes token 1
    _write_sparse_vectors(qpath, [([3, 3, 1], [1.0, 1.0, 10.0])], qid=["dq0"])

    out = tmp / "dup_out"
    out.mkdir(exist_ok=True)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding", id_column="id"),
        queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1),
        searches=[SearchSpec(name="test", k=1, metric="dot", vector_type="sparse")],
    )
    t = pq.read_table(run_compute(cfg)["test"]).to_pydict()
    # correct: query{3:2.0, 1:10.0} . corpus{3:7.0, 1:1.0} = 2.0*7.0 + 10.0*1.0 = 24.0
    # if either side silently overwrote instead of summing, this would come out wrong
    assert t["hit_scores"][0][0] == pytest.approx(24.0, abs=1e-4)


def test_coalesce_by_row_col_merges_duplicates_and_builds_valid_csr():
    """_coalesce_by_row_col must both (a) sum duplicate (row, col) values and
    (b) leave every (row, col) pair appearing at most once, sorted by row then
    col — i.e. genuinely satisfy torch's CSR invariants, not just "happen to
    score right" via undefined behavior. Verified by constructing the result
    with check_invariants=True (torch's own validator), which rejects
    unsorted-or-duplicate CSR data — this is the strongest possible guard
    against silently regressing back to sort-only (no dedup)."""
    import torch

    from nova_bf.compute import _coalesce_by_row_col

    # row 0 has a duplicate col (3 twice: 2.0 + 5.0 -> 7.0) plus col 1 = 1.0
    row_ids = np.array([0, 0, 0], dtype=np.int64)
    col_ids = np.array([3, 1, 3], dtype=np.int64)
    values = np.array([2.0, 1.0, 5.0], dtype=np.float32)
    r, c, v = _coalesce_by_row_col(row_ids, col_ids, values)

    assert list(zip(r.tolist(), c.tolist(), v.tolist())) == [(0, 1, 1.0), (0, 3, 7.0)]

    counts = np.bincount(r, minlength=1)
    crow = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    # this raises if the result violates CSR's sorted-and-distinct invariant
    Cb = torch.sparse_csr_tensor(
        torch.from_numpy(crow), torch.from_numpy(c), torch.from_numpy(v),
        size=(1, 5), check_invariants=True,
    )
    assert Cb.to_dense().tolist() == [[0.0, 1.0, 0.0, 7.0, 0.0]]


def test_duplicate_indices_cosine_norm_is_correct(ds):
    """A row's cosine norm must be computed from its per-token-id SUMMED
    value, not from the sum of squares of each raw (pre-merge) occurrence —
    those differ whenever a row repeats a token id, and differ drastically
    for opposite-sign duplicates that net to a small true value but would
    inflate a sum-of-squares-of-parts norm."""
    tmp = ds["tmp"]
    cdir = tmp / "cosnorm_corpus"
    cdir.mkdir(exist_ok=True)
    # true row vector: {1: 1.0, 3: 7.0} (token 3 = 2.0 + 5.0) -> norm = sqrt(1+49) = sqrt(50)
    _write_sparse_vectors(cdir / "f0.parquet", [([3, 1, 3], [2.0, 1.0, 5.0])], id=["c0"], language=["eng"])
    qpath = tmp / "cosnorm_queries.parquet"
    _write_sparse_vectors(qpath, [([3, 1], [1.0, 1.0])], qid=["q0"])

    out = tmp / "cosnorm_out"
    out.mkdir(exist_ok=True)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding", id_column="id"),
        queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1),
        searches=[SearchSpec(name="test", k=1, metric="cosine", vector_type="sparse")],
    )
    t = pq.read_table(run_compute(cfg)["test"]).to_pydict()
    # dot = 1*1(token1) + 1*7(token3) = 8; query norm = sqrt(2); corpus norm = sqrt(50)
    expected = 8.0 / (np.sqrt(2) * np.sqrt(50))
    assert t["hit_scores"][0][0] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("batch", [None, 2])
def test_filter_compaction_with_multibatch_tiling(tmp_path, batch):
    """The shared `ds` fixture's per-file row counts are too small for any
    `params.sparse_batch_size` tiling to actually span more than one batch
    AFTER a filter has compacted a file — so this exercises that specific
    interaction directly: a file where filtering leaves enough rows that
    batch=2 forces multiple iterations (r0=0, 2, 4) of the post-compaction
    array, checking the same row-number-preservation invariant `orig_rows`
    relies on."""
    rng = np.random.default_rng(1)
    vocab, nnz = 10, 6  # 2*nnz > vocab: pigeonhole guarantees overlap, avoids ties
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    n = 10
    rows = [_random_row(rng, vocab, nnz) for _ in range(n)]
    lang = ["eng" if i % 2 == 0 else "fra" for i in range(n)]  # 5 eng rows survive
    ids = [f"c{i}" for i in range(n)]
    _write_sparse_vectors(cdir / "f0.parquet", rows, id=ids, language=lang)

    query_rows = [_random_row(rng, vocab, nnz) for _ in range(2)]
    qpath = tmp_path / "queries.parquet"
    _write_sparse_vectors(qpath, query_rows, qid=["q0", "q1"])

    eng_globals = [i for i in range(n) if lang[i] == "eng"]
    corpus_dense = _dense(rows, vocab)
    queries_dense = _dense(query_rows, vocab)
    s = queries_dense @ corpus_dense[eng_globals].T
    expected_ids = {
        f"q{qi}": [ids[eng_globals[j]] for j in np.argsort(-s[qi])[:2]] for qi in range(2)
    }

    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding", id_column="id"),
        queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, sparse_batch_size=batch),
        searches=[SearchSpec(
            name="test", k=2, metric="dot", vector_type="sparse",
            filter=Filter(must=[FilterCondition(field="language", match="eng")]),
        )],
    )
    t = pq.read_table(run_compute(cfg)["test"]).to_pydict()
    got_ids = {q: hi for q, hi in zip(t["query_id"], t["hit_ids"])}
    assert got_ids == expected_ids


def test_filter_compaction_preserves_duplicate_index_summing(tmp_path):
    """A row surviving the payload filter must still have its duplicate token
    ids correctly summed (via `_coalesce_by_row_col`) — i.e. `_compact_sparse_rows`'s
    nnz-level row compaction must preserve a surviving row's raw (possibly
    duplicate-containing) nonzeros intact, not just its row identity."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # row0 (kept, "eng"): token 3 twice (2.0, 5.0 -> sum 7.0) + token 1 (1.0)
    # row1 (dropped, "fra"): irrelevant content
    _write_sparse_vectors(
        cdir / "f0.parquet",
        [([3, 3, 1], [2.0, 5.0, 1.0]), ([2], [99.0])],
        id=["keep0", "drop1"], language=["eng", "fra"],
    )
    qpath = tmp_path / "queries.parquet"
    _write_sparse_vectors(qpath, [([3, 1], [1.0, 10.0])], qid=["q0"])

    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding", id_column="id"),
        queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1),
        searches=[SearchSpec(
            name="test", k=1, metric="dot", vector_type="sparse",
            filter=Filter(must=[FilterCondition(field="language", match="eng")]),
        )],
    )
    t = pq.read_table(run_compute(cfg)["test"]).to_pydict()
    # query{3:1.0, 1:10.0} . corpus{3:7.0, 1:1.0} = 1.0*7.0 + 10.0*1.0 = 17.0
    assert t["hit_ids"][0] == ["keep0"]
    assert t["hit_scores"][0][0] == pytest.approx(17.0, abs=1e-4)


def _run_tiny(tmp_path, corpus_files, query_rows, *, metric="dot", k=5, filt=None,
              languages=None, batch=None):
    """Standalone mini-runner for datasets that need EXACT control over token
    overlap (the module `ds` fixture deliberately guarantees every pair
    overlaps, see NNZ's comment — useless for zero-overlap tests)."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    g = 0
    for fi, rows in enumerate(corpus_files):
        ids = [f"c{g + r}" for r in range(len(rows))]
        cols = {"id": ids}
        if languages is not None:
            cols["language"] = languages[g : g + len(rows)]
        _write_sparse_vectors(cdir / f"f{fi}.parquet", rows, **cols)
        g += len(rows)
    qpath = tmp_path / "queries.parquet"
    _write_sparse_vectors(qpath, query_rows, qid=[f"q{i}" for i in range(len(query_rows))])
    out = tmp_path / "out"
    out.mkdir()
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding", id_column="id"),
        queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, sparse_batch_size=batch),
        searches=[SearchSpec(name="test", k=k, metric=metric, vector_type="sparse", filter=filt)],
    )
    t = pq.read_table(run_compute(cfg)["test"]).to_pydict()
    return {q: list(zip(hi, hs)) for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}


@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_zero_overlap_docs_are_excluded_not_zero_padded(tmp_path, metric):
    """A document sharing NO token with the query must not appear in its
    top-K at all — not even as 0.0-score padding when fewer than k real
    candidates exist. Sparse engines (Qdrant's inverted index) can never
    return such documents, so ground truth containing them would penalize a
    correctly-behaving engine. Split across two corpus files so the running
    top-K state folds a fully-excluded batch too."""
    res = _run_tiny(
        tmp_path,
        # c0/c1 overlap query tokens {0,1,2}; c2/c3 (file 2) live on disjoint
        # tokens {8,9} — in the query VOCAB of no query at all.
        [[([0, 1], [1.0, 2.0]), ([2], [3.0])],
         [([8], [4.0]), ([8, 9], [5.0, 6.0])]],
        [([0, 2], [1.0, 1.0])],
        metric=metric, k=4,
    )
    hits = res["q0"]
    assert [h for h, _ in hits] == sorted([h for h, _ in hits], key=dict(hits).get, reverse=True)
    assert {h for h, _ in hits} == {"c0", "c1"}, "zero-overlap docs must be absent, k unfilled"
    assert all(np.isfinite(s) for _, s in hits)


def test_zero_overlap_after_filter_yields_empty_result(tmp_path):
    """Filter leaves only zero-overlap docs -> the query gets ZERO hits (the
    Qdrant-parity answer), not k arbitrary 0.0-score rows. This is the exact
    shape of the live repro (sparse query + restrictive payload filter)."""
    res = _run_tiny(
        tmp_path,
        [[([0, 1], [1.0, 2.0]), ([8], [4.0]), ([9], [5.0])]],
        [([0, 1], [1.0, 1.0])],
        k=3,
        languages=["eng", "fra", "fra"],
        filt=Filter(must=[FilterCondition(field="language", match="fra")]),
    )
    assert res["q0"] == []


def test_overlapping_cancellation_zero_is_still_a_hit(tmp_path):
    """The no-overlap gate is structural (shared nonzero dims), NOT
    `score == 0.0`: a signed embedding can produce a genuine 0.0 dot from
    overlapping dimensions that cancel — that document IS a real candidate
    an engine could retrieve, and must stay in the results."""
    res = _run_tiny(
        tmp_path,
        # cancel: {0:1, 1:-1} . {0:1, 1:1} = 0.0, but dims 0/1 overlap.
        # nover: token 5 only -> structurally no candidate.
        [[([0, 1], [1.0, -1.0]), ([0], [2.0]), ([5], [9.0])]],
        [([0, 1], [1.0, 1.0])],
        k=3,
    )
    hits = dict(res["q0"])
    assert set(hits) == {"c0", "c1"}
    assert hits["c1"] == pytest.approx(2.0, abs=1e-5)
    assert hits["c0"] == pytest.approx(0.0, abs=1e-6), "cancellation zero must survive"
