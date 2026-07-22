"""Correctness tests for the brute-force compute phase.

Exercises the GPU topk path on CPU (small synthetic corpus + queries), covering
the two pieces most prone to silent corruption:
  - params.dense_batch_size tiling: must yield the *same* top-K as scoring the
    whole file at once (the `gidx * MAX_ROWS_PER_FILE + row` encoding uses
    file-local row offsets, so an off-by-a-batch bug would scramble hit ids), and
  - id resolution: hit_ids come from `corpus.id_column` when set (a real,
    pre-existing identifier) and from `make_point_id(file_key, row)` otherwise.
"""

from __future__ import annotations

import logging

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
    RangeCondition,
    SearchSpec,
)
from nova_bf.ids import make_point_id
from nova_bf.io import Store

DIM, K = 8, 3
SIZES = [5, 7, 4]  # 3 corpus files → 16 vectors; row counts that don't divide the batch


def _write_vectors(path, vectors, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(), type=pa.list_(pa.float32()))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    """A tiny corpus + query set, with the numpy brute-force ground truth."""
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("bf")
    cdir = tmp / "corpus"
    cdir.mkdir()

    corpus, ids_by_g, loc_by_g, lang_by_g, cost_by_g, g = [], [], [], [], [], 0
    for fi, n in enumerate(SIZES):
        rows = rng.standard_normal((n, DIM)).astype(np.float32)
        # deterministic payload for filter tests: alternating language, cost = global idx
        lang = [("eng" if (g + r) % 2 == 0 else "fra") for r in range(n)]
        cost = [float(g + r) for r in range(n)]
        _write_vectors(
            cdir / f"f{fi}.parquet", rows,
            id=[f"c{g + r}" for r in range(n)], language=lang, cost=cost,
        )
        corpus.append(rows)
        ids_by_g += [f"c{g + r}" for r in range(n)]
        loc_by_g += [(fi, r) for r in range(n)]  # global idx → (file, file-local row)
        lang_by_g += lang
        cost_by_g += cost
        g += n
    corpus = np.concatenate(corpus)

    n_q = 4
    queries = rng.standard_normal((n_q, DIM)).astype(np.float32)
    qids = [f"q{i}" for i in range(n_q)]
    qpath = tmp / "queries.parquet"
    _write_vectors(qpath, queries, qid=qids)

    # ground truth: top-K corpus globals per query, by dot product
    expected = {}
    for i, qid in enumerate(qids):
        s = corpus @ queries[i]
        expected[qid] = [(int(g), float(s[g])) for g in np.argsort(-s)[:K]]

    return {
        "cdir": str(cdir),
        "qpath": str(qpath),
        "tmp": tmp,
        "qids": qids,
        "ids_by_g": ids_by_g,
        "loc_by_g": loc_by_g,
        "lang_by_g": lang_by_g,
        "cost_by_g": cost_by_g,
        "expected": expected,
        "corpus": corpus,
        "queries": queries,
        "files": Store(str(cdir)).list_parquets(),  # same sorted order as run_compute's gidx
    }


def _run(ds, *, batch, id_column, out_name, filt=None):
    out = ds["tmp"] / out_name
    out.mkdir(exist_ok=True)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], dense_column="dense_embedding", id_column=id_column),
        queries=QueriesConfig(path=ds["qpath"], dense_column="dense_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, dense_batch_size=batch),
        searches=[SearchSpec(name="test", k=K, metric="dot", filter=filt)],
    )
    t = pq.read_table(run_compute(cfg)["test"]).to_pydict()
    return {q: list(zip(hi, hs)) for q, hi, hs in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}


def test_neighbors_match_bruteforce(ds):
    res = _run(ds, batch=None, id_column=None, out_name="nn")
    for q in ds["qids"]:
        got = sorted((s for _, s in res[q]), reverse=True)
        exp = sorted((s for _, s in ds["expected"][q]), reverse=True)
        assert np.allclose(got, exp, atol=1e-4)


@pytest.mark.parametrize("id_column", [None, "id"])
def test_tiling_is_invariant(ds, id_column):
    """Batched scoring must give identical neighbors to whole-file scoring."""
    whole = _run(ds, batch=None, id_column=id_column, out_name=f"whole_{id_column}")
    tiled = _run(ds, batch=K, id_column=id_column, out_name=f"tiled_{id_column}")  # K < file sizes → sub-file batches
    for q in ds["qids"]:
        ids_whole = [i for i, _ in whole[q]]
        ids_tiled = [i for i, _ in tiled[q]]
        assert ids_whole == ids_tiled  # identical hit ids, identical order
        sc_whole = [s for _, s in whole[q]]
        sc_tiled = [s for _, s in tiled[q]]
        assert np.allclose(sc_whole, sc_tiled, atol=1e-3)  # equal modulo float32 matmul noise


@pytest.mark.parametrize("batch", [None, K])
def test_id_column_uses_raw_ids(ds, batch):
    res = _run(ds, batch=batch, id_column="id", out_name=f"idcol_{batch}")
    for q in ds["qids"]:
        got = [i for i, _ in res[q]]
        want = [ds["ids_by_g"][g] for g, _ in ds["expected"][q]]
        assert got == want


def test_default_uses_make_point_id(ds):
    res = _run(ds, batch=None, id_column=None, out_name="hash")
    files = ds["files"]
    for q in ds["qids"]:
        got = [i for i, _ in res[q]]
        want = []
        for g, _ in ds["expected"][q]:
            fi, row = ds["loc_by_g"][g]
            want.append(make_point_id(files[fi].key, row))
        assert got == want


def test_corpus_batch_below_k_warns_and_clamps(ds, caplog):
    with caplog.at_level(logging.WARNING, logger="nova_bf.compute"):
        below = _run(ds, batch=K - 1, id_column="id", out_name="belowk")
    assert any("below k" in r.message for r in caplog.records)
    # clamped to k → still correct (identical to whole-file)
    whole = _run(ds, batch=None, id_column="id", out_name="belowk_ref")
    for q in ds["qids"]:
        assert [i for i, _ in below[q]] == [i for i, _ in whole[q]]


def _filtered_ground_truth(ds, qid, allowed_globals):
    i = ds["qids"].index(qid)
    s = ds["corpus"][allowed_globals] @ ds["queries"][i]
    return [allowed_globals[j] for j in np.argsort(-s)[:K]]


def test_filter_match_restricts_candidates(ds):
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    res = _run(ds, batch=None, id_column="id", out_name="filter_lang", filt=filt)
    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    for q in ds["qids"]:
        got = [i for i, _ in res[q]]
        assert got  # sanity: something survived the filter
        assert got == [ds["ids_by_g"][g] for g in _filtered_ground_truth(ds, q, eng_globals)]


def test_filter_range_condition(ds):
    filt = Filter(must=[FilterCondition(field="cost", range=RangeCondition(gte=5))])
    res = _run(ds, batch=None, id_column="id", out_name="filter_cost", filt=filt)
    allowed = [g for g, c in enumerate(ds["cost_by_g"]) if c >= 5]
    for q in ds["qids"]:
        got = [i for i, _ in res[q]]
        assert got == [ds["ids_by_g"][g] for g in _filtered_ground_truth(ds, q, allowed)]


@pytest.mark.parametrize("batch", [None, K])
def test_filter_preserves_row_numbers_under_batching(ds, batch):
    """A filter must resolve the same (correct) point ids whether or not
    params.dense_batch_size tiles the file — the bug this guards against is
    filtering renumbering rows instead of keeping their true file-row number."""
    filt = Filter(must=[FilterCondition(field="language", match="eng")])
    res = _run(ds, batch=batch, id_column=None, out_name=f"filter_defid_{batch}", filt=filt)
    eng_globals = [g for g, lang in enumerate(ds["lang_by_g"]) if lang == "eng"]
    files = ds["files"]
    for q in ds["qids"]:
        want = []
        for g in _filtered_ground_truth(ds, q, eng_globals):
            fi, row = ds["loc_by_g"][g]
            want.append(make_point_id(files[fi].key, row))
        assert [i for i, _ in res[q]] == want


def test_filter_timing_is_reported_only_when_filtering(ds, caplog):
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        _run(ds, batch=None, id_column="id", out_name="filter_timing_on",
             filt=Filter(must=[FilterCondition(field="language", match="eng")]))
    assert any("filter eval" in r.message for r in caplog.records)
    assert any("filter_s=" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        _run(ds, batch=None, id_column="id", out_name="filter_timing_off")
    assert not any("filter eval" in r.message for r in caplog.records)
    assert any("filter_s=0.0" in r.message for r in caplog.records)  # stable bf-bench schema


def test_bad_filter_field_raises_instead_of_hanging(ds):
    """A filter referencing a column absent from the corpus schema makes
    `evaluate()` raise inside a reader thread — this must surface as a clear
    exception in the main thread, not hang forever (an uncaught exception in
    a daemon thread silently kills it, and the consumer's fixed-count
    `fq.get()` loop would otherwise block waiting for an item that never
    arrives)."""
    filt = Filter(must=[FilterCondition(field="no_such_column", match="eng")])
    with pytest.raises(RuntimeError, match="reader thread failed"):
        _run(ds, batch=None, id_column="id", out_name="filter_bad_field", filt=filt)


def test_next_in_order_reorders_scrambled_arrivals():
    """Regression test: reader threads can finish corpus files in any order,
    but `_next_in_order` must always hand the consumer files back in a
    fixed, deterministic order (ascending `gidx`) — otherwise which of
    several EXACTLY tied candidates wins a spot in the top-K merge varies
    nondeterministically run to run, even for the identical corpus and
    queries (see `run_compute`'s consumer loop)."""
    from nova_bf.compute import _next_in_order

    # Arrives scrambled: 2, 0, 3, 1 — must still be consumable in order 0, 1, 2, 3.
    arrivals = iter([(2, "b"), (0, "a"), (3, "d"), (1, "c")])
    pending: dict = {}
    results = [_next_in_order(g, pending, lambda: next(arrivals)) for g in [0, 1, 2, 3]]
    assert results == [(0, "a"), (1, "c"), (2, "b"), (3, "d")]
    assert pending == {}  # every buffered arrival was eventually consumed


def test_next_in_order_reraises_from_fetch():
    """`fetch` raising (e.g. the consumer's `_fetch_or_raise` re-raising a
    reader thread's forwarded exception) must propagate straight through,
    not get silently swallowed while waiting for `want_gidx`'s turn."""
    from nova_bf.compute import _next_in_order

    def fetch():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _next_in_order(0, {}, fetch)


def test_slow_first_file_cannot_flood_host_memory(tmp_path, monkeypatch):
    """Regression test for the lookahead `window` semaphore in `run_compute`:
    the consumer folds files in ascending `gidx` order, so while it waits for
    a slow file 0 it drains `fq` into the unbounded `pending` dict — every
    drain frees a queue slot, so `fq`'s own bound provides NO backpressure and
    (before the window) readers could stream the ENTIRE remaining corpus,
    decoded, into host RAM during one stalled read. The window holds a permit
    per file from read-start until the consumer consumes it, so no more than
    `io_workers * 2` reads may even START while file 0 stalls.

    Also checks the stalled run's results are bit-identical to an unstalled
    one: the window changes memory behavior, never consumption order."""
    import threading
    import time

    rng = np.random.default_rng(7)
    n_files, rows, n_q, io_workers = 24, 3, 2, 4
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    for fi in range(n_files):
        _write_vectors(cdir / f"f{fi:03d}.parquet", rng.standard_normal((rows, DIM)).astype(np.float32))
    qpath = tmp_path / "queries.parquet"
    _write_vectors(qpath, rng.standard_normal((n_q, DIM)).astype(np.float32), qid=[f"q{i}" for i in range(n_q)])

    def _cfg(out_name):
        out = tmp_path / out_name
        out.mkdir()
        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), dense_column="dense_embedding"),
            queries=QueriesConfig(path=str(qpath), dense_column="dense_embedding", id_column="qid"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=io_workers),
            searches=[SearchSpec(name="test", k=K, metric="dot")],
        )

    control = pq.read_table(run_compute(_cfg("out_control"))["test"]).to_pydict()

    first_path = Store(str(cdir)).list_parquets()[0].read_path
    lock = threading.Lock()
    corpus_reads_started: list[str] = []
    started_during_stall: list[int] = []
    real_read = Store.read_columns

    def stalling_read(self, read_path, columns):
        if str(cdir) in str(read_path):
            with lock:
                corpus_reads_started.append(read_path)
        if read_path == first_path:
            time.sleep(1.0)  # ample for 23 tiny local reads — without the
            # window, every other file would start (and finish) in here
            result = real_read(self, read_path, columns)
            with lock:
                started_during_stall.append(len(corpus_reads_started) - 1)
            return result
        return real_read(self, read_path, columns)

    monkeypatch.setattr(Store, "read_columns", stalling_read)
    stalled = pq.read_table(run_compute(_cfg("out_stalled"))["test"]).to_pydict()

    # Structural bound: file 0 holds 1 of the io_workers*2 permits for the
    # whole stall, so at most io_workers*2 - 1 OTHER reads can have started.
    assert started_during_stall[0] <= io_workers * 2 - 1, (
        f"{started_during_stall[0]} reads started while file 0 stalled — "
        f"the lookahead window (io_workers*2 = {io_workers * 2}) is not binding"
    )
    assert stalled == control
