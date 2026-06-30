"""Correctness tests for the brute-force compute phase.

Exercises the GPU topk path on CPU (small synthetic corpus + queries), covering
the two pieces most prone to silent corruption:
  - corpus_batch_size tiling: must yield the *same* top-K as scoring the whole
    file at once (the `gidx * MAX_ROWS_PER_FILE + row` encoding uses file-local
    row offsets, so an off-by-a-batch bug would scramble hit ids), and
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
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
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

    corpus, ids_by_g, loc_by_g, g = [], [], [], 0
    for fi, n in enumerate(SIZES):
        rows = rng.standard_normal((n, DIM)).astype(np.float32)
        _write_vectors(cdir / f"f{fi}.parquet", rows, id=[f"c{g + r}" for r in range(n)])
        corpus.append(rows)
        ids_by_g += [f"c{g + r}" for r in range(n)]
        loc_by_g += [(fi, r) for r in range(n)]  # global idx → (file, file-local row)
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
        "expected": expected,
        "files": Store(str(cdir)).list_parquets(),  # same sorted order as run_compute's gidx
    }


def _run(ds, *, batch, id_column, out_name):
    out = ds["tmp"] / out_name
    out.mkdir(exist_ok=True)
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], dense_column="dense_embedding", id_column=id_column),
        queries=QueriesConfig(path=ds["qpath"], dense_column="dense_embedding", id_column="qid"),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(k=K, metric="dot", corpus_batch_size=batch, io_workers=2),
    )
    t = pq.read_table(run_compute(cfg)).to_pydict()
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
