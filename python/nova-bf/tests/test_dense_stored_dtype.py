"""R3: the dense corpus keeps its stored width on the host and is widened on
the device.

`io.dense_to_2d` used to upcast float16 to float32 in the reader thread — a
large extra allocation per corpus file — and then send twice
the bytes over PCIe. Now the host array carries the parquet's dtype and
`DenseCorpusBatch.transfer` casts on the GPU.

Two places, and only two, are allowed to know the stored dtype. These tests
pin that: everything below `transfer` sees float32, the scores do not move
(fp16 -> fp32 is exact), `compact` preserves the dtype, `_concat_dense_batches`
widens a mixed-dtype group exactly and uniformly, and the run's reported
throughput/byte counters stay on the float32-equivalent width so they are
comparable across corpora stored at different widths.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import compute as C
from nova_bf.io import dense_to_2d

torch = pytest.importorskip("torch")

from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)


def _col(arr, pa_type, fixed=True):
    n, dim = arr.shape
    flat = pa.array(arr.reshape(-1), type=pa_type)
    typ = (pa.list_(pa_type, dim) if fixed else pa.list_(pa_type))
    if fixed:
        col = pa.FixedSizeListArray.from_arrays(flat, dim)
    else:
        col = pa.ListArray.from_arrays(
            pa.array(np.arange(n + 1) * dim, type=pa.int32()), flat)
    return pa.chunked_array([col])


@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("pa_type,np_type", [(pa.float16(), np.float16),
                                             (pa.float32(), np.float32)])
def test_dense_to_2d_returns_the_stored_dtype(pa_type, np_type, fixed):
    a = (np.random.default_rng(0).standard_normal((7, 4))).astype(np_type)
    got = dense_to_2d(_col(a, pa_type, fixed))
    assert got.dtype == np_type
    np.testing.assert_array_equal(got, a)


def test_dense_to_2d_still_widens_a_dtype_that_is_not_worth_carrying():
    a = np.random.default_rng(1).standard_normal((5, 3))
    got = dense_to_2d(_col(a, pa.float64()))
    assert got.dtype == np.float32


@pytest.mark.parametrize("np_type", [np.float16, np.float32])
def test_transfer_always_hands_back_float32(np_type):
    a = np.ascontiguousarray(
        np.random.default_rng(2).standard_normal((16, 8)).astype(np_type))
    sl = C.DenseCorpusBatch(a).transfer(2, 11, "cpu")
    assert sl.Cb.dtype is torch.float32
    np.testing.assert_array_equal(sl.Cb.numpy(), a[2:11].astype(np.float32))


def test_the_widening_moves_no_score_bit():
    """The whole exactness claim: an fp16 host array widened on the device is
    bit-identical to the fp32 array the host used to build."""
    rng = np.random.default_rng(3)
    a16 = np.ascontiguousarray(rng.standard_normal((64, 32)).astype(np.float16))
    a32 = a16.astype(np.float32)
    Q = torch.from_numpy(rng.standard_normal((5, 32)).astype(np.float32))
    lo = C.DenseCorpusBatch(a16).transfer(0, 64, "cpu")
    hi = C.DenseCorpusBatch(a32).transfer(0, 64, "cpu")
    for metric in ("cosine", "dot", "euclidean"):
        qn = Q.norm(dim=1)
        s_lo = lo.score(Q, metric, qn)
        s_hi = hi.score(Q, metric, qn)
        assert torch.equal(s_lo, s_hi), metric


@pytest.mark.parametrize("np_type", [np.float16, np.float32])
def test_compact_preserves_the_stored_dtype(np_type):
    a = np.ascontiguousarray(
        np.random.default_rng(4).standard_normal((10, 3)).astype(np_type))
    keep = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1], dtype=bool)
    b, orig = C.DenseCorpusBatch(a).compact(keep)
    assert b.arr.dtype == np_type
    np.testing.assert_array_equal(orig, np.flatnonzero(keep))
    np.testing.assert_array_equal(b.arr, a[keep])


@pytest.mark.parametrize("np_type", [np.float16, np.float32])
def test_concat_preserves_the_stored_dtype(np_type):
    rng = np.random.default_rng(5)
    parts = [C.DenseCorpusBatch(np.ascontiguousarray(
        rng.standard_normal((n, 3)).astype(np_type))) for n in (2, 5, 1)]
    out = C._concat_dense_batches(parts)
    assert out.arr.dtype == np_type
    assert out.n_rows == 8


def test_concat_widens_a_mixed_dtype_group(monkeypatch, caplog):
    """A corpus whose files disagree on stored width must still run. The group
    is widened to one common dtype (`np.result_type`) — exactly, since fp16 ->
    fp32 loses nothing — so which files happened to coalesce together cannot
    move a score. The only visible trace is one INFO line per run."""
    monkeypatch.setattr(C, "_MIXED_DENSE_DTYPE_LOGGED", False)
    rng = np.random.default_rng(8)
    a16 = np.ascontiguousarray(rng.standard_normal((2, 3)).astype(np.float16))
    b32 = np.ascontiguousarray(rng.standard_normal((2, 3)).astype(np.float32))

    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        out = C._concat_dense_batches(
            [C.DenseCorpusBatch(a16), C.DenseCorpusBatch(b32)])
        # the notice is once per run, not once per coalesced group
        C._concat_dense_batches(
            [C.DenseCorpusBatch(a16), C.DenseCorpusBatch(b32)])

    assert out.arr.dtype == np.float32
    assert out.n_rows == 4
    # the widening is exact: every row is bit-identical to widening it alone
    np.testing.assert_array_equal(out.arr[:2], a16.astype(np.float32))
    np.testing.assert_array_equal(out.arr[2:], b32)
    notices = [r for r in caplog.records if "disagree on the stored dense" in r.message]
    assert len(notices) == 1, [r.message for r in caplog.records]


def _write_dense(path, vectors, pa_type, **columns):
    data = {"dense_embedding": pa.array(vectors.tolist(), type=pa.list_(pa_type))}
    data.update({k: pa.array(v) for k, v in columns.items()})
    pq.write_table(pa.table(data), str(path))


def _corpus_at(root, pa_type, np_type):
    """The same corpus values, written at one stored width. Queries stay fp32."""
    rng = np.random.default_rng(11)
    cdir = root / "c"
    cdir.mkdir(parents=True)
    g = 0
    for fi, n in enumerate((5, 4)):
        rows = rng.standard_normal((n, 6)).astype(np_type)
        _write_dense(cdir / f"f{fi}.parquet", rows, pa_type,
                     id=[f"c{g + r}" for r in range(n)])
        g += n
    qpath = root / "q.parquet"
    _write_dense(qpath, rng.standard_normal((4, 6)).astype(np.float32),
                 pa.float32(), qid=[f"q{i}" for i in range(4)])
    return cdir, qpath


def test_reported_decoded_bytes_do_not_depend_on_the_stored_width(tmp_path):
    """`bytes_seen` feeds `wall_mbps`, `stream_mbps` and the manifest's
    `corpus_bytes_decoded`. Since R3 the host array is fp16 on an fp16 corpus,
    so `DenseCorpusBatch.nbytes` (the honest host footprint) halves — but the
    reported counter must not, or every throughput number silently halves and
    stops being comparable with any earlier run or any fp32 corpus."""
    counts = {}
    for name, pa_type, np_type in (("h", pa.float16(), np.float16),
                                   ("f", pa.float32(), np.float32)):
        root = tmp_path / name
        root.mkdir()
        cdir, qpath = _corpus_at(root, pa_type, np_type)
        out = root / "out"
        out.mkdir()
        run_compute(BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="qid"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1),
            searches=[SearchSpec(name="plain", metric="dot", k=3)],
        ))
        doc = json.loads(
            (out / "_bf_manifest_q_compute.json").read_text()
            if (out / "_bf_manifest_q_compute.json").exists()
            else next(out.glob("_bf_manifest*compute.json")).read_text())
        counts[name] = doc["counts"]["corpus_bytes_decoded"]

    assert counts["h"] == counts["f"], counts
    # and it is the float32-equivalent count: 9 rows x 6 dims x 4 bytes
    assert counts["f"] == 9 * 6 * 4
