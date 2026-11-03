"""The `tiebreak='id'` startup pass must not buy whole files for one column.

WHAT THIS GUARDS. `io_ranged_get` makes `read_columns` download the ENTIRE file
into a buffer and then parse the requested columns out of it. That is the right
trade for the vector scan, which wants most of a file's bytes. The id pass wants
one narrow column -- ~180 KB of a ~4.3 GB pubmed shard -- so taking the same
path bought 4.3 GB to keep 0.004% of it.

Measured on the 2026-09-22 pubmed GT run, at the pass's 32-way default:
  * ~137 GB of resident whole-file buffers, which Ray OOM-killed on all 16
    ranks at `0/500` -- before a single corpus vector had been scored; and
  * with the fan-out throttled to 4 to survive that, 1055 s per rank spent
    re-reading 2.15 TB of corpus for 1.5 M ids, with the GPU idle throughout,
    because `_await_ordinals()` joins the pass before the first ordinal fold.

So this is a memory AND a throughput regression guard, and neither shows up in
the pass's own output -- the ids are identical either way. Only the byte volume
differs, which is why these assert on HOW the read was issued rather than on
results.

WHY BOTH ASSERTIONS. Counting `_ranged_download` alone proves only that a code
path was not taken; it would still pass if a future edit dropped `columns` and
read whole files through the ordinary reader. So the id-pass read is also
checked to be a genuine single-column projection. The byte saving that buys is
a property of parquet, not of this repo: one contiguous column chunk per column
per row group, so a projected read fetches the footer plus that one chunk --
verified by strace on a one-row-group file as 3 preads / 1.24 MB against the
whole 104 MB. One row group costs PARALLELISM (see the `io.py` module comment),
not selectivity.
"""
from __future__ import annotations

import importlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf import io as io_mod
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, OutputConfig, ParamsConfig, QueriesConfig,
    SearchSpec,
)

DIM = 8
ID_COL = "sid"



@pytest.fixture(autouse=True)
def _pyarrow_read_path(monkeypatch):
    """These tests classify Store.read_columns calls; the native dense path
    (NOVA_BF_RUST_FETCH_DENSE) adds a projected id-column read of its own that
    looks exactly like an id-pass read. Pin the pyarrow path."""
    monkeypatch.setenv("NOVA_BF_RUST_FETCH_DENSE", "0")
    monkeypatch.setenv("NOVA_BF_RUST_FETCH_COLUMNS", "0")

@pytest.fixture
def corpus_and_queries(tmp_path):
    """Several shards, so the pass has more than one file to fan out over."""
    rng = np.random.default_rng(7)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    for f in range(3):
        n = 16
        pq.write_table(pa.table({
            "embedding": pa.array(rng.random((n, DIM)).tolist(),
                                  pa.list_(pa.float32())),
            ID_COL: pa.array([f"s{f}-{i:03d}" for i in range(n)]),
        }), str(cdir / f"part{f}.parquet"))

    qpath = tmp_path / "queries.parquet"
    pq.write_table(pa.table({
        "embedding": pa.array(rng.random((4, DIM)).tolist(),
                              pa.list_(pa.float32())),
        "qid": pa.array([f"q{i}" for i in range(4)]),
    }), str(qpath))
    return cdir, qpath


def _cfg(cdir, qpath, out, tiebreak="id"):
    """Identical every way but `tiebreak`, so a comparison isolates the pass.

    `id_column` stays set even for the `ordinal` baseline: the scan reads it
    for `hit_ids` either way, so leaving it on keeps the two runs differing in
    exactly one variable.
    """
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), dense_column="embedding",
                            id_column=ID_COL),
        queries=QueriesConfig(path=str(qpath), dense_column="embedding",
                              id_column="qid"),
        output=OutputConfig(path=str(out)),
        # `io_ranged_get` ON is the whole point: that is the setting under
        # which the pass used to buy whole files.
        params=ParamsConfig(io_workers=1, tiebreak=tiebreak,
                            io_ranged_get=True),
        searches=[SearchSpec(name="d", k=4, metric="dot", vector_type="dense")],
    )


class _Spy:
    """Every `read_columns` call and every whole-file download, in order."""

    def __init__(self):
        self.reads: list[tuple] = []        # (columns, ranged)
        self.downloads: list[int] = []      # sizes

    def clear(self):
        self.reads.clear()
        self.downloads.clear()

    @property
    def id_pass_reads(self):
        """Reads of the id column ALONE -- uniquely the startup pass; the scan
        always reads it bundled with a vector column."""
        return [r for r in self.reads if r[0] == (ID_COL,)]


@pytest.fixture
def spy(monkeypatch):
    s = _Spy()
    real_read = io_mod.Store.read_columns
    real_dl = io_mod.Store._ranged_download

    def read_columns(self, read_path, columns, ranged=None):
        s.reads.append((tuple(columns) if columns is not None else None, ranged))
        return real_read(self, read_path, columns, ranged)

    def ranged_download(self, read_path, size):
        s.downloads.append(size)
        return real_dl(self, read_path, size)

    monkeypatch.setattr(io_mod.Store, "read_columns", read_columns)
    monkeypatch.setattr(io_mod.Store, "_ranged_download", ranged_download)
    # Tiny fixtures have to qualify for the ranged path at all.
    monkeypatch.setattr(io_mod, "_RANGED_GET_MIN_BYTES", 0)
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 4096)
    return s


def _reloaded_compute(monkeypatch, ranged_env: str | None):
    """`ID_PASS_RANGED_GET` is read at import, so the env var needs a reload."""
    if ranged_env is None:
        monkeypatch.delenv("NOVA_BF_ID_PASS_RANGED_GET", raising=False)
    else:
        monkeypatch.setenv("NOVA_BF_ID_PASS_RANGED_GET", ranged_env)
    from nova_bf import compute as compute_mod
    return importlib.reload(compute_mod)


@pytest.fixture(autouse=True)
def _restore_compute_module(monkeypatch):
    """Leave `nova_bf.compute` back on its default for every other test."""
    yield
    monkeypatch.delenv("NOVA_BF_ID_PASS_RANGED_GET", raising=False)
    from nova_bf import compute as compute_mod
    importlib.reload(compute_mod)


def test_the_id_pass_adds_no_whole_file_downloads(corpus_and_queries, tmp_path,
                                                  spy, monkeypatch):
    """The scan downloads its own files, so this compares against a baseline
    that differs ONLY in `tiebreak`: the id pass must add none of its own."""
    cdir, qpath = corpus_and_queries
    compute_mod = _reloaded_compute(monkeypatch, None)

    spy.clear()
    compute_mod.run_compute(_cfg(cdir, qpath, tmp_path / "o1",
                                 tiebreak="ordinal"))
    n_scan_only = len(spy.downloads)
    assert not spy.id_pass_reads, "the baseline ran an id pass; it should not"

    spy.clear()
    compute_mod.run_compute(_cfg(cdir, qpath, tmp_path / "o2"))

    assert len(spy.downloads) == n_scan_only, (
        f"the id pass added {len(spy.downloads) - n_scan_only} whole-file "
        f"download(s) over the {n_scan_only} the scan makes on its own"
    )


def test_the_id_pass_reads_the_id_column_alone(corpus_and_queries, tmp_path,
                                               spy, monkeypatch):
    """The other half of the guard, and the one a download count cannot give.

    Zero whole-file downloads would ALSO be satisfied by reading whole files
    through the ordinary reader with `columns=None`. What makes the read cheap
    is that it projects a single column, so assert that directly.
    """
    cdir, qpath = corpus_and_queries
    compute_mod = _reloaded_compute(monkeypatch, None)

    spy.clear()
    compute_mod.run_compute(_cfg(cdir, qpath, tmp_path / "o"))

    id_reads = spy.id_pass_reads
    assert len(id_reads) == 3, (
        f"expected one projected id read per corpus file, got {len(id_reads)}"
    )
    assert all(ranged is False for _, ranged in id_reads), (
        f"the id pass did not opt out of ranged-GET: {id_reads}"
    )
    assert not any(cols is None for cols, _ in spy.reads if cols is None), \
        "something read every column"


def test_the_env_var_restores_the_old_behaviour(corpus_and_queries, tmp_path,
                                                spy, monkeypatch):
    """`NOVA_BF_ID_PASS_RANGED_GET=1` is the documented escape hatch, for a
    store where one big parallel download beats a narrow seek.

    It restores DEFERENCE to the store (`ranged=None`), it does not force the
    path on -- so with `io_ranged_get: false` setting it changes nothing. Here
    the store has ranged-GET on, so deferring puts the pass back on it.
    """
    cdir, qpath = corpus_and_queries

    compute_mod = _reloaded_compute(monkeypatch, None)
    spy.clear()
    compute_mod.run_compute(_cfg(cdir, qpath, tmp_path / "off"))
    n_off = len(spy.downloads)

    compute_mod = _reloaded_compute(monkeypatch, "1")
    spy.clear()
    compute_mod.run_compute(_cfg(cdir, qpath, tmp_path / "on"))

    assert len(spy.downloads) > n_off, "the env var did not re-enable ranged reads"
    assert all(ranged is None for _, ranged in spy.id_pass_reads), \
        "the env var should defer to the store, not force `ranged=True`"


def test_the_ids_are_identical_either_way(corpus_and_queries, tmp_path,
                                          monkeypatch):
    """Only the byte volume changes; hit_ids must be identical."""
    cdir, qpath = corpus_and_queries

    compute_mod = _reloaded_compute(monkeypatch, None)
    a = compute_mod.run_compute(_cfg(cdir, qpath, tmp_path / "a"))

    compute_mod = _reloaded_compute(monkeypatch, "1")
    b = compute_mod.run_compute(_cfg(cdir, qpath, tmp_path / "b"))

    def _ids(res):
        # `run_compute` returns {search_name: output_path}.
        path = res["d"] if isinstance(res, dict) else res
        return pq.read_table(str(path)).column("hit_ids").to_pylist()

    assert _ids(a) == _ids(b)
