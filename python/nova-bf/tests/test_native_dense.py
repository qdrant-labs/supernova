"""Native dense read path (NOVA_BF_NATIVE_DENSE=1, `Store.read_dense`).

Contract: when it returns an array, the bytes equal `dense_to_2d` on the
pyarrow read; when it can't reproduce that exactly, it returns None and the
reader falls back to pyarrow; and a whole `run_compute` produces the same
output files with the flag on as off -- with ids, filters and a second
vector type riding along on the pyarrow side.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

rust_fetch = pytest.importorskip("rust_fetch")
pytest.importorskip("torch")

from nova_bf import io as nio
from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.io import Store, dense_to_2d


@pytest.fixture
def native_on(monkeypatch):
    monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")


def _dense_col(v, kind="list"):
    n, dim = v.shape
    vt = pa.float16() if v.dtype == np.float16 else pa.float32()
    flat = pa.array(v.reshape(-1), type=vt)
    if kind == "fixed":
        return pa.FixedSizeListArray.from_arrays(flat, dim)
    return pa.ListArray.from_arrays(pa.array(np.arange(n + 1, dtype=np.int32) * dim), flat)


def _pyarrow_ref(path, col):
    return dense_to_2d(pq.read_table(path, columns=[col])[col])


@pytest.mark.parametrize("np_type", [np.float32, np.float16])
@pytest.mark.parametrize("kind", ["list", "fixed"])
@pytest.mark.parametrize("compression", ["none", "snappy", "zstd"])
def test_read_dense_matches_dense_to_2d(tmp_path, native_on, np_type, kind, compression):
    v = np.random.default_rng(3).standard_normal((300, 10)).astype(np_type)
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"id": pa.array(range(300)), "dense_embedding": _dense_col(v, kind)}),
                   str(p), compression=compression, row_group_size=70)
    store = Store(str(tmp_path))
    f = store.list_parquets()[0]
    got = store.read_dense(f.read_path, "dense_embedding", f.size)
    ref = _pyarrow_ref(str(p), "dense_embedding")
    assert got is not None
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert got.tobytes() == ref.tobytes()
    assert got.flags.c_contiguous


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("NOVA_BF_NATIVE_DENSE", raising=False)
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"dense_embedding": _dense_col(np.ones((3, 2), np.float32))}), str(p))
    assert Store(str(tmp_path)).read_dense(str(p), "dense_embedding") is None


@pytest.mark.parametrize("rows,typ", [
    ([[1.0, 2.0], None], pa.list_(pa.float32())),            # null row
    ([[1.0, 2.0], [3.0]], pa.list_(pa.float32())),           # ragged
    ([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64())),      # widened by dense_to_2d
])
def test_unsupported_layouts_return_none(tmp_path, native_on, rows, typ):
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"dense_embedding": pa.array(rows, typ)}), str(p))
    assert Store(str(tmp_path)).read_dense(str(p), "dense_embedding") is None


def test_empty_file_matches_dense_to_2d(tmp_path, native_on):
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"dense_embedding": _dense_col(np.zeros((0, 4), np.float32))}), str(p))
    got = Store(str(tmp_path)).read_dense(str(p), "dense_embedding")
    ref = _pyarrow_ref(str(p), "dense_embedding")
    assert got.shape == ref.shape == (0, 0) and got.dtype == ref.dtype


def test_store_uri_gating(native_on):
    assert Store("s3://bucket/prefix")._native_ok
    s = Store("s3://bucket/prefix?region=us-west-2")
    assert s._native_ok and s._native_region == "us-west-2"
    assert not Store("s3://bucket/prefix?endpoint_override=localhost:9000&scheme=http")._native_ok


def test_missing_extension_falls_back(tmp_path, native_on, monkeypatch, caplog):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rust_fetch":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(nio._NATIVE, "warned", set())
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"dense_embedding": _dense_col(np.ones((3, 2), np.float32))}), str(p))
    with caplog.at_level(logging.WARNING, logger="nova_bf.io"):
        assert Store(str(tmp_path)).read_dense(str(p), "dense_embedding") is None
    assert "rust_fetch is not importable" in caplog.text


# --- whole runs: flag on == flag off ----------------------------------------


def _corpus(root, np_type, *, ragged_file=False):
    rng = np.random.default_rng(5)
    cdir = root / "c"
    cdir.mkdir(parents=True)
    g = 0
    for fi, n in enumerate((37, 1, 50, 23)):
        v = rng.standard_normal((n, 12)).astype(np_type)
        dense = _dense_col(v)
        if ragged_file and fi == 2:
            # One file the native path must refuse; the run reads it with
            # pyarrow ... which rejects ragged rows itself, so make it a
            # float64 file instead (valid, but not a native layout).
            dense = pa.array(v.astype(np.float64).tolist(), pa.list_(pa.float64()))
        sparse = pa.array(
            [{"indices": [int(i) for i in rng.choice(30, 3, replace=False)],
              "values": [float(x) for x in rng.random(3)]} for _ in range(n)],
            pa.struct([("indices", pa.list_(pa.uint32())), ("values", pa.list_(pa.float32()))]),
        )
        t = pa.table({
            "sid": pa.array([f"d{g + r}" for r in range(n)]),
            "dense_embedding": dense,
            "lang": pa.array(rng.choice(["en", "de"], n)),
            "sparse_embedding": sparse,
        })
        pq.write_table(t, str(cdir / f"f{fi}.parquet"), row_group_size=16)
        g += n
    qv = rng.standard_normal((6, 12)).astype(np.float32)
    qs = pa.array(
        [{"indices": [int(i) for i in rng.choice(30, 4, replace=False)],
          "values": [float(x) for x in rng.random(4)]} for _ in range(6)],
        pa.struct([("indices", pa.list_(pa.uint32())), ("values", pa.list_(pa.float32()))]),
    )
    qpath = root / "q.parquet"
    pq.write_table(pa.table({
        "qid": pa.array([f"q{i}" for i in range(6)]),
        "dense_embedding": _dense_col(qv),
        "sparse_embedding": qs,
        "want": pa.array(["en", "de", "en", "de", "en", "en"]),
    }), str(qpath))
    return cdir, qpath


def _run(cdir, qpath, out, searches, id_column="sid", **params):
    out.mkdir()
    run_compute(BruteForceConfig(
        corpus=CorpusConfig(path=str(cdir), id_column=id_column),
        queries=QueriesConfig(path=str(qpath), id_column="qid", payload_fields=["want"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=2, dense_batch_size=16, **params),
        searches=searches,
    ))
    return {
        p.name: pq.read_table(str(p)).to_pydict()
        for p in sorted(out.glob("*.parquet"))
    }


SEARCHES = {
    "dense_only": [SearchSpec(name="d", k=7, metric="cosine")],
    "dense_filtered": [
        SearchSpec(name="d", k=7, metric="dot"),
        SearchSpec(name="f", k=5, metric="dot",
                   filter={"must": [{"field": "lang", "match_from_query": "want"}]}),
    ],
    "dense_and_sparse": [
        SearchSpec(name="d", k=7, metric="euclidean"),
        SearchSpec(name="s", k=4, metric="dot", vector_type="sparse"),
    ],
}


@pytest.mark.parametrize("np_type", [np.float32, np.float16])
@pytest.mark.parametrize("which", list(SEARCHES))
@pytest.mark.parametrize("id_column", ["sid", None])
def test_run_outputs_identical_flag_on_vs_off(tmp_path, monkeypatch, np_type, which, id_column):
    cdir, qpath = _corpus(tmp_path, np_type)
    monkeypatch.delenv("NOVA_BF_NATIVE_DENSE", raising=False)
    off = _run(cdir, qpath, tmp_path / "off", SEARCHES[which], id_column)
    monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")
    on = _run(cdir, qpath, tmp_path / "on", SEARCHES[which], id_column)
    assert off and off == on


def test_run_with_one_fallback_file_is_identical(tmp_path, monkeypatch):
    cdir, qpath = _corpus(tmp_path, np.float32, ragged_file=True)
    monkeypatch.delenv("NOVA_BF_NATIVE_DENSE", raising=False)
    off = _run(cdir, qpath, tmp_path / "off", SEARCHES["dense_filtered"])
    monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")
    on = _run(cdir, qpath, tmp_path / "on", SEARCHES["dense_filtered"])
    assert off and off == on


def test_native_path_is_actually_taken(tmp_path, monkeypatch):
    """Guard against the equality tests passing because the flag silently
    did nothing."""
    cdir, qpath = _corpus(tmp_path, np.float32)
    monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")
    calls = []
    real = Store.read_dense_versioned

    def spy(self, *a, **k):
        r = real(self, *a, **k)
        calls.append(r is not None)
        return r

    monkeypatch.setattr(Store, "read_dense_versioned", spy)
    _run(cdir, qpath, tmp_path / "on", SEARCHES["dense_filtered"])
    assert calls and all(calls), calls


# --- hardening from the 2026-11-01 review rounds ----------------------------


def test_endpoint_override_env_keeps_pyarrow(monkeypatch, native_on):
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://localhost:1")
    assert not Store("s3://bucket/prefix")._native_ok


def test_transport_failure_disables_native_for_the_store(tmp_path, native_on, monkeypatch):
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"dense_embedding": _dense_col(np.ones((3, 2), np.float32))}), str(p))
    store = Store(str(tmp_path))

    class Boom:
        def dense_plan(self, *a, **k):
            raise OSError("simulated transport failure")

    monkeypatch.setattr(nio, "_native_client", lambda region: Boom())
    assert store.read_dense(str(p), "dense_embedding") is None
    assert store._native_failed
    monkeypatch.setattr(nio, "_native_client", lambda region: pytest.fail("native used after failure"))
    assert store.read_dense(str(p), "dense_embedding") is None


def test_stale_listing_size_is_not_trusted(tmp_path, native_on):
    v = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"dense_embedding": _dense_col(v)}), str(p))
    got = Store(str(tmp_path)).read_dense(str(p), "dense_embedding", size=1)  # wildly wrong
    assert got is not None and got.tobytes() == v.tobytes()


def test_object_rewritten_mid_read_falls_back(tmp_path, native_on, monkeypatch):
    v = np.ones((10, 4), np.float32)
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"dense_embedding": _dense_col(v)}), str(p), row_group_size=2)
    client = nio._native_client(None)
    real_plan = type(client).dense_plan

    class Rewriting:
        def dense_plan(self, *a, **k):
            plan = client.dense_plan(*a, **k)
            # Rewrite the file between planning and reading (new version).
            pq.write_table(pa.table({"dense_embedding": _dense_col(v * 2)}), str(p), row_group_size=2)
            return plan

        def read_dense_into(self, *a, **k):
            return client.read_dense_into(*a, **k)

    monkeypatch.setattr(nio, "_native_client", lambda region: Rewriting())
    assert Store(str(tmp_path)).read_dense(str(p), "dense_embedding") is None


@pytest.mark.parametrize("role", ["id", "filter"])
def test_dense_column_with_another_role_uses_pyarrow(tmp_path, monkeypatch, role):
    cdir, qpath = _corpus(tmp_path, np.float32)
    monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")
    calls = []
    monkeypatch.setattr(Store, "read_dense_versioned", lambda self, *a, **k: calls.append(1))
    searches = [SearchSpec(name="d", k=3, metric="dot")]
    with pytest.raises(Exception):  # nonsensical configs: they fail, but via pyarrow
        if role == "id":
            _run(cdir, qpath, tmp_path / "o", searches, id_column="dense_embedding")
        else:
            _run(cdir, qpath, tmp_path / "o", [SearchSpec(
                name="f", k=3, metric="dot",
                filter={"must": [{"field": "dense_embedding", "match_from_query": "want"}]})])
    assert not calls


def test_interpreter_exit_with_daemon_readers_mid_call_does_not_abort(tmp_path):
    """nova-bf's reader threads are daemon threads; without the atexit drain,
    finalizing while one is inside a native call aborts the process (rc 134)."""
    import subprocess
    import sys
    import textwrap

    v = np.random.default_rng(0).standard_normal((100_000, 32)).astype(np.float32)
    pq.write_table(pa.table({"e": _dense_col(v, "fixed")}), str(tmp_path / "f.parquet"), row_group_size=5000)
    src = str(Path(nio.__file__).resolve().parents[1])
    code = textwrap.dedent(f"""
        import os, sys, threading, time
        sys.path.insert(0, {src!r})
        os.environ["NOVA_BF_NATIVE_DENSE"] = "1"
        from nova_bf.io import Store
        st = Store({str(tmp_path)!r}); p = {str(tmp_path / "f.parquet")!r}
        def loop():
            while True:
                st.read_dense(p, "e")
        for _ in range(6):
            threading.Thread(target=loop, daemon=True).start()
        time.sleep(0.5)
    """)
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
        assert r.returncode == 0, (r.returncode, r.stderr[-500:])


def test_file_rewritten_between_native_and_side_column_reads_is_reread(tmp_path, monkeypatch):
    """Ids must never be paired with vectors from another file version."""
    cdir, qpath = _corpus(tmp_path, np.float32)
    monkeypatch.delenv("NOVA_BF_NATIVE_DENSE", raising=False)
    files = sorted(cdir.glob("*.parquet"))
    target = files[0]
    # Version B: same rows/ids, different vectors.
    t = pq.read_table(str(target))
    vb = np.random.default_rng(123).standard_normal((len(t), 12)).astype(np.float32)
    tb = t.set_column(t.schema.get_field_index("dense_embedding"), "dense_embedding", _dense_col(vb))
    real = Store.read_dense_versioned

    def rewrite_after(self, read_path, column):
        r = real(self, read_path, column)
        if read_path.endswith(target.name):
            pq.write_table(tb, str(target), row_group_size=16)
        return r

    expect_root = tmp_path / "expect"
    expect_root.mkdir()
    ec = expect_root / "c"
    ec.mkdir()
    for p in files:
        pq.write_table(tb if p == target else pq.read_table(str(p)), str(ec / p.name), row_group_size=16)
    expected = _run(ec, qpath, tmp_path / "off", SEARCHES["dense_filtered"])
    monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")
    monkeypatch.setattr(Store, "read_dense_versioned", rewrite_after)
    got = _run(cdir, qpath, tmp_path / "on", SEARCHES["dense_filtered"])
    assert got == expected


def test_drain_cancels_inflight_native_calls(tmp_path):
    """The atexit drain must not wait out a stalled native read."""
    import subprocess
    import sys
    import textwrap

    src = str(Path(nio.__file__).resolve().parents[1])
    code = textwrap.dedent(f"""
        import os, sys, threading, time
        sys.path.insert(0, {src!r})
        os.environ["NOVA_BF_NATIVE_DENSE"] = "1"
        import nova_bf.io as nio
        # A stalled 'S3' endpoint: accepts, never answers.
        import socket
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(64)
        port = srv.getsockname()[1]
        import rust_fetch
        c = rust_fetch.Client(endpoint=f"http://127.0.0.1:{{port}}", allow_http=True,
                              part_timeout_s=600, connect_timeout_s=600, region="us-east-1")
        nio._NATIVE["pid"] = os.getpid(); nio._NATIVE["clients"][None] = c
        os.environ["AWS_ACCESS_KEY_ID"] = "x"; os.environ["AWS_SECRET_ACCESS_KEY"] = "y"
        def stuck():
            nio._native_enter()
            try:
                c.dense_plan("s3://b/k.parquet", "e")
            except Exception:
                pass
            finally:
                nio._native_exit()
        threading.Thread(target=stuck, daemon=True).start()
        time.sleep(0.5)
        sys.exit(3)
    """)
    import time as _t
    t0 = _t.time()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    assert r.returncode == 3, (r.returncode, r.stderr[-500:])
    assert _t.time() - t0 < 20


def test_fork_resets_inflight_accounting():
    import os

    nio._native_enter()
    try:
        pid = os.fork()
        if pid == 0:
            ok = nio._INFLIGHT["n"] == 0
            os._exit(0 if ok else 1)
        _, st = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(st) == 0
    finally:
        nio._native_exit()


@pytest.mark.parametrize("name,sub,native", [
    ("__filename", "plain", False),   # pyarrow's dataset reserves __-fields
    ("e", "e=7", False),              # hive segments: pyarrow parses ANY k=v
    ("e", "x=1", False),              # directory into a partition field
    ("emb", "e=7", False),
    ("e", "%65=5", False),
    ("e", "k=%FF", False),
    ("e", "plain_dir", True),
])
def test_dataset_special_names_keep_pyarrow(tmp_path, native_on, name, sub, native):
    d = tmp_path / sub
    d.mkdir()
    p = d / f"{name}.parquet"
    pq.write_table(pa.table({name: _dense_col(np.ones((3, 4), np.float32), "fixed")}), str(p))
    got = Store(str(d)).read_dense(str(p), name)
    assert (got is not None) == native


def test_equals_sign_in_the_file_name_is_not_a_partition(tmp_path, native_on):
    p = tmp_path / "e=5.parquet"
    pq.write_table(pa.table({"e": _dense_col(np.ones((3, 4), np.float32), "fixed")}), str(p))
    assert Store(str(tmp_path)).read_dense(str(p), "e") is not None


# --- native DOWNLOAD of pyarrow-decoded columns (NOVA_BF_NATIVE_FETCH) -------


@pytest.fixture
def fetch_on(monkeypatch):
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH", "1")
    # Fill unfetched bytes with garbage: any read outside the fetched ranges
    # then breaks equality (or decoding) instead of silently reading zeros.
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH_CANARY", "1")
    # Small test files would otherwise arrive whole in the speculative tail
    # read; shrink it and stop coalescing so the projected ranges are what
    # actually gets exercised (gaps between them stay canary garbage).
    monkeypatch.setattr(nio, "_FETCH_TAIL", 64)
    monkeypatch.setattr(nio, "_FETCH_COALESCE", 0)
    monkeypatch.setattr(nio, "_FETCH_PAD", 0)


def _mixed_table(n, rng):
    sparse = pa.array(
        [{"indices": [int(i) for i in rng.choice(50, 3, replace=False)],
          "values": [float(x) for x in rng.random(3)]} for _ in range(n)],
        pa.struct([("indices", pa.list_(pa.uint32())), ("values", pa.list_(pa.float32()))]),
    )
    return pa.table({
        "id": pa.array([f"doc-{i}" for i in range(n)]),
        "text": pa.array([" ".join(rng.choice(["alpha", "beta", "gamma", "δέλτα"], 5)) for _ in range(n)]),
        "dense_embedding": _dense_col(rng.standard_normal((n, 6)).astype(np.float16)),
        "sparse_embedding": sparse,
        "score": pa.array(rng.random(n)),
        "lang": pa.array(rng.choice(["en", "de"], n)).dictionary_encode(),
        "tags": pa.array([["a", "b"][: i % 3] for i in range(n)], pa.list_(pa.string())),
        "kv": pa.array([[("k", i)] for i in range(n)], pa.map_(pa.string(), pa.int64())),
        "maybe": pa.array([None if i % 5 == 0 else i for i in range(n)], pa.int32()),
    })


WRITE_OPTS = [
    {},
    {"compression": "zstd", "row_group_size": 17},
    {"compression": "gzip", "use_dictionary": False, "data_page_version": "2.0"},
    {"compression": "lz4", "data_page_size": 128, "write_page_index": True},
    {"compression": "snappy", "write_page_checksum": True, "row_group_size": 5},
    {"compression": "brotli", "column_encoding": {"score": "BYTE_STREAM_SPLIT"}, "use_dictionary": ["id"]},
    {"store_schema": False},
]


@pytest.mark.parametrize("opts", WRITE_OPTS, ids=[str(i) for i in range(len(WRITE_OPTS))])
@pytest.mark.parametrize("cols", [
    ["id"], ["text", "sparse_embedding"], ["kv", "tags", "maybe", "lang"],
    ["id", "text", "sparse_embedding", "score", "lang", "tags", "kv", "maybe", "dense_embedding"],
])
def test_fetched_columns_equal_pyarrow_path_read(tmp_path, fetch_on, opts, cols):
    rng = np.random.default_rng(7)
    p = tmp_path / "f.parquet"
    pq.write_table(_mixed_table(60, rng), str(p), **opts)
    store = Store(str(tmp_path))
    ref = store.read_columns(str(p), cols)
    got = store.read_columns_fetched(str(p), cols)
    assert got is not None
    assert got[0].equals(ref) and got[0].schema.equals(ref.schema, check_metadata=True)


def test_fetch_downloads_only_the_requested_columns(tmp_path, fetch_on, monkeypatch):
    rng = np.random.default_rng(1)
    t = _mixed_table(3000, rng)
    p = tmp_path / "f.parquet"
    pq.write_table(t, str(p), row_group_size=1000)
    client = nio._native_client(None)
    fetched_bytes = []
    real = type(client).read_ranges_into

    class Spy:
        def head(self, url):
            return client.head(url)

        def read_ranges_into(self, url, dest, ranges, if_match=None):
            fetched_bytes.append(sum(r[1] for r in ranges))
            return client.read_ranges_into(url, dest, ranges, if_match=if_match)

    monkeypatch.setattr(nio, "_native_client", lambda region: Spy())
    monkeypatch.setattr(nio, "_FETCH_TAIL", 8192)  # tail smaller than the file
    monkeypatch.setattr(nio, "_FETCH_COALESCE", 0)
    got = Store(str(tmp_path)).read_columns_fetched(str(p), ["id"])
    assert got is not None and got[0].equals(pq.read_table(str(p), columns=["id"]))
    total = os.path.getsize(p)
    id_bytes = sum(pq.read_metadata(str(p)).row_group(i).column(0).total_compressed_size
                   for i in range(3))
    assert sum(fetched_bytes) < id_bytes + 8192 + 3 * nio._FETCH_PAD + 4096
    assert sum(fetched_bytes) < total / 2


def test_fetch_falls_back_when_pinned_version_is_gone(tmp_path, fetch_on):
    p = tmp_path / "f.parquet"
    pq.write_table(_mixed_table(10, np.random.default_rng(0)), str(p))
    assert Store(str(tmp_path)).read_columns_fetched(str(p), ["id"], etag="\"not-it\"") is None


@pytest.mark.parametrize("dense_flag", [False, True])
@pytest.mark.parametrize("which", list(SEARCHES))
def test_run_outputs_identical_with_native_fetch(tmp_path, monkeypatch, which, dense_flag):
    cdir, qpath = _corpus(tmp_path, np.float16)
    monkeypatch.delenv("NOVA_BF_NATIVE_DENSE", raising=False)
    monkeypatch.delenv("NOVA_BF_NATIVE_FETCH", raising=False)
    off = _run(cdir, qpath, tmp_path / "off", SEARCHES[which])
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH", "1")
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH_CANARY", "1")
    monkeypatch.setattr(nio, "_FETCH_TAIL", 64)
    monkeypatch.setattr(nio, "_FETCH_COALESCE", 0)
    if dense_flag:
        monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")
    calls = []
    real = Store.read_columns_fetched

    def spy(self, *a, **k):
        r = real(self, *a, **k)
        calls.append(r is not None)
        return r

    monkeypatch.setattr(Store, "read_columns_fetched", spy)
    on = _run(cdir, qpath, tmp_path / "on", SEARCHES[which])
    assert off and off == on
    assert calls and all(calls), calls


@pytest.mark.parametrize("dense_flag", [False, True])
def test_id_pass_uses_native_fetch_and_outputs_match(tmp_path, monkeypatch, dense_flag):
    """tiebreak='id': the startup id pass reads the id column through the
    native download (one call per corpus file) and nothing changes."""
    cdir, qpath = _corpus(tmp_path, np.float32)
    n_files = len(list(cdir.glob("*.parquet")))
    monkeypatch.delenv("NOVA_BF_NATIVE_DENSE", raising=False)
    monkeypatch.delenv("NOVA_BF_NATIVE_FETCH", raising=False)
    off = _run(cdir, qpath, tmp_path / "off", SEARCHES["dense_filtered"], tiebreak="id")
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH", "1")
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH_CANARY", "1")
    monkeypatch.setattr(nio, "_FETCH_TAIL", 64)
    monkeypatch.setattr(nio, "_FETCH_COALESCE", 0)
    if dense_flag:
        monkeypatch.setenv("NOVA_BF_NATIVE_DENSE", "1")
    id_calls = []
    real = Store.read_columns_fetched

    def spy(self, read_path, columns, *a, **k):
        r = real(self, read_path, columns, *a, **k)
        if list(columns) == ["sid"]:
            id_calls.append(r is not None)
        return r

    monkeypatch.setattr(Store, "read_columns_fetched", spy)
    on = _run(cdir, qpath, tmp_path / "on", SEARCHES["dense_filtered"], tiebreak="id")
    assert off and off == on
    assert len(id_calls) == n_files and all(id_calls), id_calls


def test_id_pass_falls_back_to_pyarrow_when_fetch_declines(tmp_path, monkeypatch):
    cdir, qpath = _corpus(tmp_path, np.float32)
    monkeypatch.delenv("NOVA_BF_NATIVE_DENSE", raising=False)
    monkeypatch.delenv("NOVA_BF_NATIVE_FETCH", raising=False)
    off = _run(cdir, qpath, tmp_path / "off", SEARCHES["dense_only"], tiebreak="id")
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH", "1")
    monkeypatch.setattr(Store, "read_columns_fetched", lambda self, *a, **k: None)
    on = _run(cdir, qpath, tmp_path / "on", SEARCHES["dense_only"], tiebreak="id")
    assert off and off == on


def test_fetch_declines_files_with_duplicate_top_level_names(tmp_path, fetch_on):
    """The path read's dataset layer refuses duplicate names (even unread
    ones); a buffer read would not -- so the fetch must defer, keeping
    pass/fail identical."""
    p = tmp_path / "dup.parquet"
    t = pa.Table.from_arrays([pa.array(["a", "b"]), pa.array([1, 2]), pa.array([3, 4])],
                             names=["id", "x", "x"])
    pq.write_table(t, str(p))
    store = Store(str(tmp_path))
    with pytest.raises(Exception):
        store.read_columns(str(p), ["id"])
    assert store.read_columns_fetched(str(p), ["id"]) is None


def test_missing_column_defers_without_downloading_the_file(tmp_path, fetch_on, monkeypatch):
    p = tmp_path / "f.parquet"
    pq.write_table(_mixed_table(2000, np.random.default_rng(3)), str(p), row_group_size=500)
    client = nio._native_client(None)
    fetched = []

    class Spy:
        def head(self, url):
            return client.head(url)

        def read_ranges_into(self, url, dest, ranges, if_match=None):
            fetched.append(sum(r[1] for r in ranges))
            return client.read_ranges_into(url, dest, ranges, if_match=if_match)

    monkeypatch.setattr(nio, "_native_client", lambda region: Spy())
    assert Store(str(tmp_path)).read_columns_fetched(str(p), ["nope"]) is None
    footer = pq.read_metadata(str(p)).serialized_size
    assert sum(fetched) <= 64 + footer + 8  # tail + footer only, never the file
    assert sum(fetched) < os.path.getsize(p) / 4


def test_a_short_range_fails_loudly_instead_of_reading_stale_memory(tmp_path, monkeypatch):
    """No canary here (production): unfetched bytes must still never decode
    as data -- e.g. a previous file's bytes left in recycled heap memory."""
    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH", "1")
    monkeypatch.setattr(nio, "_FETCH_TAIL", 64)
    monkeypatch.setattr(nio, "_FETCH_COALESCE", 0)
    rng = np.random.default_rng(5)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    ids = lambda pfx: pa.array([f"{pfx}{i:07d}" for i in range(20000)])  # noqa: E731
    pq.write_table(pa.table({"id": ids("A"), "x": pa.array(rng.random(20000))}), str(a))
    pq.write_table(pa.table({"id": ids("B"), "x": pa.array(rng.random(20000))}), str(b))
    store = Store(str(tmp_path))
    assert store.read_columns_fetched(str(a), ["id"]) is not None   # leaves A's bytes behind
    real = nio._column_ranges
    monkeypatch.setattr(nio, "_column_ranges",
                        lambda *a, **k: (lambda r: r[:-1] if r else r)(real(*a, **k)))  # drop one range
    got = store.read_columns_fetched(str(b), ["id"])
    assert got is None or got[0].column("id")[0].as_py().startswith("B")


def test_narrow_fetch_keeps_only_fetched_pages_resident(tmp_path, monkeypatch):
    """A narrow read of a big file must not keep the file resident: no 2 MiB
    huge pages faulted in by a few KB of writes, and the image's pages are
    returned once pyarrow has decoded (only the keepalive object remains)."""
    import gc

    monkeypatch.setenv("NOVA_BF_NATIVE_FETCH", "1")
    n, dim = 24000, 256
    rng = np.random.default_rng(0)
    vec = pa.FixedSizeListArray.from_arrays(pa.array(rng.standard_normal(n * dim).astype(np.float32)), dim)
    p = tmp_path / "big.parquet"
    pq.write_table(pa.table({"id": pa.array([f"d{i}" for i in range(n)]), "vec": vec}), str(p),
                   row_group_size=500, compression="none")
    del vec
    gc.collect()

    def rss_mb():
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024

    store = Store(str(tmp_path))
    store.read_columns_fetched(str(p), ["id"])   # warm the client / imports
    nio._FETCH_KEEPALIVE.clear()
    gc.collect()
    before = rss_mb()
    got = store.read_columns_fetched(str(p), ["id"])
    grew = rss_mb() - before
    assert got is not None and got[0].equals(pq.read_table(str(p), columns=["id"]))
    assert os.path.getsize(p) > 20 * 1024 * 1024
    assert grew < 8, f"narrow fetch of a {os.path.getsize(p) >> 20} MiB file kept {grew:.0f} MiB resident"
