"""Hand-crafted (adversarial) parquet files for the dense fast path, from the
2026-11-01 adversarial review. `_pqw.py` writes thrift by hand so headers,
footers and schemas can say things no normal writer would.

Rule under test: the native decoder may FAIL (UnsupportedLayout /
CorruptParquet) where pyarrow succeeds, but it must never return data that
differs from pyarrow, never hang, never write outside its destination.
"""

import os
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pyarrow.parquet as pq
import pytest

import rust_fetch
from rust_fetch import CorruptParquet, UnsupportedLayout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pqw import (  # noqa: E402
    FLBA, FLOAT, I32, OPT, REP, REQ, S, STD_SCHEMA, STRUCT, def_levels_rle, dict_page, packed_run,
    page_header, rep_levels_rle, rle_run, selem, uvar, v1_page, write_file,
)

DIM = 4


@pytest.fixture(scope="module")
def client():
    return rust_fetch.Client()


def _ref(p):
    col = pq.read_table(p, columns=["e"])["e"].combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1)


def _native_never_differs(client, p):
    """Native either fails cleanly or matches pyarrow byte-for-byte."""
    try:
        ref = _ref(p)
    except Exception:
        ref = None
    try:
        got = rust_fetch.read_dense(client, p, "e")
    except (UnsupportedLayout, CorruptParquet):
        return "fallback"
    assert ref is not None, "native returned data for a file pyarrow rejects"
    assert got.dtype == ref.dtype and got.shape == ref.shape and got.tobytes() == ref.tobytes()
    return "identical"


def _f32(a):
    return np.asarray(a, np.float32).tobytes()


def test_wrapping_binary_length_in_page_header_does_not_hang(tmp_path):
    # Field 1 typed binary with a length that wraps the cursor back to 0.
    p = str(tmp_path / "hang.parquet")
    hdr = bytes([0x18]) + uvar((1 << 64) - 11)
    write_file(p, STD_SCHEMA, [dict(pages=hdr + b"\x00" * 16, n_rows=2, num_values=8)])
    code = textwrap.dedent(f"""
        import rust_fetch
        try:
            rust_fetch.read_dense(rust_fetch.Client(), {p!r}, "e")
        except (rust_fetch.CorruptParquet, rust_fetch.UnsupportedLayout):
            print("clean error")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert "clean error" in r.stdout, r.stderr


def test_footer_counts_that_overflow_the_output_size_are_rejected_at_plan_time(tmp_path, client):
    p = str(tmp_path / "ovf.parquet")
    dpg = dict_page(_f32([1.0]), 1)
    n0 = 1000
    pg0 = v1_page(bytes([0]) + rle_run(0, n0, 0), n0, rle_run(0, n0, 1), rle_run(3, n0, 2), encoding=8, maxdef=3)
    huge = (1 << 62) - 4
    pg1 = v1_page(_f32(np.full(8, 2.0)), 8, rle_run(0, 8, 1), rle_run(3, 8, 2), maxdef=3)
    write_file(p, STD_SCHEMA, [dict(pages=dpg + pg0, n_rows=huge, num_values=huge, has_dict=True, dict_len=len(dpg)),
                               dict(pages=pg1, n_rows=8, num_values=8)])
    with pytest.raises(CorruptParquet):
        client.dense_plan(p, "e")


def _dict_pages():
    dpg = dict_page(_f32([10, 20, 30, 40]), 4)
    rep8, def8 = rep_levels_rle(8, DIM), def_levels_rle(8, 3)
    idx8 = bytes([2]) + packed_run([0, 1, 2, 3, 0, 1, 2, 3], 2)
    body = struct.pack("<I", len(rep8)) + rep8 + struct.pack("<I", len(def8)) + def8 + idx8
    page_b = v1_page(bytes([2]) + rle_run(3, 4, 2), 4, rep_levels_rle(4, DIM), def_levels_rle(4, 3), encoding=8, maxdef=3)
    return dpg, body, page_b


@pytest.mark.parametrize("stray_id,stray", [
    (7, S((1, I32, 4), (2, I32, 8))),                                                      # dictionary_page_header
    (8, S((1, I32, 4), (2, I32, 0), (3, I32, 1), (4, I32, 8), (5, I32, 0), (6, I32, 0))),  # data_page_header_v2
])
@pytest.mark.parametrize("rows", [2, 3])
def test_stray_sub_header_of_another_page_type_is_ignored(tmp_path, client, stray_id, stray, rows):
    dpg, body, page_b = _dict_pages()
    hdr = page_header(0, len(body), len(body), S((1, I32, 8), (2, I32, 8), (3, I32, 3), (4, I32, 3)), 5,
                      extra=[(stray_id, STRUCT, stray)])
    p = str(tmp_path / "stray.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=dpg + hdr + body + page_b, n_rows=rows, num_values=rows * DIM,
                                    has_dict=True, dict_len=len(dpg))])
    _native_never_differs(client, p)


def test_data_page_missing_its_sub_header_is_corrupt(tmp_path, client):
    vals = _f32(np.arange(8))
    rep, dfl = rep_levels_rle(8, DIM), def_levels_rle(8, 3)
    body = struct.pack("<I", len(rep)) + rep + struct.pack("<I", len(dfl)) + dfl + vals
    p = str(tmp_path / "nosub.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=page_header(0, len(body), len(body)) + body, n_rows=2, num_values=8)])
    with pytest.raises(CorruptParquet):
        rust_fetch.read_dense(client, p, "e")


@pytest.mark.parametrize("label,idx", [
    ("zero_rle_run", bytes([2]) + rle_run(1, 4, 2) + rle_run(0, 0, 2) + rle_run(2, 4, 2)),
    ("zero_packed_run", bytes([2]) + rle_run(1, 4, 2) + uvar(1) + rle_run(2, 4, 2)),
    ("truncated_packed_run", bytes([2]) + packed_run([0, 1, 2, 3, 3, 2, 1, 0], 2, truncate_to=1) + rle_run(3, 4, 2)),
    ("packed_run_claims_more_than_present", bytes([1]) + bytes([0x7F, 0x00])),
    ("rle_count_past_n", bytes([2]) + rle_run(1, 100, 2)),
    ("bw_33", bytes([33]) + rle_run(1, 8, 33)),
])
def test_dictionary_index_corners_never_differ(tmp_path, client, label, idx):
    dpg = dict_page(_f32([10, 20, 30, 40]), 4)
    pg = v1_page(idx, 8, rep_levels_rle(8, DIM), def_levels_rle(8, 3), encoding=8, maxdef=3)
    p = str(tmp_path / f"{label}.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=dpg + pg, n_rows=2, num_values=8, has_dict=True, dict_len=len(dpg))])
    _native_never_differs(client, p)


def test_truncated_bit_packed_levels_are_corrupt(tmp_path, client):
    rep = packed_run([0, 1, 1, 1, 0, 1, 1, 1], 1, truncate_to=0)
    pg = v1_page(_f32(np.arange(8)), 8, rep, def_levels_rle(8, 3), maxdef=3)
    p = str(tmp_path / "trunc_rep.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=pg, n_rows=2, num_values=8)])
    _native_never_differs(client, p)


R = lambda: selem("schema", nchild=1)  # noqa: E731
SCHEMAS = {
    "legacy_array_struct": ([R(), selem("e", rep=OPT, nchild=1, conv=3, logical=3), selem("array", rep=REP, nchild=1),
                             selem("item", type=FLOAT, rep=REQ)], ("e", "array", "item"), 2),
    "legacy_tuple_struct": ([R(), selem("e", rep=OPT, nchild=1, conv=3, logical=3), selem("e_tuple", rep=REP, nchild=1),
                             selem("item", type=FLOAT, rep=REQ)], ("e", "e_tuple", "item"), 2),
    "struct_element": ([R(), selem("e", rep=OPT, nchild=1, conv=3, logical=3), selem("list", rep=REP, nchild=1),
                        selem("element", rep=OPT, nchild=1), selem("x", type=FLOAT, rep=OPT)], ("e", "list", "element", "x"), 4),
    "map_kv_inside_list": ([R(), selem("e", rep=OPT, nchild=1, conv=3, logical=3), selem("key_value", rep=REP, nchild=1, conv=2),
                            selem("key", type=FLOAT, rep=REQ)], ("e", "key_value", "key"), 2),
    "top_repeated_list_group": ([R(), selem("e", rep=REP, nchild=1, conv=3, logical=3), selem("element", type=FLOAT, rep=REQ)],
                                ("e", "element"), 1),
    "top_repeated_group": ([R(), selem("e", rep=REP, nchild=1), selem("x", type=FLOAT, rep=REQ)], ("e", "x"), 1),
    "f16_without_logical": ([R(), selem("e", rep=OPT, nchild=1, conv=3, logical=3), selem("list", rep=REP, nchild=1),
                             selem("element", type=FLBA, tlen=2, rep=OPT)], ("e", "list", "element"), 3),
    # accepted shapes
    "legacy_2level": ([R(), selem("e", rep=OPT, nchild=1, conv=3, logical=3), selem("array", type=FLOAT, rep=REP)],
                      ("e", "array"), 2),
    "top_repeated_primitive": ([R(), selem("e", type=FLOAT, rep=REP)], ("e",), 1),
    "all_required": ([R(), selem("e", rep=REQ, nchild=1, conv=3, logical=3), selem("list", rep=REP, nchild=1),
                      selem("element", type=FLOAT, rep=REQ)], ("e", "list", "element"), 1),
}
ACCEPTED = {"legacy_2level", "top_repeated_primitive", "all_required"}


@pytest.mark.parametrize("name", list(SCHEMAS))
def test_schema_shapes(tmp_path, client, name):
    schema, path, maxdef = SCHEMAS[name]
    n = 3
    leaf = FLBA if "f16" in name else FLOAT
    vb = (np.arange(n * DIM, dtype=np.float16) if leaf == FLBA else np.arange(n * DIM, dtype=np.float32) + 0.5).tobytes()
    pg = v1_page(vb, n * DIM, rep_levels_rle(n * DIM, DIM), def_levels_rle(n * DIM, maxdef), maxdef=maxdef)
    p = str(tmp_path / f"{name}.parquet")
    write_file(p, schema, [dict(pages=pg, n_rows=n, num_values=n * DIM)], leaf_type=leaf, path_in_schema=path)
    outcome = _native_never_differs(client, p)
    assert outcome == ("identical" if name in ACCEPTED else "fallback"), outcome


def test_index_page_and_empty_page_after_the_last_value_are_fine(tmp_path, client):
    vals = np.arange(8, dtype=np.float32) + 1
    pg = v1_page(vals.tobytes(), 8, rep_levels_rle(8, DIM), def_levels_rle(8, 3), maxdef=3)
    index_page = page_header(1, 0, 0)
    empty = v1_page(b"", 0, b"", b"", maxdef=3)
    p = str(tmp_path / "trailing.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=pg + index_page + empty, n_rows=2, num_values=8)])
    got = rust_fetch.read_dense(client, p, "e")
    assert got.tobytes() == vals.tobytes()


# --- empty data pages: pyarrow tolerates exactly one shape ------------------


def _empty_v1():
    return v1_page(b"", 0, b"", b"", maxdef=3)


def _rows(vals, maxdef=3):
    n = len(vals) // DIM * DIM
    return v1_page(_f32(vals), n, rep_levels_rle(n, DIM), def_levels_rle(n, maxdef), maxdef=maxdef)


@pytest.mark.parametrize("name,pages,accepted", [
    # single empty page at a row boundary after a data page: pyarrow reads it
    ("single_mid", lambda: _rows(np.arange(4)) + _empty_v1() + _rows(np.arange(4, 12)), True),
    # pyarrow drops data or errors on all of these
    ("first", lambda: _empty_v1() + _rows(np.arange(12)), False),
    ("two_mid", lambda: _rows(np.arange(4)) + _empty_v1() + _empty_v1() + _rows(np.arange(4, 12)), False),
    ("after_index", lambda: _rows(np.arange(4)) + page_header(1, 0, 0) + _empty_v1() + _rows(np.arange(4, 12)), False),
])
def test_empty_data_page_placements(tmp_path, client, name, pages, accepted):
    p = str(tmp_path / f"{name}.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=pages(), n_rows=3, num_values=12)])
    outcome = _native_never_differs(client, p)
    assert outcome == ("identical" if accepted else "fallback"), outcome


def test_pyarrow_writer_empty_v2_page_decodes_natively(tmp_path, client):
    # pyarrow emits a 0-value v2 page before the last one with tiny
    # dictionary-encoded pages; it must stay on the native path.
    rng = np.random.default_rng(0)
    pool = rng.standard_normal(3).astype(np.float32)
    v = pool[rng.integers(0, 3, (97, 13))]
    import pyarrow as pa
    col = pa.ListArray.from_arrays(pa.array(np.arange(98, dtype=np.int32) * 13), pa.array(v.reshape(-1)))
    p = str(tmp_path / "w.parquet")
    pq.write_table(pa.table({"e": col}), p, use_dictionary=True, data_page_version="2.0",
                   data_page_size=64, dictionary_pagesize_limit=64, write_batch_size=7)
    got = rust_fetch.read_dense(client, p, "e")
    assert got.tobytes() == v.tobytes()


# --- metadata of OTHER columns that pyarrow still acts on --------------------


def _find_num_values_offset(footer: bytes, rg: int, col: int) -> tuple[int, int]:
    """(offset, length) of FileMetaData.row_groups[rg].columns[col]
    .meta_data.num_values's varint inside the thrift-compact footer."""
    pos = 0

    def uv():
        nonlocal pos
        v = s = 0
        while True:
            b = footer[pos]; pos += 1
            v |= (b & 0x7F) << s; s += 7
            if b < 0x80:
                return v

    def skip(t):
        nonlocal pos
        if t in (1, 2):
            return
        if t == 3:
            pos += 1
        elif t in (4, 5, 6):
            uv()
        elif t == 7:
            pos += 8
        elif t == 8:
            n = uv(); pos += n
        elif t in (9, 10):
            h = footer[pos]; pos += 1
            n = h >> 4
            if n == 15:
                n = uv()
            for _ in range(n):
                if (h & 15) in (1, 2):
                    pos += 1
                else:
                    skip(h & 15)
        elif t == 12:
            walk(())
        else:
            raise AssertionError(t)

    found = {}

    def walk(path):
        nonlocal pos
        last = 0
        idx = {"rg": -1, "col": -1}
        while True:
            h = footer[pos]; pos += 1
            if h == 0:
                return
            t, d = h & 15, h >> 4
            fid = last + d if d else (lambda u: (u >> 1) ^ -(u & 1))(uv())
            last = fid
            here = path + (fid,)
            if here == (4,) and t == 9:              # row_groups list
                hh = footer[pos]; pos += 1
                n = hh >> 4 if hh >> 4 != 15 else uv()
                for i in range(n):
                    walk(("rg", i))
                continue
            if len(path) == 2 and path[0] == "rg" and fid == 1 and t == 9:   # columns list
                hh = footer[pos]; pos += 1
                n = hh >> 4 if hh >> 4 != 15 else uv()
                for j in range(n):
                    walk(("col", path[1], j))
                continue
            if len(path) == 3 and path[0] == "col" and fid == 3 and t == 12:  # meta_data
                walk(("md", path[1], path[2]))
                continue
            if len(path) == 3 and path[0] == "md" and fid == 5 and t == 6:     # num_values
                start = pos; uv()
                found[(path[1], path[2])] = (start, pos - start)
                continue
            skip(t)

    walk(())
    return found[(rg, col)]


def test_other_column_with_fewer_values_than_rows_falls_back(tmp_path, client):
    """pyarrow silently truncates a row group to ANY chunk's num_values when
    that is below num_rows -- even for a column it wasn't asked to read."""
    import pyarrow as pa
    import struct as st

    n = 13
    v = np.arange(n * DIM, dtype=np.float32).reshape(n, DIM)
    p = str(tmp_path / "base.parquet")
    col = pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)), DIM)
    pq.write_table(pa.table({"id": pa.array(np.arange(n, dtype=np.int64)), "e": col}), p)
    b = bytearray(open(p, "rb").read())
    flen = st.unpack("<I", b[-8:-4])[0]
    fstart = len(b) - 8 - flen
    off, ln = _find_num_values_offset(bytes(b[fstart:len(b) - 8]), 0, 0)
    assert ln == 1 and b[fstart + off] == n << 1       # zigzag(13) in one byte
    b[fstart + off] = 5 << 1                            # id claims 5 values
    q = str(tmp_path / "under.parquet")
    open(q, "wb").write(b)
    # pyarrow <= 24 truncates the row group to 5 rows; 26 reads all 13. Either
    # way native must refuse and leave the answer to pyarrow.
    assert pq.read_table(q, columns=["e"]).num_rows in (5, n)
    assert _native_never_differs(client, q) == "fallback"
    b[fstart + off] = 20 << 1                           # MORE values: harmless
    open(q, "wb").write(b)
    assert _native_never_differs(client, q) == "identical"


def test_lying_uncompressed_size_is_refused_without_allocating(tmp_path, client):
    import resource

    rows = 1 << 27
    body = bytes([0x0A, 0x24, 0x00]) + b"\x00" * 7
    hdr = page_header(0, (1 << 31) - 1, len(body), S((1, I32, rows), (2, I32, 0), (3, I32, 3), (4, I32, 3)), 5)
    p = str(tmp_path / "lie.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=hdr + body, n_rows=rows, num_values=rows)] * 4, codec=1)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    plan = client.dense_plan(p, "e")
    dest = np.empty(plan.nbytes, np.uint8)  # lazily mapped, never touched
    with pytest.raises((UnsupportedLayout, CorruptParquet)):
        client.read_dense_into(plan, dest)
    grew_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert grew_mb < 512, f"peak RSS grew {grew_mb:.0f} MB on a lying header"


@pytest.mark.parametrize("label,empty", [
    ("garbage_body", lambda: page_header(0, 5, 5, S((1, I32, 0), (2, I32, 0), (3, I32, 3), (4, I32, 3)), 5) + b"\xff" * 5),
    ("unknown_encoding", lambda: page_header(0, 8, 8, S((1, I32, 0), (2, I32, 99), (3, I32, 3), (4, I32, 3)), 5) + b"\x00" * 8),
    ("dict_without_dictionary", lambda: page_header(0, 9, 9, S((1, I32, 0), (2, I32, 8), (3, I32, 3), (4, I32, 3)), 5) + b"\x00" * 8 + b"\x02"),
])
def test_malformed_mid_chunk_empty_page_is_validated(tmp_path, client, label, empty):
    """A tolerated-position empty page must still be well formed: pyarrow
    decodes it and fails on garbage."""
    p = str(tmp_path / f"{label}.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=_rows(np.arange(8)) + empty() + _rows(np.arange(8, 12)), n_rows=3, num_values=12)])
    assert _native_never_differs(client, p) == "fallback"


def test_lying_uncompressed_dictionary_page_is_refused_without_allocating(tmp_path, client):
    import resource

    dpg = page_header(2, 200_000_000, 4, S((1, I32, 1), (2, I32, 0)), 7) + _f32([1.0])
    rows = 50_000_000
    pg = v1_page(bytes([0]) + rle_run(0, 8, 0), 8, rep_levels_rle(8, DIM), def_levels_rle(8, 3), encoding=8, maxdef=3)
    p = str(tmp_path / "dlie.parquet")
    write_file(p, STD_SCHEMA, [dict(pages=dpg + pg, n_rows=rows, num_values=rows, has_dict=True, dict_len=len(dpg))] * 4)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        plan = client.dense_plan(p, "e")
        client.read_dense_into(plan, np.empty(plan.nbytes, np.uint8))
    except (UnsupportedLayout, CorruptParquet):
        pass
    grew_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert grew_mb < 64, f"peak RSS grew {grew_mb:.0f} MB on a lying dictionary page"


@pytest.mark.parametrize("url", ["s3://b//x.parquet", "s3://b/d//x.parquet", "s3://b/x.parquet/"])
def test_s3_keys_with_empty_segments_are_refused(url):
    c = rust_fetch.Client(region="us-east-1", retries=0, connect_timeout_s=0.5, part_timeout_s=0.5)
    with pytest.raises(ValueError, match="empty path segment"):
        c.head(url)
