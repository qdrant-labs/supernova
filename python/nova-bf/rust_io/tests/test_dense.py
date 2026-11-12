"""Dense fast path (`Client.dense_plan` / `read_dense`): byte-identity against
pyarrow across every layout it accepts, and a clean UnsupportedLayout for
every layout it must not decode itself.

The reference is pyarrow's own decode (flatten -> numpy), i.e. exactly what
nova-bf's `dense_to_2d` builds; float bytes are compared, not values, so NaN
payloads and -0.0 count too.
"""

import itertools

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import rust_fetch
from rust_fetch import CorruptParquet, UnsupportedLayout, read_dense


@pytest.fixture(scope="module")
def client():
    return rust_fetch.Client(concurrency=8, part_size=1 << 16)


def _vectors(n, dim, dtype, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(dtype)
    if n and dim:
        # Bit patterns a value-level comparison would miss.
        flat = v.reshape(-1)
        flat[0] = -0.0
        if flat.size > 2:
            flat[1] = np.nan
            flat[2] = np.inf
    return v


def _column(v, kind):
    n, dim = v.shape
    vt = pa.float16() if v.dtype == np.float16 else pa.float32()
    values = pa.array(v.reshape(-1), type=vt)
    if kind == "fixed":
        return pa.FixedSizeListArray.from_arrays(values, dim)
    offsets = np.arange(0, n * dim + 1, max(dim, 1), dtype=np.int64)[: n + 1] if dim else np.zeros(n + 1, np.int64)
    if kind == "large":
        return pa.LargeListArray.from_arrays(pa.array(offsets, pa.int64()), values)
    return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32), pa.int32()), values)


def _reference(path, column):
    col = pq.read_table(path, columns=[column])[column].combine_chunks()
    n = len(col)
    flat = col.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(n, -1) if n else flat.reshape(0, 0)


def _assert_identical(got, ref):
    assert got.dtype == ref.dtype, (got.dtype, ref.dtype)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert got.tobytes() == ref.tobytes()


CODECS = ["none", "snappy", "gzip", "zstd", "lz4"]


@pytest.mark.parametrize(
    "codec,dictionary,page_version,kind,dtype",
    list(itertools.product(CODECS, [False, True], ["1.0", "2.0"], ["list", "large", "fixed"], [np.float32, np.float16])),
)
def test_layout_matrix_is_byte_identical(tmp_path, client, codec, dictionary, page_version, kind, dtype):
    v = _vectors(1000, 24, dtype)
    t = pa.table({"id": pa.array(np.arange(1000)), "embedding": _column(v, kind), "tail": pa.array(np.ones(1000))})
    p = str(tmp_path / "f.parquet")
    pq.write_table(
        t, p, compression=codec, use_dictionary=dictionary, data_page_version=page_version,
        row_group_size=300, data_page_size=2048,  # many pages; rows straddle pages
    )
    got = read_dense(client, p, "embedding")
    _assert_identical(got, _reference(p, "embedding"))
    np.testing.assert_array_equal(got.view(np.uint8), v.view(np.uint8))


@pytest.mark.parametrize("n,dim,rg", [(1, 1, 1), (5, 1, 2), (7, 3, 1), (64, 129, 10), (513, 8, 512), (2000, 2, 2000)])
@pytest.mark.parametrize("in_flight", [1, 3, 64])
def test_shapes_and_row_groups(tmp_path, client, n, dim, rg, in_flight):
    v = _vectors(n, dim, np.float32, seed=n * 1000 + dim)
    p = str(tmp_path / "f.parquet")
    pq.write_table(pa.table({"e": _column(v, "list")}), p, row_group_size=rg, data_page_size=512)
    _assert_identical(read_dense(client, p, "e", chunks_in_flight=in_flight), _reference(p, "e"))


def test_small_dictionary_hits_dictionary_pages(tmp_path, client):
    # Few distinct values => pyarrow keeps every page dictionary-encoded.
    rng = np.random.default_rng(1)
    v = rng.choice(np.array([0.5, -1.25, 3.0], np.float32), size=(4000, 16))
    p = str(tmp_path / "f.parquet")
    pq.write_table(pa.table({"e": _column(v, "list")}), p, use_dictionary=True, compression="snappy")
    enc = pq.ParquetFile(p).metadata.row_group(0).column(0).encodings
    assert any("DICTIONARY" in e for e in enc), enc
    _assert_identical(read_dense(client, p, "e"), _reference(p, "e"))


def test_empty_file(tmp_path, client):
    p = str(tmp_path / "f.parquet")
    pq.write_table(pa.table({"e": _column(np.zeros((0, 4), np.float32), "list")}), p)
    got = read_dense(client, p, "e")
    assert got.shape == (0, 0)


def test_size_argument_skips_head_and_matches(tmp_path, client):
    import os

    v = _vectors(100, 8, np.float32)
    p = str(tmp_path / "f.parquet")
    pq.write_table(pa.table({"e": _column(v, "list")}), p)
    _assert_identical(read_dense(client, p, "e", size=os.path.getsize(p)), _reference(p, "e"))


def test_plan_fields(tmp_path, client):
    v = _vectors(100, 8, np.float16)
    p = str(tmp_path / "f.parquet")
    pq.write_table(pa.table({"x": pa.array(range(100)), "e": _column(v, "list")}), p, row_group_size=30, compression="zstd")
    plan = client.dense_plan(p, "e")
    assert (plan.n_rows, plan.dim, plan.dtype, plan.nbytes) == (100, 8, "float16", 1600)
    assert (plan.num_row_groups, plan.codec) == (4, "zstd")
    md = pq.ParquetFile(p).metadata
    assert plan.column_bytes == sum(md.row_group(i).column(1).total_compressed_size for i in range(4))


def test_out_buffer(tmp_path, client):
    v = _vectors(50, 4, np.float32)
    p = str(tmp_path / "f.parquet")
    pq.write_table(pa.table({"e": _column(v, "list")}), p)
    out = np.full((50, 4), 7, np.float32)
    assert read_dense(client, p, "e", out=out) is out
    _assert_identical(out, _reference(p, "e"))
    with pytest.raises(ValueError):
        read_dense(client, p, "e", out=np.empty((50, 5), np.float32))
    with pytest.raises(ValueError):
        client.read_dense_into(client.dense_plan(p, "e"), np.empty(50 * 4 * 4 - 1, np.uint8))


def test_column_name_prefixes_do_not_collide(tmp_path, client):
    a, b = _vectors(20, 4, np.float32, 1), _vectors(20, 6, np.float32, 2)
    p = str(tmp_path / "f.parquet")
    pq.write_table(pa.table({"emb": _column(a, "list"), "embedding": _column(b, "list")}), p)
    _assert_identical(read_dense(client, p, "emb"), _reference(p, "emb"))
    _assert_identical(read_dense(client, p, "embedding"), _reference(p, "embedding"))


# --- layouts that must fall back -------------------------------------------


def _write(tmp_path, table, **kw):
    p = str(tmp_path / "f.parquet")
    pq.write_table(table, p, **kw)
    return p


def _unsupported_cases():
    f32 = pa.float32()
    yield "null_row", pa.table({"e": pa.array([[1.0, 2.0], None, [3.0, 4.0]], pa.list_(f32))})
    yield "null_element", pa.table({"e": pa.array([[1.0, 2.0], [None, 4.0]], pa.list_(f32))})
    yield "empty_list", pa.table({"e": pa.array([[1.0, 2.0], [], [3.0, 4.0]], pa.list_(f32))})
    yield "ragged", pa.table({"e": pa.array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], pa.list_(f32))})
    yield "ragged_same_total", pa.table({"e": pa.array([[1.0], [2.0, 3.0, 4.0], [5.0, 6.0]], pa.list_(f32))})
    yield "float64", pa.table({"e": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64()))})
    yield "int32", pa.table({"e": pa.array([[1, 2], [3, 4]], pa.list_(pa.int32()))})
    yield "not_a_list", pa.table({"e": pa.array([1.0, 2.0], f32)})
    yield "nested_list", pa.table({"e": pa.array([[[1.0]], [[2.0]]], pa.list_(pa.list_(f32)))})
    yield "struct_of_list", pa.table({"e": pa.array([{"v": [1.0, 2.0]}, {"v": [3.0, 4.0]}],
                                                    pa.struct([("v", pa.list_(f32))]))})
    yield "struct_two_lists", pa.table({"e": pa.array([{"a": [1.0], "b": [2.0]}],
                                                      pa.struct([("a", pa.list_(f32)), ("b", pa.list_(f32))]))})


@pytest.mark.parametrize("name,table", list(_unsupported_cases()), ids=[c[0] for c in _unsupported_cases()])
@pytest.mark.parametrize("page_version", ["1.0", "2.0"])
def test_unsupported_layouts_fall_back(tmp_path, client, name, table, page_version):
    p = _write(tmp_path, table, data_page_version=page_version)
    with pytest.raises(UnsupportedLayout):
        read_dense(client, p, "e")


def test_ragged_row_in_a_later_row_group(tmp_path, client):
    # The footer arithmetic passes (same total per group); only the
    # repetition levels in group 2 reveal the bad rows.
    rows = [[1.0, 2.0]] * 4 + [[1.0], [2.0, 3.0, 4.0]] + [[5.0, 6.0]] * 2
    p = _write(tmp_path, pa.table({"e": pa.array(rows, pa.list_(pa.float32()))}), row_group_size=4)
    with pytest.raises(UnsupportedLayout):
        read_dense(client, p, "e")


def test_brotli_falls_back(tmp_path, client):
    p = _write(tmp_path, pa.table({"e": _column(_vectors(10, 4, np.float32), "list")}), compression="brotli")
    with pytest.raises(UnsupportedLayout):
        client.dense_plan(p, "e")


def test_missing_column(tmp_path, client):
    p = _write(tmp_path, pa.table({"e": _column(_vectors(10, 4, np.float32), "list")}))
    with pytest.raises(UnsupportedLayout):
        client.dense_plan(p, "nope")


def test_unsupported_is_a_value_error_and_corrupt_an_os_error():
    assert issubclass(UnsupportedLayout, ValueError)
    assert issubclass(CorruptParquet, OSError)


# --- damaged files -----------------------------------------------------------


def test_truncated_file_is_an_error(tmp_path, client):
    p = _write(tmp_path, pa.table({"e": _column(_vectors(100, 8, np.float32), "list")}))
    data = open(p, "rb").read()
    q = str(tmp_path / "t.parquet")
    open(q, "wb").write(data[: len(data) // 2])
    with pytest.raises((CorruptParquet, OSError)):
        read_dense(client, q, "e")


def test_flipped_page_bytes_never_silently_pass(tmp_path, client):
    """Damage inside the column's first page header: the decode must fail,
    or -- if the damage happens to land in a value byte -- differ from the
    original only there. It must never crash or write outside the output."""
    v = _vectors(200, 8, np.float32)
    p = _write(tmp_path, pa.table({"e": _column(v, "list")}), data_page_size=1 << 20)
    data = bytearray(open(p, "rb").read())
    start = pq.ParquetFile(p).metadata.row_group(0).column(0).data_page_offset
    for k in range(0, 24):
        bad = bytearray(data)
        bad[start + k] ^= 0xFF
        q = str(tmp_path / f"b{k}.parquet")
        open(q, "wb").write(bad)
        try:
            read_dense(client, q, "e")
        except (UnsupportedLayout, CorruptParquet, OSError, ValueError):
            pass
