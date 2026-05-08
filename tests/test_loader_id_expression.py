"""
Regression tests for the loader's point-id derivation.

The loader, brute-force search, and query generator must all agree on the row
index used in `make_point_id(file, row)`. Brute-force and query generation
read parquets with pyarrow, which emits rows in physical row-group order. The
loader must therefore use DuckDB's `file_row_number` virtual column -- NOT
`ROW_NUMBER() OVER (PARTITION BY filename)`, which reflects DuckDB's parallel
scan order and reorders rows.
"""

import os
import tempfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.loader.datasource.s3 import S3DataReader
from vectorforge.utils import make_point_id


def _write_multi_rg_parquet(path: str, n: int, row_group_size: int) -> None:
    """Parquet with `id` matching the physical row index, in multiple row groups."""
    table = pa.table(
        {
            "id": list(range(n)),
            "tag": [f"tag_{i:05d}" for i in range(n)],
            "dense_embedding": [
                [float(i), float(i + 1), float(i + 2)] for i in range(n)
            ],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def test_file_row_number_matches_pyarrow_physical_order():
    """
    With high parallelism DuckDB scans row groups out of order, but the
    `file_row_number` virtual column always reflects the physical row index
    inside each parquet -- the same index pyarrow exposes.
    """
    n = 1000
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.parquet")
        _write_multi_rg_parquet(path, n=n, row_group_size=100)

        conn = duckdb.connect()
        conn.execute("SET threads = 8;")
        sql = f"""
            SELECT file_row_number AS frn, id AS true_id
            FROM read_parquet('{path}', file_row_number=true)
        """
        rows = conn.execute(sql).fetchall()

        assert len(rows) == n
        for frn, true_id in rows:
            assert frn == true_id, f"file_row_number={frn} but physical row={true_id}"


def test_loader_id_matches_brute_force_id():
    """
    End-to-end: the IDs produced by the loader's read_batches() must match the
    IDs computed by `make_point_id(file_key, pyarrow_row)` -- which is what
    brute-force search and query generation use.

    Uses a local file via S3DataReader's file_list path. The vf_point_id macro
    strips ``s3://{bucket}/`` from the filename; for the test we mimic that
    stripping in Python so the "key" both sides see is identical.
    """
    n = 1000
    bucket = "b"
    bucket_uri_len = len(f"s3://{bucket}/")  # macro strips this many chars

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.parquet")
        _write_multi_rg_parquet(path, n=n, row_group_size=100)

        reader = S3DataReader(
            bucket=bucket,
            prefix="unused-when-file-list-is-set",
            id_expression="vf_point_id(filename, file_row_number)",
            vectors={"dense": {"type": "dense", "column": "dense_embedding"}},
            payload_fields={"id": "id"},
            file_list=[path],  # bare local path; DuckDB reads it directly
            duckdb_threads=8,
        )

        loader_ids: dict[int, str] = {}
        for batch in reader.read_batches(batch_size=128):
            for record in batch:
                true_id = record["payload"]["id"]
                loader_ids[true_id] = record["id"]
        reader.close()

        # Mimic the macro's substr: brute-force / query gen would call
        # make_point_id with the same post-strip key string.
        expected_key = path[bucket_uri_len:]
        assert len(loader_ids) == n
        for i in range(n):
            expected = make_point_id(expected_key, i)
            assert loader_ids[i] == expected, (
                f"row {i}: loader produced {loader_ids[i]!r}, expected {expected!r}"
            )


def test_row_number_window_does_not_match_physical_order():
    """
    Negative regression: documents *why* file_row_number is required.
    `ROW_NUMBER() OVER (PARTITION BY filename)` reflects DuckDB's scan order
    and reorders rows under parallelism. If this test ever starts failing
    (i.e. ROW_NUMBER becomes deterministic), check whether DuckDB's behavior
    changed -- but do NOT switch the loader to ROW_NUMBER on the basis of
    that, because scan order is still implementation-defined.
    """
    n = 1000
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.parquet")
        _write_multi_rg_parquet(path, n=n, row_group_size=100)

        conn = duckdb.connect()
        conn.execute("SET threads = 8;")
        sql = f"""
            SELECT ROW_NUMBER() OVER (PARTITION BY filename) - 1 AS rn,
                   id AS true_id
            FROM read_parquet('{path}', filename=true)
        """
        rows = conn.execute(sql).fetchall()
        mismatches = sum(1 for rn, true_id in rows if rn != true_id)
        assert mismatches > 100, (
            f"ROW_NUMBER() unexpectedly matched physical order ({mismatches} mismatches). "
            "DuckDB scan order may have changed; verify the loader still uses file_row_number."
        )
