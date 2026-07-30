from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nova_sweep.workload_prep import (
    SOURCE_PARQUET_COL,
    SOURCE_ROW_COL,
    WorkloadSplitConfig,
    dedupe_rows,
    default_query_result_dedupe_subset,
    filter_query_results,
    normalize_legacy_query_results,
    prepare_workload_split,
    sample_parquet_rows,
)


def _write_shard(path: Path, ids: list[str], start: int) -> None:
    table = pa.table(
        {
            "id": ids,
            "dense_embedding": [[float(start + i), 0.0] for i in range(len(ids))],
            "sparse_embedding": [
                {"indices": [0, 1], "values": [1.0, 0.5]} for _ in ids
            ],
            "payload": [f"p{start + i}" for i in range(len(ids))],
        }
    )
    pq.write_table(table, path)


def test_sample_parquet_rows_random_is_deterministic(tmp_path: Path) -> None:
    _write_shard(tmp_path / "a.parquet", ["a1", "a2", "a3"], 0)
    _write_shard(tmp_path / "b.parquet", ["b1", "b2", "b3"], 10)
    _write_shard(tmp_path / "c.parquet", ["c1", "c2", "c3"], 20)

    t1, sf1, sr1 = sample_parquet_rows(
        tmp_path,
        ["id", "dense_embedding", "sparse_embedding"],
        max_files=2,
        max_rows=4,
        seed=7,
        read_order="random",
    )
    t2, sf2, sr2 = sample_parquet_rows(
        tmp_path,
        ["id", "dense_embedding", "sparse_embedding"],
        max_files=2,
        max_rows=4,
        seed=7,
        read_order="random",
    )
    assert t1.to_pylist() == t2.to_pylist()
    assert sf1 == sf2
    assert sr1 == sr2


def test_prepare_workload_split_writes_expected_files(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _write_shard(in_dir / "s1.parquet", ["x1", "x2", "x3"], 0)
    _write_shard(in_dir / "s2.parquet", ["y1", "y2", "y3"], 100)

    paths = prepare_workload_split(
        in_dir,
        out_dir,
        max_files=0,
        max_rows=4,
        seed=1,
        read_order="sequential",
        config=WorkloadSplitConfig(
            query_columns=["dense_embedding", "id"],
            delete_columns=["id"],
            upsert_columns=["dense_embedding", "sparse_embedding", "id", "payload"],
        ),
    )

    assert set(paths) == {"query", "delete", "upsert", "shared"}
    query = pq.read_table(paths["query"])
    assert SOURCE_PARQUET_COL in query.column_names
    assert SOURCE_ROW_COL in query.column_names
    assert query.num_rows == 4

    delete = pq.read_table(paths["delete"])
    assert delete.column_names == ["id"]
    assert delete.num_rows == 4

    upsert = pq.read_table(paths["upsert"])
    assert upsert.column_names == ["dense_embedding", "sparse_embedding", "id", "payload"]


def test_filter_and_dedupe_query_results() -> None:
    table = pa.table(
        {
            "query_type": ["exact", "exact", "exact", "ann"],
            "error": [None, "boom", "", None],
            "hit_count": [1, 3, 0, 2],
            "point_id": ["p1", "p2", "p3", "p4"],
            "hit_point_ids_json": ["[\"p1\"]", "[\"p2\"]", "[\"p3\"]", "[\"p4\"]"],
            "input_parquet_file": ["f.parquet", "f.parquet", "f.parquet", "g.parquet"],
            "input_row_index": [1, 2, 3, 9],
        }
    )
    filtered, n_before, n_after = filter_query_results(
        table, min_hit_count=1, drop_errors=True, query_type="exact"
    )
    assert n_before == 4
    assert n_after == 1
    assert filtered.column("point_id").to_pylist() == ["p1"]

    duped = pa.concat_tables([filtered, filtered], promote_options="default")
    subset = default_query_result_dedupe_subset(duped)
    deduped, d_before, d_after = dedupe_rows(duped, subset)
    assert d_before == 2
    assert d_after == 1
    assert deduped.column("point_id").to_pylist() == ["p1"]


def test_normalize_legacy_query_results() -> None:
    table = pa.table(
        {
            "qdrant_query_point_json": ['{"id":"q1","vector":[0.1,0.2]}'],
            "qdrant_neighbor_points_json": ['[{"id":"a","score":0.9},{"id":"b","score":0.8}]'],
            "qdrant_error": [None],
        }
    )
    normalized = normalize_legacy_query_results(table)
    assert normalized.num_rows == 1
    row = normalized.to_pylist()[0]
    assert row["query_type"] == "legacy"
    assert row["hit_count"] == 2
    assert row["point_id"] == "a"
