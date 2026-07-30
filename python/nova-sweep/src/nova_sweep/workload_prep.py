"""Framework-agnostic parquet workload prep + query-result cleanup utilities.

These helpers port the reusable concepts from thirdparty/ten-billion/locust:
- deterministic parquet shard/row sampling
- workload file splitting (query/delete/upsert/shared)
- query-result filtering and deduplication

They intentionally avoid any Locust runtime dependency so they can be reused by
`nova sweep`, `nova dist`, standalone scripts, or future command wrappers.
"""

from __future__ import annotations

import hashlib
import json
import math
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_PARQUET_COL = "_workload_source_parquet"
SOURCE_ROW_COL = "_workload_source_row_index"

ReadOrder = Literal["sequential", "random"]


@dataclass(frozen=True)
class WorkloadSplitConfig:
    query_columns: list[str]
    delete_columns: list[str]
    upsert_columns: list[str]
    include_source_metadata: bool = True
    compression: str = "zstd"


def discover_parquet_paths(root: Path) -> list[Path]:
    """Return all parquet files under `root` in deterministic path order."""
    root = root.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() != ".parquet":
            raise ValueError(f"not a parquet file: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"parquet path not found: {root}")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no *.parquet files under {root}")
    return files


def allocate_even_counts(capacities: list[int], total_rows: int) -> list[int]:
    """Split `total_rows` across capacities as evenly as possible."""
    remaining = min(total_rows, sum(capacities))
    counts = [0] * len(capacities)
    active = {idx for idx, cap in enumerate(capacities) if cap > 0}
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active: set[int] = set()
        for idx in active:
            room = capacities[idx] - counts[idx]
            if room <= 0:
                continue
            take = min(share, room)
            counts[idx] += take
            remaining -= take
            if counts[idx] < capacities[idx]:
                next_active.add(idx)
            if remaining == 0:
                break
        active = next_active
    return counts


def _per_file_rng(seed: int, file_path: Path) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{file_path.resolve()}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def sample_parquet_rows(
    root: Path,
    columns: list[str],
    *,
    max_files: int = 0,
    max_rows: int = 0,
    seed: int | None = None,
    read_order: ReadOrder = "sequential",
) -> tuple[pa.Table, list[str], list[int]]:
    """Sample rows from parquet files and return table + source lineage arrays."""
    files = discover_parquet_paths(root)
    rng = random.Random(seed)
    if max_files > 0:
        if read_order == "sequential":
            files = files[:max_files]
        else:
            if len(files) < max_files:
                raise ValueError(
                    f"only {len(files)} parquet file(s) under {root}; need max_files={max_files}"
                )
            files = rng.sample(files, max_files)

    tables = [pq.read_table(path, columns=columns) for path in files]
    capacities = [tbl.num_rows for tbl in tables]
    targets = allocate_even_counts(capacities, max_rows) if max_rows > 0 else capacities

    sampled: list[pa.Table] = []
    source_files: list[str] = []
    source_rows: list[int] = []
    for path, table, take in zip(files, tables, targets):
        if take <= 0:
            continue
        row_idxs = list(range(table.num_rows))
        if take < table.num_rows:
            if read_order == "sequential":
                row_idxs = list(range(take))
            else:
                file_rng = _per_file_rng(seed, path) if seed is not None else random.Random()
                row_idxs = sorted(file_rng.sample(row_idxs, take))
            table = table.take(pa.array(row_idxs, type=pa.int64()))
        sampled.append(table)
        source_files.extend([str(path)] * table.num_rows)
        source_rows.extend(row_idxs[: table.num_rows])

    if not sampled:
        raise ValueError(f"no rows sampled under {root}")
    return pa.concat_tables(sampled), source_files, source_rows


def append_source_columns(table: pa.Table, source_files: list[str], source_rows: list[int]) -> pa.Table:
    """Append stable source lineage columns to a sampled table."""
    if len(source_files) != table.num_rows or len(source_rows) != table.num_rows:
        raise ValueError("source metadata length mismatch")
    if SOURCE_PARQUET_COL in table.column_names or SOURCE_ROW_COL in table.column_names:
        raise ValueError("table already has source metadata columns")
    out = table.append_column(SOURCE_PARQUET_COL, pa.array(source_files, type=pa.string()))
    out = out.append_column(SOURCE_ROW_COL, pa.array(source_rows, type=pa.int64()))
    return out


def prepare_workload_split(
    root: Path,
    output_dir: Path,
    *,
    max_files: int = 0,
    max_rows: int = 0,
    seed: int | None = None,
    read_order: ReadOrder = "random",
    config: WorkloadSplitConfig,
) -> dict[str, Path]:
    """Create query/delete/upsert/shared parquet files from sampled source rows."""
    all_cols = _dedupe_columns(config.query_columns + config.delete_columns + config.upsert_columns)
    sampled, source_files, source_rows = sample_parquet_rows(
        root,
        all_cols,
        max_files=max_files,
        max_rows=max_rows,
        seed=seed,
        read_order=read_order,
    )
    if config.include_source_metadata:
        sampled = append_source_columns(sampled, source_files, source_rows)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comp = None if config.compression == "none" else config.compression

    result: dict[str, Path] = {}
    for name, columns in {
        "query": config.query_columns,
        "delete": config.delete_columns,
        "upsert": config.upsert_columns,
        "shared": all_cols,
    }.items():
        cols = list(columns)
        if config.include_source_metadata and name in {"query", "shared"}:
            cols += [SOURCE_PARQUET_COL, SOURCE_ROW_COL]
        table = sampled.select(cols)
        path = output_dir / f"{name}.parquet"
        pq.write_table(table, path, compression=comp)
        result[name] = path
    return result


def _json_cell(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    return raw


def normalize_legacy_query_results(
    table: pa.Table,
    *,
    query_point_col: str = "qdrant_query_point_json",
    neighbors_col: str = "qdrant_neighbor_points_json",
    error_col: str = "qdrant_error",
) -> pa.Table:
    """Normalize legacy 3-column query-result parquet into modern shape."""
    if query_point_col not in table.column_names or neighbors_col not in table.column_names:
        return table
    errors = (
        table.column(error_col).to_pylist() if error_col in table.column_names else [None] * table.num_rows
    )
    rows: list[dict[str, Any]] = []
    for raw_query, raw_neighbors, raw_error in zip(
        table.column(query_point_col).to_pylist(),
        table.column(neighbors_col).to_pylist(),
        errors,
    ):
        parsed_neighbors = _json_cell(raw_neighbors)
        hit_ids: list[str] = []
        if isinstance(parsed_neighbors, list):
            for item in parsed_neighbors:
                if isinstance(item, dict):
                    key = item.get("id") or item.get("point_id") or item.get("pointId")
                    if key is not None:
                        hit_ids.append(str(key))
                else:
                    hit_ids.append(str(item))
        elif isinstance(parsed_neighbors, dict):
            hit_ids = [str(k) for k in parsed_neighbors]

        rows.append(
            {
                "query_type": "legacy",
                "error": None if raw_error in (None, "") else str(raw_error),
                "hit_count": len(hit_ids),
                "point_id": hit_ids[0] if hit_ids else None,
                "hit_point_ids_json": json.dumps(hit_ids, separators=(",", ":")) if hit_ids else None,
                "query_vector_json": (
                    raw_query
                    if isinstance(raw_query, str)
                    else json.dumps(raw_query, separators=(",", ":")) if raw_query is not None else None
                ),
            }
        )
    return pa.Table.from_pylist(rows)


def filter_query_results(
    table: pa.Table,
    *,
    point_id_column: str = "point_id",
    min_hit_count: int = 1,
    drop_errors: bool = True,
    query_type: str | None = None,
) -> tuple[pa.Table, int, int]:
    """Drop rows with errors/empty point ids/low hit_count and optionally by query_type."""
    n_before = table.num_rows
    if n_before == 0:
        return table, 0, 0
    keep = [True] * n_before
    if drop_errors and "error" in table.column_names:
        for idx, value in enumerate(table.column("error").to_pylist()):
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            keep[idx] = False
    if min_hit_count > 0 and "hit_count" in table.column_names:
        for idx, value in enumerate(table.column("hit_count").to_pylist()):
            if not keep[idx]:
                continue
            try:
                if int(value) < min_hit_count:
                    keep[idx] = False
            except (TypeError, ValueError):
                keep[idx] = False
    if point_id_column in table.column_names:
        for idx, value in enumerate(table.column(point_id_column).to_pylist()):
            if not keep[idx]:
                continue
            if value is None:
                keep[idx] = False
                continue
            if isinstance(value, float) and math.isnan(value):
                keep[idx] = False
                continue
            if isinstance(value, (str, bytes, bytearray)) and str(value).strip() == "":
                keep[idx] = False
    if query_type is not None:
        if "query_type" not in table.column_names:
            raise ValueError("query_type filter requested but table has no query_type column")
        for idx, value in enumerate(table.column("query_type").to_pylist()):
            if keep[idx] and str(value) != query_type:
                keep[idx] = False
    n_after = sum(keep)
    if n_after == n_before:
        return table, n_before, n_after
    return table.filter(pa.array(keep, type=pa.bool_())), n_before, n_after


def default_query_result_dedupe_subset(table: pa.Table) -> list[str] | None:
    """Best-effort dedupe keys for query-result tables with optional lineage."""
    names = set(table.column_names)
    if {"query_type", "hit_point_ids_json", "point_id"} - names:
        return None
    if SOURCE_PARQUET_COL in names and SOURCE_ROW_COL in names:
        return [SOURCE_PARQUET_COL, SOURCE_ROW_COL, "query_type"]
    if {"input_parquet_file", "input_row_index"}.issubset(names):
        return ["input_parquet_file", "input_row_index", "query_type"]
    if "idx" in names:
        return ["idx", "query_type"]
    if "timestamp_epoch_ms" in names:
        return ["timestamp_epoch_ms", "query_type"]
    return None


def dedupe_rows(table: pa.Table, subset: list[str] | None) -> tuple[pa.Table, int, int]:
    """Drop duplicates, keeping first row by optional key subset."""
    n_before = table.num_rows
    if n_before == 0:
        return table, 0, 0
    if subset is not None:
        missing = [col for col in subset if col not in table.column_names]
        if missing:
            raise ValueError(f"dedupe subset columns missing from table: {missing}")

    rows = table.to_pylist()
    seen: set[Any] = set()
    keep_rows: list[dict[str, Any]] = []
    if subset is None:
        for row in rows:
            key = tuple(sorted((k, repr(v)) for k, v in row.items()))
            if key in seen:
                continue
            seen.add(key)
            keep_rows.append(row)
    else:
        for row in rows:
            key = tuple(repr(row.get(col)) for col in subset)
            if key in seen:
                continue
            seen.add(key)
            keep_rows.append(row)

    if len(keep_rows) == len(rows):
        return table, n_before, n_before
    out = pa.Table.from_pylist(keep_rows, schema=table.schema)
    return out, n_before, out.num_rows
