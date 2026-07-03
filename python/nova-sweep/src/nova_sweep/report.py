"""Flattens per-point result rows and writes the combined sweep report.

One row per (`data_layout` x `index_variant` x `search`) point. Phase 1 writes
a single `output.path/sweep_results.parquet` (no per-rank files — there's only
one process; per-rank naming returns once multi-target
distribution is built).
"""

from __future__ import annotations

import os

from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


def _flatten(prefix: str, d: dict[str, Any], out: dict[str, Any]) -> None:
    """`{"hnsw": {"m": 8}}` under prefix `"index_variant"` -> `{"index_variant.hnsw.m": 8}`.
    Skips the internal `_name` key (surfaced separately as `<prefix>_name`).
    """
    for key, value in d.items():
        if key == "_name":
            continue
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            _flatten(path, value, out)
        else:
            out[path] = value


def build_row(
    *,
    data_layout: dict,
    data_layout_name: str,
    collection_name: str,
    index_variant: dict,
    search: dict,
    summary: dict | None,
    reindex_seconds: float,
    search_seconds: float,
    ok: bool,
    error: str | None,
) -> dict[str, Any]:
    """One flattened report row for a single (data_layout, index_variant,
    search) point. `summary` is nova-storm's `--json` output (already a flat
    dict of its `Summary` fields — see `crates/nova-storm/src/runner.rs`);
    `None` when the point never got as far as a storm run (recorded as an
    error row instead — errors are data, not aborts)."""
    row: dict[str, Any] = {
        "collection_name": collection_name,
        "data_layout_name": data_layout_name,
        "index_variant_name": index_variant.get("_name", "default"),
        "search_name": search.get("_name", "default"),
        "reindex_seconds": reindex_seconds,
        "search_seconds": search_seconds,
        "ok": ok,
        "error": error,
    }
    _flatten("data_layout", data_layout, row)
    _flatten("index_variant", index_variant, row)
    _flatten("search", search, row)
    if summary:
        row.update(summary)
    return row


def write_report(output_path: str, rows: list[dict[str, Any]]) -> str:
    """Write `rows` (a list of possibly-ragged flat dicts — different points
    can have different parameter columns, e.g. one data_layout's `datatype`
    vs another's) to `output_path/sweep_results.parquet`, local or `s3://`."""
    columns: dict[str, list] = {}
    all_keys = {k for row in rows for k in row}
    for key in all_keys:
        columns[key] = [row.get(key) for row in rows]
    table = pa.table(columns)

    if output_path.startswith("s3://"):
        fs, root = pafs.FileSystem.from_uri(output_path)
    else:
        fs, root = pafs.LocalFileSystem(), os.path.abspath(output_path)
        os.makedirs(root, exist_ok=True)

    dest = f"{root.rstrip('/')}/sweep_results.parquet"
    with fs.open_output_stream(dest) as sink:
        pq.write_table(table, sink, compression="snappy")
    return dest
