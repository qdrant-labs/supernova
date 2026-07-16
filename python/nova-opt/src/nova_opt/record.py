"""Writes the tuning run's outputs: one parquet of trial rows (possibly
ragged — same union-of-keys flattening as nova-sweep's report writer) plus a
JSON sidecar of run-level metadata (settings, stats provenance, classifier
LODO metrics) so the run can be re-analyzed or fed back into training."""

from __future__ import annotations

import json
import os

from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


def _open_fs(output_path: str) -> tuple[pafs.FileSystem, str]:
    if output_path.startswith("s3://"):
        return pafs.FileSystem.from_uri(output_path)
    root = os.path.abspath(output_path)
    os.makedirs(root, exist_ok=True)
    return pafs.LocalFileSystem(), root


def write_trials(output_path: str, rows: list[dict[str, Any]]) -> str:
    columns: dict[str, list] = {}
    all_keys = {k for row in rows for k in row}
    for key in sorted(all_keys):
        columns[key] = [row.get(key) for row in rows]
    table = pa.table(columns)
    fs, root = _open_fs(output_path)
    dest = f"{root.rstrip('/')}/opt_trials.parquet"
    with fs.open_output_stream(dest) as sink:
        pq.write_table(table, sink, compression="snappy")
    return dest


def write_run_meta(output_path: str, meta: dict[str, Any]) -> str:
    fs, root = _open_fs(output_path)
    dest = f"{root.rstrip('/')}/opt_run.json"
    with fs.open_output_stream(dest) as sink:
        sink.write(json.dumps(meta, indent=2, default=str).encode())
    return dest
