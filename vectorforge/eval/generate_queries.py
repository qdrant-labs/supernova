"""
Sample N rows from an embedded corpus as eval query vectors.

This is the programmatic core of `vf generate-queries`. The CLI in
``cli/run_generate_queries.py`` only wraps argparse + the SkyPilot/EC2
launcher around ``generate_queries``.

Output: ``{corpus_uri}/eval/queries_<n>.parquet``
  All requested parquet columns (dense_embedding, sparse_embedding, text, …)
  plus ``__source_file__`` (bare key) and ``__source_row__`` for provenance.
"""

from __future__ import annotations

import bisect
import logging
import os
import random
import tempfile
from collections import defaultdict

from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.destinations import (
    bare_key_for_uri,
    discover_corpus_parquets,
    filesystem_for_uri,
    fs_path_for_uri,
    parse_destination,
    upload_bytes_to_uri,
)

logger = logging.getLogger(__name__)


def build_manifest(uris: list[str]) -> tuple[list[dict], int]:
    """Read parquet footers only — range requests, no data download."""
    manifest = []
    global_offset = 0
    for i, uri in enumerate(uris):
        logger.info("[%d/%d] metadata: %s", i + 1, len(uris), uri)
        fs = filesystem_for_uri(uri)
        fs_path = fs_path_for_uri(uri)
        meta = pq.read_metadata(fs_path, filesystem=fs)
        manifest.append({
            "uri": uri,
            "global_start": global_offset,
            "row_count": meta.num_rows,
        })
        global_offset += meta.num_rows
    return manifest, global_offset


def sample_offsets(
    manifest: list[dict], total_rows: int, n: int, seed: int,
) -> dict[str, list[int]]:
    """Sample n global indices, return as {file_uri: [local_offsets]}."""
    rng = random.Random(seed)
    global_indices = sorted(rng.sample(range(total_rows), n))
    cum_ends = [e["global_start"] + e["row_count"] for e in manifest]
    file_map: dict[str, list[int]] = defaultdict(list)
    for gi in global_indices:
        fi = bisect.bisect_right(cum_ends, gi)
        entry = manifest[fi]
        file_map[entry["uri"]].append(gi - entry["global_start"])
    return dict(file_map)


def _extract_rows(
    pf: pq.ParquetFile,
    row_offsets: list[int],
    columns: list[str] | None,
) -> list[tuple[int, pa.Table]]:
    """Read only the row groups needed and extract the requested rows."""
    rg_starts = []
    cum = 0
    for rg_i in range(pf.metadata.num_row_groups):
        rg_starts.append(cum)
        cum += pf.metadata.row_group(rg_i).num_rows
    rg_to_offsets: dict[int, list[int]] = defaultdict(list)
    for lo in row_offsets:
        rg_idx = bisect.bisect_right(rg_starts, lo) - 1
        rg_to_offsets[rg_idx].append(lo)
    results = []
    for rg_idx in sorted(rg_to_offsets):
        rg_start = rg_starts[rg_idx]
        table = pf.read_row_group(rg_idx, columns=columns)
        for lo in rg_to_offsets[rg_idx]:
            sliced = table.slice(lo - rg_start, 1)
            # slice() is zero-copy and keeps the entire row group buffer alive.
            # Rebuild each column from Python values to break that reference.
            copied = pa.table({
                name: pa.array(sliced.column(name).to_pylist(), type=sliced.schema.field(name).type)
                for name in sliced.schema.names
            })
            results.append((lo, copied))
    return results


def fetch_file_rows_remote(
    uri: str,
    row_offsets: list[int],
    columns: list[str] | None,
) -> list[tuple[int, pa.Table]]:
    """Range-request mode: fetch only needed row groups via the URI's filesystem."""
    fs = filesystem_for_uri(uri)
    fs_path = fs_path_for_uri(uri)
    pf = pq.ParquetFile(fs_path, filesystem=fs)
    return _extract_rows(pf, row_offsets, columns)


def fetch_file_rows_prefetch(
    uri: str,
    row_offsets: list[int],
    columns: list[str] | None,
    tmpdir: str,
) -> list[tuple[int, pa.Table]]:
    """Prefetch mode: download full file first, read locally, then delete."""
    safe_name = uri.replace("/", "_").replace(":", "_")
    local_path = os.path.join(tmpdir, safe_name)
    fs = filesystem_for_uri(uri)
    fs_path = fs_path_for_uri(uri)
    # fsspec / pyarrow filesystem both support open_input_file → read; for
    # download we use fsspec's get_file when available, else read+write.
    with fs.open_input_file(fs_path) if hasattr(fs, "open_input_file") else fs.open(fs_path, "rb") as src:
        with open(local_path, "wb") as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    try:
        return _extract_rows(pq.ParquetFile(local_path), row_offsets, columns)
    finally:
        os.remove(local_path)


def generate_queries(
    corpus_uri: str,
    n: int,
    seed: int,
    columns: list[str] | None,
    output: str,
    prefetch: bool = False,
):
    """
    Sample ``n`` rows uniformly at random from the corpus at ``corpus_uri``
    and write them to ``{corpus_uri}/eval/{output}`` with provenance columns
    ``__source_file__`` and ``__source_row__`` appended.
    """
    dest = parse_destination(corpus_uri)
    logger.info("Listing parquets at %s/...", dest.root_uri)
    uris = discover_corpus_parquets(dest)
    if not uris:
        logger.warning("No parquet files found.")
        return
    logger.info("Found %d parquet files", len(uris))

    logger.info("Reading metadata (parquet footers)...")
    manifest, total_rows = build_manifest(uris)
    logger.info("Total rows: %d", total_rows)

    actual_n = min(n, total_rows)
    if actual_n < n:
        logger.warning("Only %d rows available, capping at %d.", total_rows, actual_n)

    logger.info("Sampling %d query pointers (seed=%d)...", actual_n, seed)
    file_map = sample_offsets(manifest, total_rows, actual_n, seed)
    total_files = len(file_map)

    mode = "prefetch (download-first)" if prefetch else "range requests"
    logger.info("Fetching rows from %d files (%s)...", total_files, mode)
    all_slices = []

    with tqdm(total=total_files, unit="file", dynamic_ncols=True) as bar:
        if prefetch:
            with tempfile.TemporaryDirectory() as tmpdir:
                for src_uri, offsets in file_map.items():
                    src_key = bare_key_for_uri(src_uri)
                    for lo, t in fetch_file_rows_prefetch(src_uri, offsets, columns, tmpdir):
                        t = t.append_column("__source_file__", pa.array([src_key]))
                        t = t.append_column("__source_row__", pa.array([lo], type=pa.int64()))
                        all_slices.append(t)
                    bar.update(1)
                    bar.set_postfix_str(src_key, refresh=False)
        else:
            for src_uri, offsets in file_map.items():
                src_key = bare_key_for_uri(src_uri)
                for lo, t in fetch_file_rows_remote(src_uri, offsets, columns):
                    t = t.append_column("__source_file__", pa.array([src_key]))
                    t = t.append_column("__source_row__", pa.array([lo], type=pa.int64()))
                    all_slices.append(t)
                bar.update(1)
                bar.set_postfix_str(src_key, refresh=False)

    result = pa.concat_tables(all_slices)
    logger.info("Collected %d rows", len(result))

    pq.write_table(result, output, compression="snappy")
    logger.info("Wrote %s", output)

    buf = pa.BufferOutputStream()
    pq.write_table(result, buf, compression="snappy")
    eval_uri = dest.eval_uri(output)
    upload_bytes_to_uri(bytes(buf.getvalue()), eval_uri)
    logger.info("Pushed to %s", eval_uri)
