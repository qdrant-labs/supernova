"""
Sample N rows uniformly at random from a parquet corpus and save them
to a single local parquet file.

This is the programmatic core of `vf subsample`. It shares the same
discovery / manifest / fetch helpers as `generate_queries`; the only
behavioral difference is that the result is written locally and not
pushed back to the corpus under `eval/`.
"""

from __future__ import annotations

import logging
import os
import tempfile

from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from vectorforge.destinations import (
    bare_key_for_uri,
    discover_corpus_parquets,
    parse_destination,
)
from vectorforge.eval.generate_queries import (
    build_manifest,
    fetch_file_rows_prefetch,
    fetch_file_rows_remote,
    sample_offsets,
)

logger = logging.getLogger(__name__)


def subsample(
    corpus_uri: str,
    n: int,
    seed: int,
    columns: list[str] | None,
    output: str,
    prefetch: bool = False,
) -> str:
    """
    Sample ``n`` rows uniformly at random from the parquet corpus at
    ``corpus_uri`` and write them to the local file ``output``.

    The output table carries provenance columns ``__source_file__`` (bare
    key of the source parquet) and ``__source_row__`` (local row offset
    within that file) appended after the requested ``columns``.

    Returns the absolute path of the written file.
    """
    dest = parse_destination(corpus_uri)
    logger.info("Listing parquets at %s/...", dest.root_uri)
    uris = discover_corpus_parquets(dest)
    if not uris:
        raise ValueError(f"No parquet files found at {corpus_uri}.")
    logger.info("Found %d parquet files", len(uris))

    logger.info("Reading metadata (parquet footers)...")
    manifest, total_rows = build_manifest(uris)
    logger.info("Total rows: %d", total_rows)

    actual_n = min(n, total_rows)
    if actual_n < n:
        logger.warning("Only %d rows available, capping at %d.", total_rows, actual_n)

    logger.info("Sampling %d row pointers (seed=%d)...", actual_n, seed)
    file_map = sample_offsets(manifest, total_rows, actual_n, seed)
    total_files = len(file_map)

    mode = "prefetch (download-first)" if prefetch else "range requests"
    logger.info("Fetching rows from %d files (%s)...", total_files, mode)
    all_slices: list[pa.Table] = []

    with tqdm(total=total_files, unit="file", dynamic_ncols=True) as bar:
        if prefetch:
            with tempfile.TemporaryDirectory() as tmpdir:
                for src_uri, offsets in file_map.items():
                    src_key = bare_key_for_uri(src_uri)
                    for lo, t in fetch_file_rows_prefetch(
                        src_uri, offsets, columns, tmpdir
                    ):
                        t = t.append_column("__source_file__", pa.array([src_key]))
                        t = t.append_column(
                            "__source_row__", pa.array([lo], type=pa.int64())
                        )
                        all_slices.append(t)
                    bar.update(1)
                    bar.set_postfix_str(src_key, refresh=False)
        else:
            for src_uri, offsets in file_map.items():
                src_key = bare_key_for_uri(src_uri)
                for lo, t in fetch_file_rows_remote(src_uri, offsets, columns):
                    t = t.append_column("__source_file__", pa.array([src_key]))
                    t = t.append_column(
                        "__source_row__", pa.array([lo], type=pa.int64())
                    )
                    all_slices.append(t)
                bar.update(1)
                bar.set_postfix_str(src_key, refresh=False)

    result = pa.concat_tables(all_slices)
    logger.info("Collected %d rows", len(result))

    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pq.write_table(result, output, compression="snappy")
    abs_path = os.path.abspath(output)
    logger.info("Wrote %s", abs_path)
    return abs_path