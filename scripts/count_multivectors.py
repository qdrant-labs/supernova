#!/usr/bin/env python3
"""
Count total multi-vector tokens in an S3 prefix of vectorforge parquets.

Sums len(multivector_embedding) across every row of every parquet under the
given prefix. Also reports doc count, parquet count, and min/mean/max N per
row.

Iterates parquets one-at-a-time and uses pyarrow's list-length API so only
the list-offset data is read -- the giant float values never enter memory.
Works on arbitrarily large datasets.

Usage:
  python scripts/count_multivectors.py s3://qdrant--vectorforge/finewiki/embed-bge-m3/en/
  python scripts/count_multivectors.py s3://bucket/prefix --column multivector_embedding
  python scripts/count_multivectors.py s3://bucket/prefix --per-file
"""

from __future__ import annotations

import argparse
import sys

from urllib.parse import urlparse

import boto3
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem


def _list_parquets(prefix: str) -> tuple[list[str], S3FileSystem | None]:
    """
    Return (list of parquet paths, pyarrow filesystem). Paths are bucket-relative.
    """
    if not prefix.startswith("s3://"):
        # local dir
        from pathlib import Path

        root = Path(prefix)
        return sorted(str(p) for p in root.rglob("*.parquet")), None

    parsed = urlparse(prefix)
    bucket = parsed.netloc
    key_prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    s3 = boto3.client("s3")
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(f"{bucket}/{obj['Key']}")
    return sorted(keys), S3FileSystem()


def main():
    parser = argparse.ArgumentParser(
        description="Count multi-vector tokens across parquets in an S3 prefix."
    )
    parser.add_argument(
        "prefix", help="S3 prefix (e.g. s3://bucket/path/to/dir/) or local directory."
    )
    parser.add_argument(
        "--column",
        default="multivector_embedding",
        help="Multi-vector column name. Default: multivector_embedding.",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print stats per parquet file in addition to the total.",
    )
    args = parser.parse_args()

    prefix = args.prefix.rstrip("/")
    print(f"Scanning: {prefix}/*.parquet")

    parquets, fs = _list_parquets(prefix)
    if not parquets:
        print("No parquet files found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(parquets)} parquet files.\n")

    total_docs = 0
    total_vectors = 0
    min_n = None
    max_n = 0

    if args.per_file:
        print(
            f"  {'file':<60} {'docs':>10} {'vectors':>15} {'min':>5} {'avg':>8} {'max':>6}"
        )

    for path in parquets:
        pf = pq.ParquetFile(path, filesystem=fs)
        file_docs = 0
        file_vectors = 0
        file_min = None
        file_max = 0

        # column projection + small batches -- each batch holds at most
        # batch_size * ~1MB (bge-m3 multivector) so 500 rows ≈ 500MB per batch.
        # The full vector data is materialized per batch but drops between batches.
        for batch in pf.iter_batches(columns=[args.column], batch_size=500):
            col = batch.column(args.column)
            # value_lengths() returns N per row without materializing the inner floats
            lengths = col.value_lengths().to_numpy()
            file_docs += len(lengths)
            batch_sum = int(lengths.sum())
            file_vectors += batch_sum
            batch_min = int(lengths.min()) if len(lengths) else 0
            batch_max = int(lengths.max()) if len(lengths) else 0
            file_min = batch_min if file_min is None else min(file_min, batch_min)
            file_max = max(file_max, batch_max)

        total_docs += file_docs
        total_vectors += file_vectors
        min_n = (
            file_min
            if min_n is None
            else min(min_n, file_min if file_min is not None else min_n)
        )
        max_n = max(max_n, file_max)

        if args.per_file:
            name = path.rsplit("/", 1)[-1]
            avg = file_vectors / file_docs if file_docs else 0
            print(
                f"  {name[:60]:<60} {file_docs:>10,} {file_vectors:>15,} {file_min or 0:>5} {avg:>8.1f} {file_max:>6}"
            )

    avg_n = total_vectors / total_docs if total_docs else 0

    if args.per_file:
        print()
    print(f"  parquet files:       {len(parquets):>15,}")
    print(f"  total docs:          {total_docs:>15,}")
    print(f"  total vectors:       {total_vectors:>15,}  (sum across all docs)")
    print()
    print(f"  vectors/doc min:     {min_n or 0:>15,}")
    print(f"  vectors/doc avg:     {avg_n:>15,.1f}")
    print(f"  vectors/doc max:     {max_n:>15,}")


if __name__ == "__main__":
    main()
