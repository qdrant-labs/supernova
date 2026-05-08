#!/usr/bin/env python3
"""Verify there are no duplicate rows under an S3 parquet prefix.

Streams every parquet file under the prefix one at a time, hashes each row
(via DuckDB's built-in hash() so we don't pay Python-per-row cost), and
tracks two collision categories:

  - row_id collisions: same id seen in different files. These would clobber
    each other on Qdrant upsert.
  - content collisions: same hash of (text, dense_embedding) seen twice.
    Indicates a re-embedded duplicate article, not necessarily a load bug.

Usage:
  uv run python scripts/verify_no_duplicates.py s3://qdrant--vectorforge/stanford-oval--ccnews/baai_bge_large_en_v1.5/

  # Custom columns
  uv run python scripts/verify_no_duplicates.py s3://bucket/prefix \\
      --id-column row_id \\
      --content-columns text dense_embedding
"""

import argparse
import os
import sys

from collections import defaultdict

import boto3
import duckdb

from tqdm import tqdm


def list_parquet_files(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    files: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                files.append(f"s3://{bucket}/{obj['Key']}")
    return sorted(files)


def configure_duckdb(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    region = os.environ.get(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    conn.execute(f"SET s3_region = '{region}';")

    key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if key and secret:
        conn.execute(f"SET s3_access_key_id = '{key}';")
        conn.execute(f"SET s3_secret_access_key = '{secret}';")

    token = os.environ.get("AWS_SESSION_TOKEN", "")
    if token:
        conn.execute(f"SET s3_session_token = '{token}';")

    # Same defaults as the loader -- avoid OOMing on a laptop.
    conn.execute("SET memory_limit = '2GB';")
    conn.execute("SET threads = 2;")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3://bucket/prefix, got {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "s3_uri", help="s3://bucket/prefix (recursively scanned for *.parquet)"
    )
    p.add_argument(
        "--id-column",
        default="row_id",
        help="Column whose uniqueness defines a row id (default: row_id)",
    )
    p.add_argument(
        "--content-columns",
        nargs="+",
        default=["text", "dense_embedding"],
        help="Columns to hash for content-duplicate detection. Pass --content-columns '' to skip.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per fetchmany (default: 10000)",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Number of duplicate examples to print (default: 10)",
    )
    args = p.parse_args(argv)

    bucket, prefix = parse_s3_uri(args.s3_uri)
    print(f"Listing parquets in s3://{bucket}/{prefix}/...")
    files = list_parquet_files(bucket, prefix)
    if not files:
        print("No parquet files found at that prefix.")
        return 1
    print(f"Found {len(files):,} parquet files")

    content_cols = [c for c in args.content_columns if c]
    track_content = bool(content_cols)
    if track_content:
        # DuckDB's hash() takes any types and returns a UBIGINT (uint64).
        content_hash_sql = f"hash({', '.join(content_cols)})"
    else:
        content_hash_sql = "NULL"

    conn = duckdb.connect()
    configure_duckdb(conn)

    seen_ids: set = set()
    seen_content: set = set()
    duplicate_id_counts: dict = defaultdict(int)
    duplicate_id_examples: dict = {}  # rid -> first file where second occurrence was seen
    duplicate_content_count = 0
    total_rows = 0
    file_errors: list[tuple[str, str]] = []

    pbar = tqdm(files, desc="Files", unit="file")
    for file_uri in pbar:
        sql = (
            f"SELECT {args.id_column}, {content_hash_sql} "
            f"FROM read_parquet('{file_uri}')"
        )
        try:
            conn.execute(sql)
        except Exception as e:
            file_errors.append((file_uri, str(e)))
            continue

        while True:
            rows = conn.fetchmany(args.batch_size)
            if not rows:
                break
            for rid, content_hash in rows:
                total_rows += 1
                if rid in seen_ids:
                    duplicate_id_counts[rid] += 1
                    if rid not in duplicate_id_examples:
                        duplicate_id_examples[rid] = file_uri
                else:
                    seen_ids.add(rid)
                if track_content and content_hash is not None:
                    if content_hash in seen_content:
                        duplicate_content_count += 1
                    else:
                        seen_content.add(content_hash)

        pbar.set_postfix(
            {
                "rows": f"{total_rows:,}",
                "dup_ids": len(duplicate_id_counts),
                "dup_content": duplicate_content_count,
            }
        )

    pbar.close()

    print()
    print("=" * 60)
    print(f"Total rows scanned:        {total_rows:,}")
    print(f"Unique row_ids ({args.id_column}): {len(seen_ids):,}")
    if track_content:
        print(f"Unique content hashes:     {len(seen_content):,}")
    print(
        f"Duplicate row_ids:         {len(duplicate_id_counts):,}"
        f" (with {sum(duplicate_id_counts.values()):,} extra occurrences)"
    )
    if track_content:
        print(f"Duplicate content rows:    {duplicate_content_count:,}")
    if file_errors:
        print(f"Files that failed to scan: {len(file_errors)}")
    print("=" * 60)

    if duplicate_id_counts:
        print(
            f"\nFirst {min(args.max_examples, len(duplicate_id_examples))} duplicate row_id examples:"
        )
        for rid, file_uri in list(duplicate_id_examples.items())[: args.max_examples]:
            extra = duplicate_id_counts[rid]
            print(
                f"  row_id={rid} (saw {extra + 1} times; second occurrence in {file_uri})"
            )

    if file_errors:
        print("\nFile read errors:")
        for file_uri, err in file_errors[:5]:
            print(f"  {file_uri}: {err}")
        if len(file_errors) > 5:
            print(f"  ... and {len(file_errors) - 5} more")

    if duplicate_id_counts or duplicate_content_count or file_errors:
        return 1
    print("\nOK — no duplicates found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
