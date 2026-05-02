"""
Analyze the output of a (distributed) embedding run.

Reads per-rank manifests and scans parquet files at the storage destination,
reporting schema, row count, per-rank throughput, and wall-clock stats.

Usage:
  vf analysis configs/embedder/google_extended_amazon.yaml
  vf analysis --path s3://qdrant--vectorforge/google--extended-amazon/embed-gte-multilingual-base
"""

import argparse
import json
import logging
import statistics
import sys

from datetime import datetime
from urllib.parse import urlparse

import boto3
import duckdb
import yaml

logger = logging.getLogger(__name__)


def _resolve_destination(args) -> str:
    if args.path:
        return args.path.rstrip("/")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    storage = cfg.get("storage", {})
    stype = storage.get("type", "s3")
    if stype == "s3":
        return f"s3://{storage['s3_bucket']}/{storage['s3_prefix']}".rstrip("/")
    if stype == "local":
        return storage["output_dir"].rstrip("/")
    raise ValueError(f"Unsupported storage type for analysis: {stype}")


def _list_keys(destination: str) -> tuple[list[str], list[str]]:
    """
    Return (parquet_keys, manifest_keys) under the destination.
    """
    if destination.startswith("s3://"):
        parsed = urlparse(destination)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        s3 = boto3.client("s3")

        parquets, manifests = [], []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    parquets.append(key)
                elif key.endswith("_manifest.json"):
                    manifests.append(key)
        return parquets, manifests

    # local
    from pathlib import Path
    root = Path(destination)
    parquets = [str(p) for p in root.rglob("*.parquet")]
    manifests = [str(p) for p in root.rglob("*_manifest.json")]
    return parquets, manifests


def _read_manifests(destination: str, manifest_keys: list[str]) -> list[dict]:
    manifests = []
    if destination.startswith("s3://"):
        parsed = urlparse(destination)
        bucket = parsed.netloc
        s3 = boto3.client("s3")
        for key in manifest_keys:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            m = json.loads(body)
            m["_key"] = key
            manifests.append(m)
    else:
        for path in manifest_keys:
            with open(path) as f:
                m = json.load(f)
                m["_key"] = path
                manifests.append(m)
    return manifests


def _ascii_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  all {len(values)} values = {lo:.1f}"

    bin_width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts)
    lines = []
    for i, c in enumerate(counts):
        edge_lo = lo + i * bin_width
        edge_hi = edge_lo + bin_width
        bar = "█" * int(width * c / max_count) if max_count else ""
        lines.append(f"  [{edge_lo:8.1f} – {edge_hi:8.1f})  {bar} {c}")
    return "\n".join(lines)


def _configure_duckdb_for_s3(con: duckdb.DuckDBPyConnection):
    """
    Install + load httpfs, configure AWS credentials for S3 reads.
    """
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # duckdb picks up default AWS credential chain via the aws extension
    try:
        con.execute("INSTALL aws; LOAD aws;")
        con.execute("CALL load_aws_credentials();")
    except duckdb.Error as e:
        logger.warning("Could not load AWS credentials via duckdb aws ext: %s", e)


def main(argv: list[str] | None = None):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Analyze a vectorforge embedding run")
    parser.add_argument("config", nargs="?", help="Path to embedder YAML config (derives destination from storage section)")
    parser.add_argument("--path", help="Override: direct s3://bucket/prefix or local dir to analyze")
    parser.add_argument(
        "--cost-per-hour",
        type=float,
        default=0.38,
        help="Per-worker hourly cost in USD (default: 0.38 = g5.xlarge A10G spot)",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Check source_row_id uniqueness and report any duplicate or missing rows",
    )
    args = parser.parse_args(argv)

    if not args.config and not args.path:
        parser.error("Provide a config path or --path")

    destination = _resolve_destination(args)
    print(f"Analyzing: {destination}\n")

    parquet_keys, manifest_keys = _list_keys(destination)
    print(f"Found {len(parquet_keys)} parquet files, {len(manifest_keys)} manifests")
    if not parquet_keys:
        print("No parquet files found. Nothing to analyze.")
        sys.exit(1)

    # schema + row count
    con = duckdb.connect()
    if destination.startswith("s3://"):
        _configure_duckdb_for_s3(con)
    # pass both globs so the same script works for flat layouts (all parquets at the
    # top of the prefix) and sharded layouts (rank00/batch_*.parquet).
    globs = f"'{destination}/**/*.parquet'"

    print("\n=== Schema ===")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet({globs})").fetchall()
    for col in schema:
        name, dtype = col[0], col[1]
        print(f"  {name:30s} {dtype}")

    row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet({globs})").fetchone()[0]
    print(f"\nTotal rows: {row_count:,}")
    print(f"Parquet files: {len(parquet_keys)}")

    manifests = _read_manifests(destination, manifest_keys) if manifest_keys else []
    manifests.sort(key=lambda m: m.get("_key", ""))

    if not manifests:
        print("\nNo manifests found, skipping throughput analysis.")
        return

    print("\n=== Per-rank manifests ===")

    rps_values = []
    elapsed_values = []
    records_values = []
    starts = []
    ends = []

    print(f"  {'rank':>6}  {'records':>12}  {'elapsed_s':>10}  {'rec/s':>10}")
    for m in manifests:
        key = m["_key"].rsplit("/", 1)[-1]
        records = m.get("total_records", 0)
        elapsed = m.get("elapsed_seconds", 0.0)
        rps = m.get("records_per_second", 0.0)
        print(f"  {key[:6]:>6}  {records:>12,}  {elapsed:>10.1f}  {rps:>10.1f}")

        rps_values.append(rps)
        elapsed_values.append(elapsed)
        records_values.append(records)

        end = m.get("created_at")
        if end and elapsed:
            try:
                end_dt = datetime.fromisoformat(end)
                ends.append(end_dt)
                starts.append(end_dt.timestamp() - elapsed)
            except ValueError:
                pass

    # aggregate stats
    print("\n=== Aggregate ===")
    print(f"  jobs:            {len(manifests)}")
    print(f"  total records:   {sum(records_values):,}")
    if rps_values:
        print(f"  rows/s  min:    {min(rps_values):.1f}")
        print(f"  rows/s  max:    {max(rps_values):.1f}")
        print(f"  rows/s  mean:   {statistics.mean(rps_values):.1f}")
        print(f"  rows/s  median: {statistics.median(rps_values):.1f}")
    if elapsed_values:
        print(f"  elapsed min:    {min(elapsed_values):.1f}s")
        print(f"  elapsed max:    {max(elapsed_values):.1f}s")
        print(f"  elapsed mean:   {statistics.mean(elapsed_values):.1f}s")
    if starts and ends:
        wall = max(e.timestamp() for e in ends) - min(starts)
        print(f"  wall clock:     {wall:.1f}s ({wall/60:.1f} min)")
        print(f"  sum cpu time:   {sum(elapsed_values):.1f}s (parallel speedup: {sum(elapsed_values)/wall:.1f}x)")

    if elapsed_values:
        total_worker_hours = sum(elapsed_values) / 3600.0
        est_cost = total_worker_hours * args.cost_per_hour
        print(f"\n=== Cost estimate (at ${args.cost_per_hour:.2f}/hr per worker) ===")
        print(f"  worker-hours:   {total_worker_hours:.2f}")
        print(f"  estimated cost: ${est_cost:.2f}")
        print(f"  cost per 1M rec: ${(est_cost / sum(records_values) * 1_000_000):.2f}" if sum(records_values) else "")
        print("  note: sums per-job elapsed time; doesn't include idle worker time between jobs")

    print("\n=== rows/s distribution ===")
    print(_ascii_histogram(rps_values, bins=10))

    if args.check_duplicates:
        print("\n=== Duplicate / coverage check (source_row_id) ===")
        stats = con.execute(f"""
            SELECT
                COUNT(*)                          AS total_rows,
                COUNT(DISTINCT source_row_id)     AS unique_source_rows,
                COUNT(*) - COUNT(DISTINCT source_row_id) AS duplicate_count,
                MIN(source_row_id)                AS min_source_row_id,
                MAX(source_row_id)                AS max_source_row_id
            FROM read_parquet({globs})
        """).fetchone()
        total, unique, dupes, min_id, max_id = stats
        print(f"  total rows:         {total:,}")
        print(f"  unique source_row_id: {unique:,}")
        print(f"  duplicates:         {dupes:,}  {'✓ none' if dupes == 0 else '✗ FOUND'}")
        print(f"  source_row_id range: [{min_id:,}, {max_id:,}]  (span: {max_id - min_id + 1:,})")
        coverage_gap = (max_id - min_id + 1) - unique
        print(f"  gaps in range:      {coverage_gap:,}  {'✓ none' if coverage_gap == 0 else '(missing rows)'}")

        if dupes > 0:
            print("\n  First 10 duplicated source_row_ids:")
            rows = con.execute(f"""
                SELECT source_row_id, COUNT(*) AS cnt
                FROM read_parquet({globs})
                GROUP BY source_row_id
                HAVING cnt > 1
                ORDER BY cnt DESC
                LIMIT 10
            """).fetchall()
            for src_id, cnt in rows:
                print(f"    source_row_id={src_id:,}  appears {cnt}x")


if __name__ == "__main__":
    main()