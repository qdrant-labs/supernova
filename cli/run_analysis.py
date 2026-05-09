"""
Analyze the output of a (distributed) embedding run.

Reads per-rank manifests and scans parquet files at the storage destination,
reporting schema, row count, per-rank throughput, and wall-clock stats.
"""

import json
import logging
import statistics
import sys

from datetime import datetime
from urllib.parse import urlparse

import boto3
import click
import duckdb
import yaml

logger = logging.getLogger(__name__)


def _resolve_destination(config: str | None, path: str | None) -> str:
    if path:
        return path.rstrip("/")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    storage = cfg.get("storage", {})
    stype = storage.get("type", "s3")
    if stype == "s3":
        return f"s3://{storage['bucket']}/{storage['prefix']}".rstrip("/")
    if stype == "local":
        return storage["output_dir"].rstrip("/")
    raise ValueError(f"Unsupported storage type for analysis: {stype}")


def _list_corpus_and_manifests(destination: str) -> tuple[list[str], list[str]]:
    """
    Return (corpus_parquet_uris, manifest_uris) under the destination.

    Corpus parquets come from ``discover_corpus_parquets`` so ``eval/``
    artifacts are filtered out. Manifests are listed separately and the
    same eval filter is applied.
    """
    from vectorforge.destinations import (
        discover_corpus_parquets,
        parse_destination,
    )

    # Accept bare local paths too (legacy --path callers).
    if not destination.startswith(("s3://", "hf://", "file://")):
        destination = f"file://{destination}"

    dest = parse_destination(destination)
    corpus_uris = discover_corpus_parquets(dest)
    manifest_uris = _list_manifests(destination)
    return corpus_uris, manifest_uris


def _list_manifests(destination: str) -> list[str]:
    """List `*_manifest.json` URIs under the destination, excluding ``eval/``."""
    eval_segment = "/eval/"

    if destination.startswith("s3://"):
        parsed = urlparse(destination)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        s3 = boto3.client("s3")
        uris: list[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("_manifest.json"):
                    continue
                if eval_segment in f"/{key}":
                    continue
                uris.append(f"s3://{bucket}/{key}")
        return sorted(uris)

    # file:// or bare path
    import os
    from pathlib import Path

    root = destination[len("file://") :] if destination.startswith("file://") else destination
    eval_path_segment = f"{os.sep}eval{os.sep}"
    return sorted(
        f"file://{p}"
        for p in Path(root).rglob("*_manifest.json")
        if eval_path_segment not in str(p)
    )


def _read_manifests(manifest_uris: list[str]) -> list[dict]:
    manifests = []
    s3 = None
    for uri in manifest_uris:
        if uri.startswith("s3://"):
            if s3 is None:
                s3 = boto3.client("s3")
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            m = json.loads(body)
        else:
            path = uri[len("file://") :] if uri.startswith("file://") else uri
            with open(path) as f:
                m = json.load(f)
        m["_key"] = uri
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


@click.command(name="analysis", help="Analyze a (distributed) embedding run.")
@click.argument("config", required=False)
@click.option(
    "--path",
    default=None,
    help="Override: direct s3://bucket/prefix or local dir to analyze.",
)
@click.option(
    "--cost-per-hour",
    type=float,
    default=0.38,
    show_default=True,
    help="Per-worker hourly cost in USD (default: g5.xlarge A10G spot).",
)
@click.option(
    "--check-duplicates",
    is_flag=True,
    help="Check source_row_id uniqueness and report any duplicate or missing rows.",
)
def analysis(config, path, cost_per_hour, check_duplicates):
    """Analyze a vectorforge embedding run."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("vectorforge").setLevel(logging.INFO)

    if not config and not path:
        raise click.UsageError("Provide a config path or --path")

    destination = _resolve_destination(config, path)
    click.echo(f"Analyzing: {destination}\n")

    parquet_uris, manifest_uris = _list_corpus_and_manifests(destination)
    click.echo(
        f"Found {len(parquet_uris)} parquet files, {len(manifest_uris)} manifests"
    )
    if not parquet_uris:
        click.echo("No parquet files found. Nothing to analyze.")
        sys.exit(1)

    # schema + row count
    con = duckdb.connect()
    if destination.startswith("s3://"):
        _configure_duckdb_for_s3(con)

    # Pass an explicit list to read_parquet rather than a glob, so eval/ files
    # filtered out by discover_corpus_parquets stay filtered.
    files_literal = "[" + ", ".join(f"'{u}'" for u in parquet_uris) + "]"

    click.echo("\n=== Schema ===")
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({files_literal})"
    ).fetchall()
    for col in schema:
        name, dtype = col[0], col[1]
        click.echo(f"  {name:30s} {dtype}")

    row_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({files_literal})"
    ).fetchone()[0]
    click.echo(f"\nTotal rows: {row_count:,}")
    click.echo(f"Parquet files: {len(parquet_uris)}")

    manifests = _read_manifests(manifest_uris) if manifest_uris else []
    manifests.sort(key=lambda m: m.get("_key", ""))

    if not manifests:
        click.echo("\nNo manifests found, skipping throughput analysis.")
        return

    click.echo("\n=== Per-rank manifests ===")

    rps_values = []
    elapsed_values = []
    records_values = []
    starts = []
    ends = []

    click.echo(f"  {'rank':>6}  {'records':>12}  {'elapsed_s':>10}  {'rec/s':>10}")
    for m in manifests:
        key = m["_key"].rsplit("/", 1)[-1]
        records = m.get("total_records", 0)
        elapsed = m.get("elapsed_seconds", 0.0)
        rps = m.get("records_per_second", 0.0)
        click.echo(f"  {key[:6]:>6}  {records:>12,}  {elapsed:>10.1f}  {rps:>10.1f}")

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
    click.echo("\n=== Aggregate ===")
    click.echo(f"  jobs:            {len(manifests)}")
    click.echo(f"  total records:   {sum(records_values):,}")
    if rps_values:
        click.echo(f"  rows/s  min:    {min(rps_values):.1f}")
        click.echo(f"  rows/s  max:    {max(rps_values):.1f}")
        click.echo(f"  rows/s  mean:   {statistics.mean(rps_values):.1f}")
        click.echo(f"  rows/s  median: {statistics.median(rps_values):.1f}")
    if elapsed_values:
        click.echo(f"  elapsed min:    {min(elapsed_values):.1f}s")
        click.echo(f"  elapsed max:    {max(elapsed_values):.1f}s")
        click.echo(f"  elapsed mean:   {statistics.mean(elapsed_values):.1f}s")
    if starts and ends:
        wall = max(e.timestamp() for e in ends) - min(starts)
        click.echo(f"  wall clock:     {wall:.1f}s ({wall / 60:.1f} min)")
        click.echo(
            f"  sum cpu time:   {sum(elapsed_values):.1f}s (parallel speedup: {sum(elapsed_values) / wall:.1f}x)"
        )

    if elapsed_values:
        total_worker_hours = sum(elapsed_values) / 3600.0
        est_cost = total_worker_hours * cost_per_hour
        click.echo(f"\n=== Cost estimate (at ${cost_per_hour:.2f}/hr per worker) ===")
        click.echo(f"  worker-hours:   {total_worker_hours:.2f}")
        click.echo(f"  estimated cost: ${est_cost:.2f}")
        if sum(records_values):
            click.echo(
                f"  cost per 1M rec: ${(est_cost / sum(records_values) * 1_000_000):.2f}"
            )
        click.echo(
            "  note: sums per-job elapsed time; doesn't include idle worker time between jobs"
        )

    click.echo("\n=== rows/s distribution ===")
    click.echo(_ascii_histogram(rps_values, bins=10))

    if check_duplicates:
        click.echo("\n=== Duplicate / coverage check (source_row_id) ===")
        stats = con.execute(f"""
            SELECT
                COUNT(*)                          AS total_rows,
                COUNT(DISTINCT source_row_id)     AS unique_source_rows,
                COUNT(*) - COUNT(DISTINCT source_row_id) AS duplicate_count,
                MIN(source_row_id)                AS min_source_row_id,
                MAX(source_row_id)                AS max_source_row_id
            FROM read_parquet({files_literal})
        """).fetchone()
        total, unique, dupes, min_id, max_id = stats
        click.echo(f"  total rows:         {total:,}")
        click.echo(f"  unique source_row_id: {unique:,}")
        click.echo(
            f"  duplicates:         {dupes:,}  {'✓ none' if dupes == 0 else '✗ FOUND'}"
        )
        click.echo(
            f"  source_row_id range: [{min_id:,}, {max_id:,}]  (span: {max_id - min_id + 1:,})"
        )
        coverage_gap = (max_id - min_id + 1) - unique
        click.echo(
            f"  gaps in range:      {coverage_gap:,}  {'✓ none' if coverage_gap == 0 else '(missing rows)'}"
        )

        if dupes > 0:
            click.echo("\n  First 10 duplicated source_row_ids:")
            rows = con.execute(f"""
                SELECT source_row_id, COUNT(*) AS cnt
                FROM read_parquet({files_literal})
                GROUP BY source_row_id
                HAVING cnt > 1
                ORDER BY cnt DESC
                LIMIT 10
            """).fetchall()
            for src_id, cnt in rows:
                click.echo(f"    source_row_id={src_id:,}  appears {cnt}x")


if __name__ == "__main__":
    analysis()
