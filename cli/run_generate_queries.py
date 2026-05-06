#!/usr/bin/env python3
"""
Sample N rows from embedded S3 parquets as eval query vectors.

Default mode launches a single high-bandwidth EC2 instance in the bucket's
region via SkyPilot. The job runs the full pipeline in-region: build index
from parquet footers, fetch the actual row data, push queries.parquet to S3.

Use --local to run the full pipeline in-process (also what the EC2 job calls).

Output: s3://bucket/prefix/queries_<n>.parquet
  All original parquet columns (dense_embedding, sparse_embedding, text, …)
  plus __source_file__ and __source_row__ for provenance.

Usage:
  vf generate-queries s3://bucket/prefix -n 1000
  vf generate-queries s3://bucket/prefix -n 1000 --on-demand
  vf generate-queries s3://bucket/prefix -n 1000 --dry-run
  vf generate-queries s3://bucket/prefix -n 1000 --local             # run here
"""

import argparse
import bisect
import logging
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import yaml

logger = logging.getLogger(__name__)

ENV_VARS_TO_FORWARD = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
]

# "n"-family = enhanced networking (25-100Gbps). S3 reads that take
# minutes locally finish in seconds in-region.
DEFAULT_INSTANCE_TYPE = "r5n.2xlarge"  # 8 vCPU, 64GB RAM, 25Gbps

def list_s3_parquets(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return sorted(keys)


def build_manifest(bucket: str, keys: list[str]) -> tuple[list[dict], int]:
    """Read parquet footers only — range requests, no data download."""
    fs = pafs.S3FileSystem()
    manifest = []
    global_offset = 0
    for i, key in enumerate(keys):
        logger.info("[%d/%d] metadata: %s", i + 1, len(keys), key)
        with fs.open_input_file(f"{bucket}/{key}") as f:
            meta = pq.read_metadata(f)
        manifest.append({
            "key": key,
            "global_start": global_offset,
            "row_count": meta.num_rows,
        })
        global_offset += meta.num_rows
    return manifest, global_offset


def sample_offsets(
    manifest: list[dict], total_rows: int, n: int, seed: int, prefix: str,
) -> dict[str, list[int]]:
    """Sample n global indices, return as {relative_key: [local_offsets]}."""
    rng = random.Random(seed)
    global_indices = sorted(rng.sample(range(total_rows), n))
    cum_ends = [e["global_start"] + e["row_count"] for e in manifest]
    rel_prefix = prefix + "/"
    file_map: dict[str, list[int]] = defaultdict(list)
    for gi in global_indices:
        fi = bisect.bisect_right(cum_ends, gi)
        entry = manifest[fi]
        rel_key = entry["key"].removeprefix(rel_prefix)
        file_map[rel_key].append(gi - entry["global_start"])
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


def fetch_file_rows_s3(
    fs: pafs.S3FileSystem,
    bucket: str,
    prefix: str,
    source_file: str,
    row_offsets: list[int],
    columns: list[str] | None,
) -> list[tuple[int, pa.Table]]:
    """
    Range-request mode: fetch only needed row groups via S3FileSystem.
    """
    with fs.open_input_file(f"{bucket}/{prefix}/{source_file}") as f:
        return _extract_rows(pq.ParquetFile(f), row_offsets, columns)


def fetch_file_rows_prefetch(
    s3_client,
    bucket: str,
    prefix: str,
    source_file: str,
    row_offsets: list[int],
    columns: list[str] | None,
    tmpdir: str,
) -> list[tuple[int, pa.Table]]:
    """
    Prefetch mode: download the full file first (boto3 multipart, saturates
    bandwidth), read locally, then delete. Best when row groups are large.
    """
    key = f"{prefix}/{source_file}"
    local_path = os.path.join(tmpdir, source_file.replace("/", "_"))
    s3_client.download_file(bucket, key, local_path)
    try:
        return _extract_rows(pq.ParquetFile(local_path), row_offsets, columns)
    finally:
        os.remove(local_path)


def run_pipeline(
    bucket: str,
    prefix: str,
    n: int,
    seed: int,
    columns: list[str] | None,
    output: str,
    prefetch: bool = False,
):
    print(f"Listing parquets at s3://{bucket}/{prefix}/...")
    keys = list_s3_parquets(bucket, prefix)
    if not keys:
        print("No parquet files found.")
        return
    print(f"Found {len(keys)} parquet files")

    print("Reading metadata (parquet footers)...")
    manifest, total_rows = build_manifest(bucket, keys)
    print(f"Total rows: {total_rows:,}")

    actual_n = min(n, total_rows)
    if actual_n < n:
        print(f"Warning: only {total_rows} rows available, capping at {actual_n}.")

    print(f"Sampling {actual_n} query pointers (seed={seed})...")
    file_map = sample_offsets(manifest, total_rows, actual_n, seed, prefix)
    total_files = len(file_map)

    mode = "prefetch (download-first)" if prefetch else "range requests"
    print(f"Fetching rows from {total_files} files ({mode})...")
    all_slices = []

    with tqdm(total=total_files, unit="file", dynamic_ncols=True) as bar:
        if prefetch:
            s3_client = boto3.client("s3")
            with tempfile.TemporaryDirectory() as tmpdir:
                for src, offsets in file_map.items():
                    for lo, t in fetch_file_rows_prefetch(s3_client, bucket, prefix, src, offsets, columns, tmpdir):
                        t = t.append_column("__source_file__", pa.array([src]))
                        t = t.append_column("__source_row__", pa.array([lo], type=pa.int64()))
                        all_slices.append(t)
                    bar.update(1)
                    bar.set_postfix_str(src, refresh=False)
        else:
            fs = pafs.S3FileSystem()
            for src, offsets in file_map.items():
                for lo, t in fetch_file_rows_s3(fs, bucket, prefix, src, offsets, columns):
                    t = t.append_column("__source_file__", pa.array([src]))
                    t = t.append_column("__source_row__", pa.array([lo], type=pa.int64()))
                    all_slices.append(t)
                bar.update(1)
                bar.set_postfix_str(src, refresh=False)

    result = pa.concat_tables(all_slices)
    print(f"Collected {len(result)} rows")

    pq.write_table(result, output, compression="snappy")
    print(f"Wrote {output}")

    buf = pa.BufferOutputStream()
    pq.write_table(result, buf, compression="snappy")
    s3_key = f"{prefix}/{output}"
    boto3.client("s3").put_object(
        Bucket=bucket, Key=s3_key, Body=bytes(buf.getvalue())
    )
    print(f"Pushed to s3://{bucket}/{s3_key}")


def get_bucket_region(bucket: str) -> str:
    resp = boto3.client("s3").get_bucket_location(Bucket=bucket)
    return resp["LocationConstraint"] or "us-east-1"


def launch_on_ec2(
    s3_uri: str,
    bucket: str,
    prefix: str,
    n: int,
    seed: int,
    columns: list[str] | None,
    output: str,
    instance_type: str,
    on_demand: bool,
    dry_run: bool,
    prefetch: bool = False,
):
    region = get_bucket_region(bucket)
    print(f"Bucket region: {region}")

    worker_flags = f"-n {n} --seed {seed} --output {output} --local"
    if columns:
        worker_flags += " --columns " + " ".join(columns)
    if prefetch:
        worker_flags += " --prefetch"

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_dir = Path("runs") / f"{timestamp}_generate-queries"
    run_dir.mkdir(parents=True, exist_ok=True)

    job_yaml = {
        "name": "vf-generate-queries",
        "resources": {
            "cloud": "aws",
            "region": region,
            "instance_type": instance_type,
            "use_spot": not on_demand,
        },
        "file_mounts": {"/app": "."},
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync",
        "run": f"cd /app && uv run vf generate-queries {s3_uri} {worker_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    print("=" * 60)
    print("vectorforge generate-queries plan")
    print("=" * 60)
    print(f"  S3 prefix:   {s3_uri}")
    print(f"  Queries:     {n}  (seed={seed})")
    print(f"  Region:      {region}")
    print(f"  Instance:    {instance_type}  ({'on-demand' if on_demand else 'spot'})")
    print(f"  Columns:     {columns or 'all'}")
    print(f"  Fetch mode:  {'prefetch (download-first)' if prefetch else 'range requests'}")
    print(f"  Output:      s3://{bucket}/{prefix}/{output}")
    print(f"  Run dir:     {run_dir}")
    print("=" * 60)

    if dry_run:
        print(f"\n[dry run] Job config: {job_path}")
        print(f"To run manually: sky jobs launch -y {job_path}")
        return

    env_flags = []
    for var in ENV_VARS_TO_FORWARD:
        val = os.environ.get(var)
        if val:
            env_flags.extend(["--env", f"{var}={val}"])

    subprocess.run(["sky", "jobs", "launch", "-y", str(job_path), *env_flags], check=True)

    print(f"\nOutput will be at s3://{bucket}/{prefix}/{output}")
    print("Monitor: sky jobs logs")
    print("Cancel:  sky jobs cancel -a")

def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Sample N eval query rows from embedded S3 parquets"
    )
    parser.add_argument("s3_uri", help="s3://bucket/prefix")
    parser.add_argument("-n", "--num-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--columns", nargs="+", default=None,
                        help="Columns to fetch (default: all). "
                             "E.g. --columns dense_embedding sparse_embedding")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: queries_<n>.parquet)")
    # local mode
    parser.add_argument("--local", action="store_true",
                        help="Run the full pipeline in-process instead of launching EC2")
    parser.add_argument("--prefetch", action="store_true",
                        help="Download each parquet fully before reading (better for large row groups)")
    # EC2 / SkyPilot options
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use on-demand instead of spot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and write job config, don't launch")
    args = parser.parse_args(argv)

    if not args.s3_uri.startswith("s3://"):
        parser.error("s3_uri must start with s3://")
    without_scheme = args.s3_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    output = args.output or f"queries_{args.num_queries}.parquet"

    if args.local:
        run_pipeline(
            bucket=bucket,
            prefix=prefix,
            n=args.num_queries,
            seed=args.seed,
            columns=args.columns,
            output=output,
            prefetch=args.prefetch,
        )
    else:
        launch_on_ec2(
            s3_uri=args.s3_uri,
            bucket=bucket,
            prefix=prefix,
            n=args.num_queries,
            seed=args.seed,
            columns=args.columns,
            output=output,
            instance_type=args.instance_type,
            on_demand=args.on_demand,
            dry_run=args.dry_run,
            prefetch=args.prefetch,
        )


if __name__ == "__main__":
    main()