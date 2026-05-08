#!/usr/bin/env python3
"""
Sample N rows from an embedded corpus as eval query vectors.

Default mode launches a single high-bandwidth EC2 instance in the corpus's
region via SkyPilot (S3 only — HF doesn't have an obvious in-region story).
The job runs the full pipeline: build index from parquet footers, fetch the
actual row data, push queries.parquet to the corpus's eval/ subfolder.

Use --local to run the full pipeline in-process (also what the EC2 job calls).

Output: {corpus_uri}/eval/queries_<n>.parquet
  All original parquet columns (dense_embedding, sparse_embedding, text, …)
  plus __source_file__ (bare key) and __source_row__ for provenance.

Usage:
  vf generate-queries s3://bucket/prefix -n 1000
  vf generate-queries hf://datasets/ns/repo -n 1000 --local
  vf generate-queries s3://bucket/prefix -n 1000 --on-demand
  vf generate-queries s3://bucket/prefix -n 1000 --dry-run
"""

import argparse
import bisect
import logging
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cli.skypilot_utils import build_env_flags, make_run_dir, launch_single_job
from vectorforge.destinations import (
    S3Destination,
    bare_key_for_uri,
    discover_corpus_parquets,
    filesystem_for_uri,
    fs_path_for_uri,
    parse_destination,
    upload_bytes_to_uri,
)
from vectorforge.utils import get_bucket_region

logger = logging.getLogger(__name__)

# "n"-family = enhanced networking (25-100Gbps). S3 reads that take
# minutes locally finish in seconds in-region.
DEFAULT_INSTANCE_TYPE = "r5n.2xlarge"  # 8 vCPU, 64GB RAM, 25Gbps


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


def run_pipeline(
    corpus_uri: str,
    n: int,
    seed: int,
    columns: list[str] | None,
    output: str,
    prefetch: bool = False,
):
    dest = parse_destination(corpus_uri)
    print(f"Listing parquets at {dest.root_uri}/...")
    uris = discover_corpus_parquets(dest)
    if not uris:
        print("No parquet files found.")
        return
    print(f"Found {len(uris)} parquet files")

    print("Reading metadata (parquet footers)...")
    manifest, total_rows = build_manifest(uris)
    print(f"Total rows: {total_rows:,}")

    actual_n = min(n, total_rows)
    if actual_n < n:
        print(f"Warning: only {total_rows} rows available, capping at {actual_n}.")

    print(f"Sampling {actual_n} query pointers (seed={seed})...")
    file_map = sample_offsets(manifest, total_rows, actual_n, seed)
    total_files = len(file_map)

    mode = "prefetch (download-first)" if prefetch else "range requests"
    print(f"Fetching rows from {total_files} files ({mode})...")
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
    print(f"Collected {len(result)} rows")

    pq.write_table(result, output, compression="snappy")
    print(f"Wrote {output}")

    buf = pa.BufferOutputStream()
    pq.write_table(result, buf, compression="snappy")
    eval_uri = dest.eval_uri(output)
    upload_bytes_to_uri(bytes(buf.getvalue()), eval_uri)
    print(f"Pushed to {eval_uri}")


def launch_on_ec2(
    corpus_uri: str,
    n: int,
    seed: int,
    columns: list[str] | None,
    output: str,
    instance_type: str,
    on_demand: bool,
    dry_run: bool,
    prefetch: bool = False,
):
    dest = parse_destination(corpus_uri)
    if not isinstance(dest, S3Destination):
        # In-region EC2 launch only makes sense for S3 (we co-locate the
        # instance with the bucket). For HF corpora, run --local from a
        # high-bandwidth machine instead.
        raise SystemExit(
            f"EC2 launch is supported for s3:// corpora only. For {corpus_uri}, "
            "use --local to run in-process."
        )
    region = get_bucket_region(dest.bucket)
    print(f"Bucket region: {region}")

    worker_flags = f"-n {n} --seed {seed} --output {output} --local"
    if columns:
        worker_flags += " --columns " + " ".join(columns)
    if prefetch:
        worker_flags += " --prefetch"

    run_dir = make_run_dir("generate-queries")

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
        "run": f"cd /app && uv run vf generate-queries {corpus_uri} {worker_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    print("=" * 60)
    print("vectorforge generate-queries plan")
    print("=" * 60)
    print(f"  Corpus URI:  {corpus_uri}")
    print(f"  Queries:     {n}  (seed={seed})")
    print(f"  Region:      {region}")
    print(f"  Instance:    {instance_type}  ({'on-demand' if on_demand else 'spot'})")
    print(f"  Columns:     {columns or 'all'}")
    print(f"  Fetch mode:  {'prefetch (download-first)' if prefetch else 'range requests'}")
    print(f"  Output:      {dest.eval_uri(output)}")
    print(f"  Run dir:     {run_dir}")
    print("=" * 60)

    if dry_run:
        print(f"\n[dry run] Job config: {job_path}")
        print(f"To run manually: sky jobs launch -y {job_path}")
        return

    launch_single_job(job_path, build_env_flags())

    print(f"\nOutput will be at {dest.eval_uri(output)}")
    print("Monitor: sky jobs logs")
    print("Cancel:  sky jobs cancel -a")


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Sample N eval query rows from an embedded corpus (S3 or HF)"
    )
    parser.add_argument("corpus_uri", help="s3://bucket/prefix or hf://datasets/ns/repo")
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
    # EC2 / SkyPilot options (S3 only)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use on-demand instead of spot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and write job config, don't launch")
    args = parser.parse_args(argv)

    try:
        parse_destination(args.corpus_uri)
    except ValueError as e:
        parser.error(str(e))

    output = args.output or f"queries_{args.num_queries}.parquet"

    if args.local:
        run_pipeline(
            corpus_uri=args.corpus_uri,
            n=args.num_queries,
            seed=args.seed,
            columns=args.columns,
            output=output,
            prefetch=args.prefetch,
        )
    else:
        launch_on_ec2(
            corpus_uri=args.corpus_uri,
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
