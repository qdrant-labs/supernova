#!/usr/bin/env python3
"""
Distributed brute-force nearest-neighbor search via a SkyPilot GPU pool.

Splits the corpus across N GPU workers. Each worker prefetches its assigned
files to local NVMe, runs GPU similarity search, and saves a partial top-K
result to S3. Run `vf brute-force-merge` when all workers finish.

Usage:
  vf brute-force-dist s3://bucket/prefix --queries queries_1000.parquet
  vf brute-force-dist s3://bucket/prefix --queries queries_1000.parquet --num-jobs 50
  vf brute-force-dist s3://bucket/prefix --queries queries_1000.parquet --dry-run
"""

import argparse
import logging
import math
from pathlib import Path

import yaml

from cli.run_brute_force import (
    CUDA_IMAGE_IDS,
    DEFAULT_ACCELERATOR,
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_K,
    DistanceMetric,
    list_corpus_parquets,
    partial_prefix,
)
from cli.skypilot_utils import build_env_flags, make_run_dir, launch_pool_and_jobs, print_monitor
from vectorforge.utils import get_bucket_region

logger = logging.getLogger(__name__)

DEFAULT_NUM_JOBS = 50


def main(argv: list[str] | None = None):
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Distributed brute-force nearest-neighbor search via SkyPilot GPU pool"
    )
    parser.add_argument("s3_uri", help="s3://bucket/prefix (embedded corpus)")
    parser.add_argument("--queries", default="queries_1000.parquet",
                        help="Queries parquet filename within the prefix (default: queries_1000.parquet)")
    parser.add_argument("-k", type=int, default=DEFAULT_K,
                        help=f"Neighbors per query (default: {DEFAULT_K})")
    parser.add_argument("--metric", type=DistanceMetric, default=DistanceMetric.COSINE,
                        choices=list(DistanceMetric))
    parser.add_argument("--dense-column", default="dense_embedding")
    parser.add_argument("--num-jobs", type=int, default=DEFAULT_NUM_JOBS,
                        help=f"Number of GPU workers (default: {DEFAULT_NUM_JOBS})")
    parser.add_argument("--output", default=None,
                        help="Final merged output filename")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"EC2 instance type per worker (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use on-demand instead of spot")
    parser.add_argument("--pool-name", default=None,
                        help="SkyPilot pool name (default: auto-generated)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and write configs, don't launch")
    args = parser.parse_args(argv)

    if not args.s3_uri.startswith("s3://"):
        parser.error("s3_uri must start with s3://")
    without_scheme = args.s3_uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    queries_stem = Path(args.queries).stem
    output = args.output or f"brute_force_{queries_stem}_k{args.k}.parquet"

    region = get_bucket_region(bucket)
    image_id = CUDA_IMAGE_IDS.get(region)
    if image_id is None:
        print(f"Warning: no CUDA AMI configured for {region!r}. Known: {list(CUDA_IMAGE_IDS)}")

    corpus_keys = list_corpus_parquets(bucket, prefix)
    files_per_worker = math.ceil(len(corpus_keys) / args.num_jobs)

    pool_name = args.pool_name or f"vf-bf-{queries_stem}"
    run_dir = make_run_dir("brute-force-dist")

    resources = {
        "cloud": "aws",
        "region": region,
        "instance_type": args.instance_type,
        "accelerators": DEFAULT_ACCELERATOR,
        "use_spot": not args.on_demand,
    }
    if image_id:
        resources["image_id"] = image_id

    worker_flags = (
        f"--queries {args.queries} -k {args.k} "
        f"--metric {args.metric.value} --dense-column {args.dense_column} "
        f"--num-jobs {args.num_jobs} --local"
    )

    pool_yaml = {
        "pool": {
            "min_workers": args.num_jobs,
            "max_workers": args.num_jobs,
        },
        "resources": resources,
        "file_mounts": {"/app": "."},
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync --extra eval",
    }
    job_yaml = {
        "name": f"vf-bf-{queries_stem}",
        "resources": resources,
        "run": f"cd /app && uv run vf brute-force {args.s3_uri} {worker_flags}",
    }

    pool_path = run_dir / "pool.yaml"
    job_path = run_dir / "job.yaml"
    with open(pool_path, "w") as f:
        yaml.dump(pool_yaml, f, default_flow_style=False, sort_keys=False)
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    pprefix = partial_prefix(prefix, queries_stem, args.k)
    merge_cmd = (
        f"vf brute-force {args.s3_uri} --queries {args.queries} "
        f"-k {args.k} --output {output} --merge"
    )

    print("=" * 60)
    print("vectorforge brute-force-dist plan")
    print("=" * 60)
    print(f"  S3 prefix:      {args.s3_uri}")
    print(f"  Queries:        {args.queries}")
    print(f"  K:              {args.k}")
    print(f"  Metric:         {args.metric.value}")
    print(f"  Workers:        {args.num_jobs}")
    print(f"  Files/worker:   ~{files_per_worker} (of {len(corpus_keys)} total)")
    print(f"  Instance:       {args.instance_type}  ({'on-demand' if args.on_demand else 'spot'})")
    print(f"  Region:         {region}")
    print(f"  Pool:           {pool_name}")
    print(f"  Partial output: s3://{bucket}/{pprefix}/")
    print(f"  Final output:   s3://{bucket}/{prefix}/{output}")
    print(f"  Run dir:        {run_dir}")
    print("=" * 60)
    print("\nWhen all workers finish, run:")
    print(f"  {merge_cmd}")
    print()

    if args.dry_run:
        print(f"[dry run] Pool config: {pool_path}")
        print(f"[dry run] Job config:  {job_path}")
        return

    launch_pool_and_jobs(pool_name, pool_path, job_path, args.num_jobs, build_env_flags())

    print(f"Submitted {args.num_jobs} workers to pool '{pool_name}'")
    print_monitor(pool_name)
    print(f"\nWhen done: {merge_cmd}")


if __name__ == "__main__":
    main()
