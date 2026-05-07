#!/usr/bin/env python3
"""
Brute-force nearest-neighbor search for recall evaluation.

Single-worker mode (--local): one GPU instance exhaustively searches the
full corpus. Use for small corpora or testing.

Distributed mode: use `vf brute-force-dist` to split the corpus across N
GPU workers. Each worker runs with --num-jobs N and outputs a partial result
file. Merge with `vf brute-force-merge` when all workers finish.

IDs are md5(source_file:source_row) as UUIDs — the Qdrant loader must use
the same scheme so brute-force hit IDs and Qdrant point IDs can be compared.

Output: s3://bucket/prefix/eval/brute_force_<queries_stem>_k<K>.parquet
  query_id   (str)         UUID derived from md5(source_file:source_row)
  hit_ids    (list[str])   top-K hit IDs ranked best → worst
  hit_scores (list[float]) corresponding similarity scores

Sanity check: top hit for each query should be itself (score = 1.0).

Usage:
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet -k 10000 --local
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet --metric euclidean
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet --dry-run
"""

import argparse
import logging
import math
import os
import subprocess
import tempfile

from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import yaml

from tqdm import tqdm

from vectorforge.utils import make_point_id, s3_rel_key

logger = logging.getLogger(__name__)

ENV_VARS_TO_FORWARD = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
]

DEFAULT_INSTANCE_TYPE = "g4dn.2xlarge"  # 1× T4 GPU, 32GB RAM, 25Gbps
DEFAULT_ACCELERATOR = "T4:1"
DEFAULT_K = 1000
PREFETCH_QUEUE_SIZE = 4
MAX_ROWS_PER_FILE = 10_000_000  # upper bound for global ID encoding

CUDA_IMAGE_IDS = {
    "us-east-1": "ami-0038d79e7270bb987",
    "us-west-2": "ami-08a03808395c1b31f",
    "us-east-2": "ami-0a28b3d7e7c9192a7",
}


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


def partial_prefix(prefix: str, queries_stem: str, k: int) -> str:
    return f"{prefix}/eval/_bf_partial_{queries_stem}_k{k}"


def list_corpus_parquets(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet") and "/eval/" not in key:
                keys.append(key)
    return sorted(keys)


def load_queries(
    bucket: str,
    prefix: str,
    queries_filename: str,
    dense_column: str,
) -> tuple[np.ndarray, list[str]]:
    fs = pafs.S3FileSystem()
    table = pq.read_table(
        f"{bucket}/{prefix}/eval/{queries_filename}",
        filesystem=fs,
        columns=[dense_column, "__source_file__", "__source_row__"],
    )
    embeddings = np.array(table[dense_column].to_pylist(), dtype=np.float32)
    ids = [
        make_point_id(row["__source_file__"], row["__source_row__"])
        for row in table.to_pylist()
    ]
    return embeddings, ids


def _compute_scores(Q, C, metric: DistanceMetric):
    import torch
    import torch.nn.functional as F
    if metric == DistanceMetric.COSINE:
        return F.normalize(Q, dim=1) @ F.normalize(C, dim=1).T
    elif metric == DistanceMetric.DOT:
        return Q @ C.T
    elif metric == DistanceMetric.EUCLIDEAN:
        return -torch.cdist(Q.float(), C.float())


def _prefetch_files(
    bucket: str,
    keys: list[str],
    tmpdir: str,
    dense_column: str,
) -> dict[str, str]:
    """
    Download only the dense embedding column for each key to local disk.
    """
    local_paths = {}
    print(f"Prefetching {len(keys)} files (dense column only)...")
    fs = pafs.S3FileSystem()
    for key in tqdm(keys, unit="file", desc="download", dynamic_ncols=True):
        local_path = os.path.join(tmpdir, key.replace("/", "__") + ".parquet")
        table = pq.read_table(f"{bucket}/{key}", filesystem=fs, columns=[dense_column])
        pq.write_table(table, local_path, compression="snappy")
        local_paths[key] = local_path
    return local_paths


def _save_and_push(
    result: pa.Table,
    local_output: str,
    bucket: str,
    s3_key: str,
):
    pq.write_table(result, local_output, compression="snappy")
    print(f"Wrote {local_output}")
    boto3.client("s3").upload_file(local_output, bucket, s3_key)
    print(f"Pushed to s3://{bucket}/{s3_key}")


def run_pipeline(
    bucket: str,
    prefix: str,
    queries_filename: str,
    k: int,
    metric: DistanceMetric,
    dense_column: str,
    output: str,
    num_jobs: int | None = None,
    job_rank: int | None = None,
):
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is required. Install with: uv sync --extra eval")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU detected — falling back to CPU. This will be slow.")

    queries_stem = Path(queries_filename).stem

    if num_jobs is not None:
        if job_rank is None:
            job_rank = int(os.environ.get("SKYPILOT_JOB_RANK"))
            if job_rank is None:
                raise ValueError("job_rank must be provided when num_jobs is set")
        if job_rank < 0 or job_rank >= num_jobs:
            raise ValueError(f"job_rank must be in [0, {num_jobs - 1}]")

    print(f"Device: {device}  Metric: {metric.value}  K: {k}" +
          (f"  Rank: {job_rank}/{num_jobs}" if num_jobs else ""))

    print(f"Loading queries from {queries_filename}...")
    query_embeddings, query_ids = load_queries(bucket, prefix, queries_filename, dense_column)
    n_queries = len(query_embeddings)
    print(f"{n_queries} queries, dim={query_embeddings.shape[1]}")

    all_corpus_keys = list_corpus_parquets(bucket, prefix)

    if num_jobs is not None:
        chunk = math.ceil(len(all_corpus_keys) / num_jobs)
        start = job_rank * chunk
        corpus_keys = all_corpus_keys[start : start + chunk]
        print(f"Rank {job_rank}/{num_jobs}: {len(corpus_keys)} files (index {start}–{start + len(corpus_keys) - 1} of {len(all_corpus_keys)})")
    else:
        corpus_keys = all_corpus_keys
        print(f"{len(corpus_keys)} corpus files")

    Q = torch.tensor(query_embeddings, dtype=torch.float32, device=device)
    top_scores = torch.full((n_queries, k), float("-inf"), device=device)
    top_encoded_ids = torch.zeros((n_queries, k), dtype=torch.int64, device=device)

    # In distributed mode, prefetch files to local NVMe first so the GPU is
    # never blocked on S3 during the compute loop.
    local_paths: dict[str, str] = {}
    tmpdir_ctx = None
    if num_jobs is not None:
        tmpdir_ctx = tempfile.TemporaryDirectory(prefix="vf_bf_")
        local_paths = _prefetch_files(bucket, corpus_keys, tmpdir_ctx.name, dense_column)

    # Key → stable integer index (relative to this worker's slice, offset by start)
    offset = (job_rank * math.ceil(len(all_corpus_keys) / num_jobs)) if num_jobs else 0
    key_to_file_idx = {key: offset + i for i, key in enumerate(corpus_keys)}

    file_queue: Queue = Queue(maxsize=PREFETCH_QUEUE_SIZE)

    def reader():
        for key in corpus_keys:
            if key in local_paths:
                table = pq.read_table(local_paths[key], columns=[dense_column])
            else:
                fs = pafs.S3FileSystem()
                table = pq.read_table(f"{bucket}/{key}", filesystem=fs, columns=[dense_column])
            arr = np.array(table[dense_column].to_pylist(), dtype=np.float32)
            file_queue.put((key, arr))
        file_queue.put(None)

    Thread(target=reader, daemon=True).start()

    with tqdm(total=len(corpus_keys), unit="file", dynamic_ncols=True) as bar:
        while True:
            item = file_queue.get()
            if item is None:
                break
            key, arr = item
            file_idx = key_to_file_idx[key]
            n_rows = len(arr)

            C = torch.tensor(arr, dtype=torch.float32, device=device)
            scores = _compute_scores(Q, C, metric)  # (n_queries, n_rows)

            file_k = min(k, n_rows)
            file_top_scores, file_top_local_idx = torch.topk(scores, k=file_k, dim=1)

            row_offsets = torch.arange(n_rows, dtype=torch.int64, device=device)
            file_encoded = file_idx * MAX_ROWS_PER_FILE + row_offsets
            file_top_encoded = file_encoded[file_top_local_idx]

            merged_scores = torch.cat([top_scores, file_top_scores], dim=1)
            merged_encoded = torch.cat([top_encoded_ids, file_top_encoded], dim=1)
            top_scores, top_idx = torch.topk(merged_scores, k=k, dim=1)
            top_encoded_ids = merged_encoded.gather(1, top_idx)

            bar.update(1)
            bar.set_postfix_str(s3_rel_key(key, bucket, prefix), refresh=False)

    if tmpdir_ctx:
        tmpdir_ctx.cleanup()

    print("Decoding results...")
    top_encoded_np = top_encoded_ids.cpu().numpy()
    top_scores_np = top_scores.cpu().numpy()
    valid = top_scores_np > float("-inf")

    hit_ids_out, hit_scores_out = [], []
    for q in range(n_queries):
        q_enc = top_encoded_np[q][valid[q]]
        q_scores = top_scores_np[q][valid[q]]
        ids = []
        for enc in q_enc:
            f_idx = int(enc) // MAX_ROWS_PER_FILE
            r_idx = int(enc) % MAX_ROWS_PER_FILE
            ids.append(make_point_id(s3_rel_key(all_corpus_keys[f_idx], bucket, prefix), r_idx))
        hit_ids_out.append(ids)
        hit_scores_out.append(q_scores.tolist())

    result = pa.table({
        "query_id": pa.array(query_ids, type=pa.string()),
        "hit_ids": pa.array(hit_ids_out, type=pa.list_(pa.string())),
        "hit_scores": pa.array(hit_scores_out, type=pa.list_(pa.float32())),
    })

    if num_jobs is not None:
        rank_width = max(3, len(str(num_jobs - 1)))
        s3_key = f"{partial_prefix(prefix, queries_stem, k)}/rank{job_rank:0{rank_width}d}.parquet"
        local_out = f"/tmp/rank{job_rank:0{rank_width}d}.parquet"
    else:
        s3_key = f"{prefix}/eval/{output}"
        local_out = f"/tmp/{output}"

    _save_and_push(result, local_out, bucket, s3_key)


def merge_results(
    bucket: str,
    prefix: str,
    queries_filename: str,
    k: int,
    output: str,
):
    queries_stem = Path(queries_filename).stem
    pprefix = partial_prefix(prefix, queries_stem, k)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    partial_keys = sorted(
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=pprefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    )

    if not partial_keys:
        raise RuntimeError(f"No partial results found at s3://{bucket}/{pprefix}/")

    print(f"Merging {len(partial_keys)} partial results (k={k})...")
    fs = pafs.S3FileSystem()

    # {query_id: [(score, hit_id), ...]} accumulated across all workers
    candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for key in tqdm(partial_keys, unit="file", desc="load", dynamic_ncols=True):
        table = pq.read_table(f"{bucket}/{key}", filesystem=fs)
        for row in table.to_pylist():
            q_id = row["query_id"]
            for hit_id, score in zip(row["hit_ids"], row["hit_scores"]):
                candidates[q_id].append((score, hit_id))

    print(f"Sorting {len(candidates)} queries × up to {len(partial_keys) * k:,} candidates...")
    query_ids = sorted(candidates)
    hit_ids_out, hit_scores_out = [], []
    for q_id in query_ids:
        top = sorted(candidates[q_id], reverse=True)[:k]
        hit_ids_out.append([h for _, h in top])
        hit_scores_out.append([s for s, _ in top])

    result = pa.table({
        "query_id": pa.array(query_ids, type=pa.string()),
        "hit_ids": pa.array(hit_ids_out, type=pa.list_(pa.string())),
        "hit_scores": pa.array(hit_scores_out, type=pa.list_(pa.float32())),
    })

    local_out = f"/tmp/{output}"
    _save_and_push(result, local_out, bucket, f"{prefix}/eval/{output}")


def get_bucket_region(bucket: str) -> str:
    resp = boto3.client("s3").get_bucket_location(Bucket=bucket)
    return resp["LocationConstraint"] or "us-east-1"


def launch_on_ec2(
    s3_uri: str,
    bucket: str,
    prefix: str,
    queries_filename: str,
    k: int,
    metric: DistanceMetric,
    dense_column: str,
    output: str,
    instance_type: str,
    on_demand: bool,
    dry_run: bool,
):
    region = get_bucket_region(bucket)
    print(f"Bucket region: {region}")

    worker_flags = (
        f"--queries {queries_filename} -k {k} "
        f"--metric {metric.value} --dense-column {dense_column} "
        f"--output {output} --local"
    )

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_dir = Path("runs") / f"{timestamp}_brute-force"
    run_dir.mkdir(parents=True, exist_ok=True)

    image_id = CUDA_IMAGE_IDS.get(region)
    if image_id is None:
        print(f"Warning: no CUDA AMI configured for {region!r}. Known: {list(CUDA_IMAGE_IDS)}")

    resources = {
        "cloud": "aws",
        "region": region,
        "instance_type": instance_type,
        "accelerators": DEFAULT_ACCELERATOR,
        "use_spot": not on_demand,
    }
    if image_id:
        resources["image_id"] = image_id

    job_yaml = {
        "name": "vf-brute-force",
        "resources": resources,
        "file_mounts": {"/app": "."},
        "setup": "curl -LsSf https://astral.sh/uv/install.sh | sh && cd /app && uv sync --extra eval",
        "run": f"cd /app && uv run vf brute-force {s3_uri} {worker_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    print("=" * 60)
    print("vectorforge brute-force plan")
    print("=" * 60)
    print(f"  S3 prefix:   {s3_uri}")
    print(f"  Queries:     {queries_filename}")
    print(f"  K:           {k}")
    print(f"  Metric:      {metric.value}")
    print(f"  Instance:    {instance_type}  ({'on-demand' if on_demand else 'spot'})")
    print(f"  Output:      s3://{bucket}/{prefix}/eval/{output}")
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
        description="Brute-force nearest-neighbor search for recall evaluation"
    )
    parser.add_argument("s3_uri", help="s3://bucket/prefix (embedded corpus)")
    parser.add_argument("--queries", default="queries_1000.parquet",
                        help="Queries parquet filename within the prefix (default: queries_1000.parquet)")
    parser.add_argument("-k", type=int, default=DEFAULT_K,
                        help=f"Neighbors to retrieve per query (default: {DEFAULT_K})")
    parser.add_argument("--metric", type=DistanceMetric, default=DistanceMetric.COSINE,
                        choices=list(DistanceMetric),
                        help="Distance metric (default: cosine)")
    parser.add_argument("--dense-column", default="dense_embedding",
                        help="Dense embedding column name (default: dense_embedding)")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: brute_force_<queries_stem>_k<K>.parquet)")
    parser.add_argument("--local", action="store_true",
                        help="Run in-process instead of launching EC2")
    parser.add_argument("--num-jobs", type=int, default=None,
                        help="Total parallel workers (used by brute-force-dist)")
    parser.add_argument("--job-rank", type=int, default=None,
                        help="This worker's rank (0-indexed; defaults to $SKYPILOT_JOB_RANK)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge partial results from a previous distributed run")
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

    queries_stem = Path(args.queries).stem
    output = args.output or f"brute_force_{queries_stem}_k{args.k}.parquet"

    if args.merge:
        merge_results(
            bucket=bucket,
            prefix=prefix,
            queries_filename=args.queries,
            k=args.k,
            output=output,
        )
    elif args.local or args.num_jobs:
        run_pipeline(
            bucket=bucket,
            prefix=prefix,
            queries_filename=args.queries,
            k=args.k,
            metric=args.metric,
            dense_column=args.dense_column,
            output=output,
            num_jobs=args.num_jobs,
            job_rank=args.job_rank,
        )
    else:
        launch_on_ec2(
            s3_uri=args.s3_uri,
            bucket=bucket,
            prefix=prefix,
            queries_filename=args.queries,
            k=args.k,
            metric=args.metric,
            dense_column=args.dense_column,
            output=output,
            instance_type=args.instance_type,
            on_demand=args.on_demand,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
