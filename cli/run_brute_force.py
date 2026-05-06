#!/usr/bin/env python3
"""
Brute-force nearest-neighbor search for recall evaluation.

Exhaustively computes similarity between every query vector and every row in
the embedded corpus (S3 parquets), returning the true top-K nearest neighbors
per query. Use the results to measure recall against a Qdrant index.

Default mode launches a single GPU EC2 instance in-region via SkyPilot.
Use --local to run in-process (also what the EC2 job calls).

IDs are md5(source_file:source_row) — the loader must use the same scheme so
that brute-force hit IDs and Qdrant point IDs can be intersected for recall.

The top hit for each query should always be itself (cosine similarity = 1.0),
which is a useful sanity check that IDs are consistent end-to-end.

Output: s3://bucket/prefix/brute_force_<queries_stem>_k<K>.parquet
  query_id   (str)         md5 of the query's source_file:source_row
  hit_ids    (list[str])   top-K hit IDs ranked best → worst
  hit_scores (list[float]) corresponding similarity scores

Usage:
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet -k 10000
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet --metric euclidean
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet --local
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet --dry-run
"""

import argparse
import logging
import os
import subprocess
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
PREFETCH_QUEUE_SIZE = 8
MAX_ROWS_PER_FILE = 10_000_000  # upper bound for global ID encoding

# Deep Learning AMIs with CUDA drivers pre-installed.
# Same AMIs as the embed pipeline — work on any GPU instance type.
CUDA_IMAGE_IDS = {
    "us-east-1": "ami-0038d79e7270bb987",
    "us-west-2": "ami-08a03808395c1b31f",
    "us-east-2": "ami-0a28b3d7e7c9192a7",
}


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"




def list_corpus_parquets(bucket: str, prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = os.path.basename(obj["Key"])
            if obj["Key"].endswith(".parquet") and not name.startswith(("queries_", "brute_force_")):
                keys.append(obj["Key"])
    return sorted(keys)


def load_queries(
    bucket: str,
    prefix: str,
    queries_filename: str,
    dense_column: str,
) -> tuple[np.ndarray, list[str]]:
    fs = pafs.S3FileSystem()
    table = pq.read_table(
        f"{bucket}/{prefix}/{queries_filename}",
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
        # negate so that topk (higher = better) gives closest points
        return -torch.cdist(Q.float(), C.float())


def run_pipeline(
    bucket: str,
    prefix: str,
    queries_filename: str,
    k: int,
    metric: DistanceMetric,
    dense_column: str,
    output: str,
):
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is required. Install it with: uv add torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU detected — falling back to CPU. This will be slow.")
    print(f"Device: {device}  Metric: {metric.value}  K: {k}")

    print(f"Loading queries from {queries_filename}...")
    query_embeddings, query_ids = load_queries(bucket, prefix, queries_filename, dense_column)
    n_queries = len(query_embeddings)
    print(f"{n_queries} queries, dim={query_embeddings.shape[1]}")

    corpus_keys = list_corpus_parquets(bucket, prefix)
    print(f"{len(corpus_keys)} corpus files")

    Q = torch.tensor(query_embeddings, dtype=torch.float32, device=device)

    # Running top-K state.
    # Scores live on GPU as a float tensor; IDs are encoded as int64
    # (file_idx * MAX_ROWS_PER_FILE + row_offset) so all merge operations
    # stay on GPU. IDs are decoded to md5 strings once at the end.
    top_scores = torch.full((n_queries, k), float("-inf"), device=device)
    top_encoded_ids = torch.zeros((n_queries, k), dtype=torch.int64, device=device)

    # Assign each corpus key a stable integer index before any I/O starts.
    key_to_file_idx = {key: idx for idx, key in enumerate(corpus_keys)}

    # Background reader: streams files into a bounded queue so the GPU is
    # never starved waiting for a single blocking download.
    file_queue: Queue = Queue(maxsize=PREFETCH_QUEUE_SIZE)

    def s3_reader():
        fs = pafs.S3FileSystem()
        for key in corpus_keys:
            table = pq.read_table(f"{bucket}/{key}", filesystem=fs, columns=[dense_column])
            arr = np.array(table[dense_column].to_pylist(), dtype=np.float32)
            file_queue.put((key, arr))
        file_queue.put(None)  # sentinel

    reader = Thread(target=s3_reader, daemon=True)
    reader.start()

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

            # Top-K from this file
            file_k = min(k, n_rows)
            file_top_scores, file_top_local_idx = torch.topk(scores, k=file_k, dim=1)

            # Encode row positions as global integer IDs
            row_offsets = torch.arange(n_rows, dtype=torch.int64, device=device)
            file_encoded = file_idx * MAX_ROWS_PER_FILE + row_offsets
            file_top_encoded = file_encoded[file_top_local_idx]  # (n_queries, file_k)

            # Merge this file's top-K into the running top-K — pure tensor ops
            merged_scores = torch.cat([top_scores, file_top_scores], dim=1)
            merged_encoded = torch.cat([top_encoded_ids, file_top_encoded], dim=1)
            top_scores, top_idx = torch.topk(merged_scores, k=k, dim=1)
            top_encoded_ids = merged_encoded.gather(1, top_idx)

            bar.update(1)
            bar.set_postfix_str(s3_rel_key(key, bucket, prefix), refresh=False)

    reader.join()

    # Decode integer IDs → md5 strings (done once, after all GPU work is done)
    print("Decoding results...")
    top_encoded_np = top_encoded_ids.cpu().numpy()
    top_scores_np = top_scores.cpu().numpy()
    valid = top_scores_np > float("-inf")

    hit_ids_out = []
    hit_scores_out = []
    for q in range(n_queries):
        q_enc = top_encoded_np[q][valid[q]]
        q_scores = top_scores_np[q][valid[q]]
        ids = []
        for enc in q_enc:
            f_idx = int(enc) // MAX_ROWS_PER_FILE
            r_idx = int(enc) % MAX_ROWS_PER_FILE
            ids.append(make_point_id(s3_rel_key(corpus_keys[f_idx], bucket, prefix), r_idx))
        hit_ids_out.append(ids)
        hit_scores_out.append(q_scores.tolist())

    result = pa.table({
        "query_id": pa.array(query_ids, type=pa.string()),
        "hit_ids": pa.array(hit_ids_out, type=pa.list_(pa.string())),
        "hit_scores": pa.array(hit_scores_out, type=pa.list_(pa.float32())),
    })

    local_output = f"/tmp/{output}"
    pq.write_table(result, local_output, compression="snappy")
    print(f"Wrote {local_output}")

    s3_key = f"{prefix}/{output}"
    boto3.client("s3").upload_file(local_output, bucket, s3_key)
    print(f"Pushed to s3://{bucket}/{s3_key}")


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
        print(f"Warning: no pre-configured CUDA AMI for region {region!r}. "
              f"GPU may not be available. Known regions: {list(CUDA_IMAGE_IDS)}")

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
        description="Brute-force nearest-neighbor search for recall evaluation"
    )
    parser.add_argument("s3_uri", help="s3://bucket/prefix (embedded corpus)")
    parser.add_argument("--queries", required=True,
                        help="Queries parquet filename within the prefix (e.g. queries_1000.parquet)")
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

    if args.local:
        run_pipeline(
            bucket=bucket,
            prefix=prefix,
            queries_filename=args.queries,
            k=args.k,
            metric=args.metric,
            dense_column=args.dense_column,
            output=output,
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