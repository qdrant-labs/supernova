#!/usr/bin/env python3
"""
Brute-force nearest-neighbor search for recall evaluation.

Single-worker mode (--local): one GPU instance exhaustively searches the
full corpus. Use for small corpora or testing.

Distributed mode: use `vf brute-force-dist` to split the corpus across N
GPU workers. Each worker runs with --num-jobs N and outputs a partial result
file. Merge with `vf brute-force-merge` when all workers finish.

IDs are md5(bare_key:source_row) as UUIDs — both the loader's vf_point_id
macro and this script use bare_key_for_uri to derive the same key from the
backend-specific URI form, so brute-force hit IDs and Qdrant point IDs match.

Output: {corpus_uri}/eval/brute_force_<queries_stem>_k<K>.parquet
  query_id   (str)         UUID derived from md5(bare_key:source_row)
  hit_ids    (list[str])   top-K hit IDs ranked best → worst
  hit_scores (list[float]) corresponding similarity scores

Sanity check: top hit for each query should be itself (score = 1.0).

Usage:
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet
  vf brute-force hf://datasets/ns/repo --queries queries_1000.parquet --local
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet --metric euclidean
  vf brute-force s3://bucket/prefix --queries queries_1000.parquet --dry-run
"""

import argparse
import logging
import math
import os
import tempfile

from collections import defaultdict
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tqdm import tqdm

from cli.skypilot_utils import CUDA_IMAGE_IDS, build_env_flags, make_run_dir, launch_single_job
from vectorforge.destinations import (
    S3Destination,
    bare_key_for_uri,
    discover_corpus_parquets,
    filesystem_for_uri,
    fs_path_for_uri,
    list_parquets_under,
    parse_destination,
    upload_file_to_uri,
)
from vectorforge.utils import get_bucket_region, make_point_id

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE_TYPE = "g4dn.2xlarge"  # 1× T4 GPU, 32GB RAM, 25Gbps
DEFAULT_ACCELERATOR = "T4:1"
DEFAULT_K = 1000
PREFETCH_QUEUE_SIZE = 4
MAX_ROWS_PER_FILE = 10_000_000  # upper bound for global ID encoding


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


def partial_subkey(queries_stem: str, k: int) -> str:
    """Sub-path under {corpus_uri}/eval/ for per-rank partial result files."""
    return f"_bf_partial_{queries_stem}_k{k}"


def load_queries(
    dest,
    queries_filename: str,
    dense_column: str,
) -> tuple[np.ndarray, list[str]]:
    queries_uri = dest.eval_uri(queries_filename)
    fs = filesystem_for_uri(queries_uri)
    fs_path = fs_path_for_uri(queries_uri)
    table = pq.read_table(
        fs_path,
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
    uris: list[str],
    tmpdir: str,
    dense_column: str,
) -> dict[str, str]:
    """
    Download only the dense embedding column for each file to local disk.
    Keyed by the source URI.
    """
    local_paths = {}
    print(f"Prefetching {len(uris)} files (dense column only)...")
    for uri in tqdm(uris, unit="file", desc="download", dynamic_ncols=True):
        safe_name = uri.replace("/", "__").replace(":", "_")
        local_path = os.path.join(tmpdir, safe_name + ".parquet")
        fs = filesystem_for_uri(uri)
        fs_path = fs_path_for_uri(uri)
        table = pq.read_table(fs_path, filesystem=fs, columns=[dense_column])
        pq.write_table(table, local_path, compression="snappy")
        local_paths[uri] = local_path
    return local_paths


def _save_and_push(result: pa.Table, local_output: str, dest_uri: str):
    pq.write_table(result, local_output, compression="snappy")
    print(f"Wrote {local_output}")
    upload_file_to_uri(local_output, dest_uri)
    print(f"Pushed to {dest_uri}")


def run_pipeline(
    corpus_uri: str,
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

    dest = parse_destination(corpus_uri)
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
    query_embeddings, query_ids = load_queries(dest, queries_filename, dense_column)
    n_queries = len(query_embeddings)
    print(f"{n_queries} queries, dim={query_embeddings.shape[1]}")

    all_corpus_uris = discover_corpus_parquets(dest)

    if num_jobs is not None:
        chunk = math.ceil(len(all_corpus_uris) / num_jobs)
        start = job_rank * chunk
        corpus_uris = all_corpus_uris[start : start + chunk]
        print(f"Rank {job_rank}/{num_jobs}: {len(corpus_uris)} files (index {start}–{start + len(corpus_uris) - 1} of {len(all_corpus_uris)})")
    else:
        corpus_uris = all_corpus_uris
        print(f"{len(corpus_uris)} corpus files")

    Q = torch.tensor(query_embeddings, dtype=torch.float32, device=device)
    top_scores = torch.full((n_queries, k), float("-inf"), device=device)
    top_encoded_ids = torch.zeros((n_queries, k), dtype=torch.int64, device=device)

    # In distributed mode, prefetch files to local NVMe first so the GPU is
    # never blocked on remote I/O during the compute loop.
    local_paths: dict[str, str] = {}
    tmpdir_ctx = None
    if num_jobs is not None:
        tmpdir_ctx = tempfile.TemporaryDirectory(prefix="vf_bf_")
        local_paths = _prefetch_files(corpus_uris, tmpdir_ctx.name, dense_column)

    # URI → stable integer index (relative to this worker's slice, offset by start).
    offset = (job_rank * math.ceil(len(all_corpus_uris) / num_jobs)) if num_jobs else 0
    uri_to_file_idx = {uri: offset + i for i, uri in enumerate(corpus_uris)}

    file_queue: Queue = Queue(maxsize=PREFETCH_QUEUE_SIZE)

    def reader():
        for uri in corpus_uris:
            if uri in local_paths:
                table = pq.read_table(local_paths[uri], columns=[dense_column])
            else:
                fs = filesystem_for_uri(uri)
                fs_path = fs_path_for_uri(uri)
                table = pq.read_table(fs_path, filesystem=fs, columns=[dense_column])
            arr = np.array(table[dense_column].to_pylist(), dtype=np.float32)
            file_queue.put((uri, arr))
        file_queue.put(None)

    Thread(target=reader, daemon=True).start()

    with tqdm(total=len(corpus_uris), unit="file", dynamic_ncols=True) as bar:
        while True:
            item = file_queue.get()
            if item is None:
                break
            uri, arr = item
            file_idx = uri_to_file_idx[uri]
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
            bar.set_postfix_str(bare_key_for_uri(uri), refresh=False)

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
            # bare_key_for_uri so the ID matches what the loader's vf_point_id
            # macro produces and what the queries file's __source_file__ holds.
            ids.append(make_point_id(bare_key_for_uri(all_corpus_uris[f_idx]), r_idx))
        hit_ids_out.append(ids)
        hit_scores_out.append(q_scores.tolist())

    result = pa.table({
        "query_id": pa.array(query_ids, type=pa.string()),
        "hit_ids": pa.array(hit_ids_out, type=pa.list_(pa.string())),
        "hit_scores": pa.array(hit_scores_out, type=pa.list_(pa.float32())),
    })

    if num_jobs is not None:
        rank_width = max(3, len(str(num_jobs - 1)))
        partial_path = f"{partial_subkey(queries_stem, k)}/rank{job_rank:0{rank_width}d}.parquet"
        dest_uri = dest.eval_uri(partial_path)
        local_out = f"/tmp/rank{job_rank:0{rank_width}d}.parquet"
    else:
        dest_uri = dest.eval_uri(output)
        local_out = f"/tmp/{output}"

    _save_and_push(result, local_out, dest_uri)


def merge_results(
    corpus_uri: str,
    queries_filename: str,
    k: int,
    output: str,
):
    dest = parse_destination(corpus_uri)
    queries_stem = Path(queries_filename).stem
    pprefix_uri = dest.eval_uri(partial_subkey(queries_stem, k))

    partial_uris = list_parquets_under(pprefix_uri)
    if not partial_uris:
        raise RuntimeError(f"No partial results found at {pprefix_uri}/")

    print(f"Merging {len(partial_uris)} partial results (k={k})...")

    # {query_id: [(score, hit_id), ...]} accumulated across all workers.
    candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for uri in tqdm(partial_uris, unit="file", desc="load", dynamic_ncols=True):
        fs = filesystem_for_uri(uri)
        fs_path = fs_path_for_uri(uri)
        table = pq.read_table(fs_path, filesystem=fs)
        for row in table.to_pylist():
            q_id = row["query_id"]
            for hit_id, score in zip(row["hit_ids"], row["hit_scores"]):
                candidates[q_id].append((score, hit_id))

    print(f"Sorting {len(candidates)} queries × up to {len(partial_uris) * k:,} candidates...")
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
    _save_and_push(result, local_out, dest.eval_uri(output))


def launch_on_ec2(
    corpus_uri: str,
    queries_filename: str,
    k: int,
    metric: DistanceMetric,
    dense_column: str,
    output: str,
    instance_type: str,
    on_demand: bool,
    dry_run: bool,
):
    dest = parse_destination(corpus_uri)
    if not isinstance(dest, S3Destination):
        raise SystemExit(
            f"EC2 launch is supported for s3:// corpora only. For {corpus_uri}, "
            "use --local."
        )
    region = get_bucket_region(dest.bucket)
    print(f"Bucket region: {region}")

    worker_flags = (
        f"--queries {queries_filename} -k {k} "
        f"--metric {metric.value} --dense-column {dense_column} "
        f"--output {output} --local"
    )

    run_dir = make_run_dir("brute-force")

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
        "run": f"cd /app && uv run vf brute-force {corpus_uri} {worker_flags}",
    }
    job_path = run_dir / "job.yaml"
    with open(job_path, "w") as f:
        yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)

    print("=" * 60)
    print("vectorforge brute-force plan")
    print("=" * 60)
    print(f"  Corpus URI:  {corpus_uri}")
    print(f"  Queries:     {queries_filename}")
    print(f"  K:           {k}")
    print(f"  Metric:      {metric.value}")
    print(f"  Instance:    {instance_type}  ({'on-demand' if on_demand else 'spot'})")
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
        description="Brute-force nearest-neighbor search for recall evaluation"
    )
    parser.add_argument("corpus_uri", help="s3://bucket/prefix or hf://datasets/ns/repo")
    parser.add_argument("--queries", default="queries_1000.parquet",
                        help="Queries parquet filename within {corpus}/eval/ (default: queries_1000.parquet)")
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

    queries_stem = Path(args.queries).stem
    output = args.output or f"brute_force_{queries_stem}_k{args.k}.parquet"

    if args.local or args.num_jobs:
        run_pipeline(
            corpus_uri=args.corpus_uri,
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
            corpus_uri=args.corpus_uri,
            queries_filename=args.queries,
            k=args.k,
            metric=args.metric,
            dense_column=args.dense_column,
            output=output,
            instance_type=args.instance_type,
            on_demand=args.on_demand,
            dry_run=args.dry_run,
        )


def merge_main(argv: list[str] | None = None):
    """Entry point for `vf brute-force-merge`. Standalone parser; no argv injection."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Merge partial brute-force results from a distributed run"
    )
    parser.add_argument("corpus_uri", help="s3://bucket/prefix or hf://datasets/ns/repo")
    parser.add_argument("--queries", default="queries_1000.parquet",
                        help="Queries parquet filename within {corpus}/eval/ (default: queries_1000.parquet)")
    parser.add_argument("-k", type=int, default=DEFAULT_K,
                        help=f"Neighbors retrieved per query (default: {DEFAULT_K})")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: brute_force_<queries_stem>_k<K>.parquet)")
    args = parser.parse_args(argv)

    try:
        parse_destination(args.corpus_uri)
    except ValueError as e:
        parser.error(str(e))

    queries_stem = Path(args.queries).stem
    output = args.output or f"brute_force_{queries_stem}_k{args.k}.parquet"

    merge_results(
        corpus_uri=args.corpus_uri,
        queries_filename=args.queries,
        k=args.k,
        output=output,
    )


if __name__ == "__main__":
    main()
