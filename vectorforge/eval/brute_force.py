"""
Brute-force exact-nearest-neighbor search over an embedded corpus.

This is the programmatic core of `vf brute-force`. The CLI in
``cli/run_brute_force.py`` only wraps argparse + the SkyPilot/EC2 launcher
around these functions.

IDs are md5(bare_key:source_row) as UUIDs — both the loader's vf_point_id
DuckDB macro and ``run_brute_force`` here use ``bare_key_for_uri`` to derive
the same key from the backend-specific URI form, so brute-force hit IDs and
Qdrant point IDs match.

Output: ``{corpus_uri}/eval/brute_force_<queries_stem>_k<K>.parquet``
  query_id   (str)         UUID derived from md5(bare_key:source_row)
  hit_ids    (list[str])   top-K hit IDs ranked best → worst
  hit_scores (list[float]) corresponding similarity scores

Sanity check: top hit for each query should be itself (score = 1.0).
"""

from __future__ import annotations

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

from tqdm import tqdm

from vectorforge.destinations import (
    bare_key_for_uri,
    discover_corpus_parquets,
    filesystem_for_uri,
    fs_path_for_uri,
    list_parquets_under,
    parse_destination,
    upload_file_to_uri,
)
from vectorforge.utils import make_point_id

logger = logging.getLogger(__name__)

DEFAULT_K = 1000
PREFETCH_QUEUE_SIZE = 4
# Per-file ID encoding: file index * MAX_ROWS_PER_FILE + row index. As long
# as no single file has more rows than this, the encoding is collision-free
# and reversible.
MAX_ROWS_PER_FILE = 10_000_000


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
    logger.info("Prefetching %d files (dense column only)...", len(uris))
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
    logger.info("Wrote %s", local_output)
    upload_file_to_uri(local_output, dest_uri)
    logger.info("Pushed to %s", dest_uri)


def run_brute_force(
    corpus_uri: str,
    queries_filename: str,
    k: int,
    metric: DistanceMetric,
    dense_column: str,
    output: str,
    num_jobs: int | None = None,
    job_rank: int | None = None,
):
    """
    Run exact-nearest-neighbor search over the corpus at ``corpus_uri``.

    Single-worker mode: pass ``num_jobs=None``. The full corpus is searched
    and the result is written to ``{corpus_uri}/eval/{output}``.

    Distributed mode: pass ``num_jobs`` and ``job_rank`` (0-indexed). Each
    worker takes a contiguous slice of the corpus parquet list and writes a
    partial result under
    ``{corpus_uri}/eval/_bf_partial_<queries_stem>_k<K>/rank{rank}.parquet``.
    Call ``merge_results`` once all ranks finish.
    """
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
            env_rank = os.environ.get("SKYPILOT_JOB_RANK")
            if env_rank is None:
                raise ValueError(
                    "job_rank must be provided (or SKYPILOT_JOB_RANK set) when num_jobs is set"
                )
            job_rank = int(env_rank)
        if job_rank < 0 or job_rank >= num_jobs:
            raise ValueError(f"job_rank must be in [0, {num_jobs - 1}]")

    logger.info(
        "Device: %s  Metric: %s  K: %d%s",
        device, metric.value, k,
        f"  Rank: {job_rank}/{num_jobs}" if num_jobs else "",
    )

    logger.info("Loading queries from %s...", queries_filename)
    query_embeddings, query_ids = load_queries(dest, queries_filename, dense_column)
    n_queries = len(query_embeddings)
    logger.info("%d queries, dim=%d", n_queries, query_embeddings.shape[1])

    all_corpus_uris = discover_corpus_parquets(dest)

    if num_jobs is not None:
        chunk = math.ceil(len(all_corpus_uris) / num_jobs)
        start = job_rank * chunk
        corpus_uris = all_corpus_uris[start : start + chunk]
        logger.info(
            "Rank %d/%d: %d files (index %d–%d of %d)",
            job_rank, num_jobs, len(corpus_uris),
            start, start + len(corpus_uris) - 1, len(all_corpus_uris),
        )
    else:
        corpus_uris = all_corpus_uris
        logger.info("%d corpus files", len(corpus_uris))

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

    logger.info("Decoding results...")
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
    """
    Merge per-rank partial brute-force results into a single top-K parquet.

    Reads ``{corpus_uri}/eval/_bf_partial_<queries_stem>_k<K>/rank*.parquet``
    and writes ``{corpus_uri}/eval/{output}``.
    """
    dest = parse_destination(corpus_uri)
    queries_stem = Path(queries_filename).stem
    pprefix_uri = dest.eval_uri(partial_subkey(queries_stem, k))

    partial_uris = list_parquets_under(pprefix_uri)
    if not partial_uris:
        raise RuntimeError(f"No partial results found at {pprefix_uri}/")

    logger.info("Merging %d partial results (k=%d)...", len(partial_uris), k)

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

    logger.info(
        "Sorting %d queries × up to %d candidates...",
        len(candidates), len(partial_uris) * k,
    )
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
