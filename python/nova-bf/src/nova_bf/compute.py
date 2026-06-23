"""The `compute` phase: intra-worker map-reduce to a per-query top-K.

Each worker:
  1. loads the query embeddings (Q) onto the GPU,
  2. iterates its slice of corpus files (a prefetch thread overlaps IO with
     compute), scoring Q against each file's vectors and folding the file's
     top-K into a running per-query top-K held on the GPU,
  3. decodes the running top-K into hit ids and writes one parquet.

Running top-K stores `(score, encoded)` where `encoded = global_file_idx *
MAX_ROWS_PER_FILE + row`. Keeping an int on the GPU (instead of id strings) makes
the per-file merge a cheap `torch.topk`; ids are recomputed only for the final
K via `make_point_id`, so the whole corpus's ids never materialize.
"""

from __future__ import annotations

import logging
import os

from queue import Queue
from threading import Thread

import numpy as np

from tqdm import tqdm

from nova_bf.config import BruteForceConfig
from nova_bf.ids import make_point_id
from nova_bf.io import Store, dense_to_2d
from nova_bf.results import build_result_table, partial_dir, result_name

logger = logging.getLogger(__name__)

PREFETCH_QUEUE_SIZE = 4
# Per-file id encoding: global_file_idx * MAX_ROWS_PER_FILE + row. Collision-free
# and reversible as long as no single file has more rows than this.
MAX_ROWS_PER_FILE = 100_000_000


def _resolve_rank(num_jobs: int | None, job_rank: int | None) -> int | None:
    if num_jobs is None:
        return None
    if job_rank is None:
        env = os.environ.get("SKYPILOT_JOB_RANK")
        if env is None:
            raise ValueError(
                "job_rank must be provided (or SKYPILOT_JOB_RANK set) when num_jobs is set"
            )
        job_rank = int(env)
    if not 0 <= job_rank < num_jobs:
        raise ValueError(f"job_rank must be in [0, {num_jobs - 1}], got {job_rank}")
    return job_rank


def load_queries(store: Store, qcfg) -> tuple[np.ndarray, list[str], dict[str, list]]:
    cols = [qcfg.dense_column]
    if qcfg.id_column:
        cols.append(qcfg.id_column)
    cols += [c for c in qcfg.payload_fields if c not in cols]

    embs: list[np.ndarray] = []
    ids: list[str] = []
    payload: dict[str, list] = {c: [] for c in qcfg.payload_fields}
    for f in store.list_parquets():
        table = store.read_columns(f.read_path, cols)
        embs.append(dense_to_2d(table[qcfg.dense_column]))
        d = table.to_pydict()
        n = len(table)
        if qcfg.id_column:
            ids += [str(x) for x in d[qcfg.id_column]]
        else:
            ids += [make_point_id(f.key, r) for r in range(n)]
        for c in qcfg.payload_fields:
            payload[c] += d[c]
    Q = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), np.float32)
    return Q, ids, payload


def _scores(Q, C, metric: str):
    import torch.nn.functional as F

    if metric == "cosine":
        # Q is pre-normalized once; normalize C per file.
        return Q @ F.normalize(C, dim=1).T
    if metric == "dot":
        return Q @ C.T
    # euclidean: negate distance so larger = nearer (topk picks nearest).
    import torch

    return -torch.cdist(Q, C)


def run_compute(
    cfg: BruteForceConfig,
    num_jobs: int | None = None,
    job_rank: int | None = None,
) -> str:
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is required for `compute`: install nova-bf[compute]")

    job_rank = _resolve_rank(num_jobs, job_rank)
    k = cfg.params.k
    metric = cfg.params.metric
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU detected — brute force on CPU will be slow.")

    # 1. queries
    Q_np, query_ids, payload = load_queries(Store(cfg.queries.path), cfg.queries)
    n_q, dim = Q_np.shape
    logger.info(
        "queries=%d dim=%d metric=%s k=%d device=%s%s",
        n_q, dim, metric, k, device,
        f" rank={job_rank}/{num_jobs}" if num_jobs else "",
    )

    # 2. corpus files (global, deterministic order); this worker takes a stride
    #    slice so its global indices stay stable for id decoding.
    cstore = Store(cfg.corpus.path)
    all_files = cstore.list_parquets()
    if num_jobs is not None:
        mine = [(i, f) for i, f in enumerate(all_files) if i % num_jobs == job_rank]
    else:
        mine = list(enumerate(all_files))
    logger.info("corpus files: %d total, %d for this worker", len(all_files), len(mine))

    Q = torch.tensor(Q_np, dtype=torch.float32, device=device)
    if metric == "cosine":
        Q = torch.nn.functional.normalize(Q, dim=1)
    top_scores = torch.full((n_q, k), float("-inf"), device=device)
    top_enc = torch.zeros((n_q, k), dtype=torch.int64, device=device)

    # prefetch corpus files (dense column only) in a thread → overlap IO + GPU.
    fq: Queue = Queue(maxsize=PREFETCH_QUEUE_SIZE)
    dense_col = cfg.corpus.dense_column

    def reader():
        for gidx, f in mine:
            table = cstore.read_columns(f.read_path, [dense_col])
            fq.put((gidx, dense_to_2d(table[dense_col])))
        fq.put(None)

    Thread(target=reader, daemon=True).start()

    with tqdm(total=len(mine), unit="file", dynamic_ncols=True, desc="bf") as bar:
        while True:
            item = fq.get()
            if item is None:
                break
            gidx, arr = item
            if len(arr) == 0:
                bar.update(1)
                continue
            C = torch.tensor(arr, dtype=torch.float32, device=device)
            scores = _scores(Q, C, metric)  # (n_q, n_rows)
            file_k = min(k, C.shape[0])
            f_scores, f_local = torch.topk(scores, k=file_k, dim=1)
            rows = torch.arange(C.shape[0], dtype=torch.int64, device=device)
            f_enc = (gidx * MAX_ROWS_PER_FILE + rows)[f_local]
            merged_s = torch.cat([top_scores, f_scores], dim=1)
            merged_e = torch.cat([top_enc, f_enc], dim=1)
            top_scores, idx = torch.topk(merged_s, k=k, dim=1)
            top_enc = merged_e.gather(1, idx)
            bar.update(1)

    # 3. decode the final top-K into hit ids (recompute only K*n_q ids).
    enc = top_enc.cpu().numpy()
    sc = top_scores.cpu().numpy()
    valid = sc > float("-inf")
    hit_ids, hit_scores = [], []
    for q in range(n_q):
        qe, qs = enc[q][valid[q]], sc[q][valid[q]]
        hit_ids.append(
            [
                make_point_id(all_files[int(e) // MAX_ROWS_PER_FILE].key, int(e) % MAX_ROWS_PER_FILE)
                for e in qe
            ]
        )
        hit_scores.append(qs.tolist())

    table = build_result_table(query_ids, payload, hit_ids, hit_scores)
    out = Store(cfg.output.path)
    if num_jobs is not None:
        width = max(3, len(str(num_jobs - 1)))
        name = f"{partial_dir(cfg)}/rank{job_rank:0{width}d}.parquet"
    else:
        name = result_name(cfg)
    path = out.write(name, table)
    logger.info("wrote %s (%d queries)", path, n_q)
    return path
