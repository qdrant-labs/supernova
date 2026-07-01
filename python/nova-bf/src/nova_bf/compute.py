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

If `corpus.id_column` is set, hit ids come from that pre-existing column instead
of `make_point_id`. Such an id isn't recomputable from (file, row), so it's read
alongside the dense column and kept in RAM per file (the worker's slice) to resolve
the final top-K. Large files can be scored in row-batches (`params.corpus_batch_size`)
to bound the per-file score matrix `(n_queries × rows)` on the GPU.

If `filter` is set (see `nova_bf.filters`), each file's payload columns are
read alongside the dense column and evaluated into a keep-mask *before*
scoring, so filtered-out rows never reach the GPU. The mask compacts `arr` but
never renumbers rows — `orig_rows` tracks each surviving row's true file-row
number, since both `make_point_id` and the `id_column` lookup are keyed on it.
Mask evaluation is timed separately from the read (`filter_secs`, reported
alongside `read_secs`/`io_wait`/`gpu_secs`) so a slow filter is distinguishable
from slow IO in the end-of-run timing log.
"""

from __future__ import annotations

import logging
import os
import time

from queue import Empty, Queue
from threading import Thread

import numpy as np

from tqdm import tqdm

from nova_bf.config import BruteForceConfig
from nova_bf.filters import evaluate
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
    io_workers: int | None = None,
    io_thread_count: int | None = None,
    max_files: int | None = None,
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
    if cfg.filter is not None:
        logger.info("filter: %s", cfg.filter.model_dump(exclude_defaults=True))

    # 2. corpus files (global, deterministic order); this worker takes a stride
    #    slice so its global indices stay stable for id decoding.
    cstore = Store(cfg.corpus.path)
    all_files = cstore.list_parquets()
    if num_jobs is not None:
        mine = [(i, f) for i, f in enumerate(all_files) if i % num_jobs == job_rank]
    else:
        mine = list(enumerate(all_files))
    logger.info("corpus files: %d total, %d for this worker", len(all_files), len(mine))
    if max_files is not None and max_files < len(mine):
        logger.warning(
            "--max-files=%d: benchmarking on the first %d of %d slice files — "
            "OUTPUT WILL BE PARTIAL (not valid ground truth).",
            max_files, max_files, len(mine),
        )
        mine = mine[:max_files]

    Q = torch.tensor(Q_np, dtype=torch.float32, device=device)
    if metric == "cosine":
        Q = torch.nn.functional.normalize(Q, dim=1)
    top_scores = torch.full((n_q, k), float("-inf"), device=device)
    top_enc = torch.zeros((n_q, k), dtype=torch.int64, device=device)

    # Prefetch corpus files (dense column only) with a POOL of reader threads so
    # many S3 GETs are in flight at once — otherwise the GPU sits idle behind one
    # file's latency at a time. Order doesn't matter (the top-K merge is
    # commutative). pyarrow releases the GIL during IO, so threads parallelize.
    dense_col = cfg.corpus.dense_column
    id_col = cfg.corpus.id_column  # None → derive make_point_id(file_key, row) at decode
    filt = cfg.filter  # None → no filtering; else a corpus-side payload predicate
    filter_cols = sorted(filt.fields()) if filt else []
    read_cols = list(dict.fromkeys([dense_col] + ([id_col] if id_col else []) + filter_cols))
    # Corpus id strings carried through to decode, kept in RAM per file (only when
    # id_column is set). gidx → pyarrow string array aligned with that file's rows.
    corpus_ids: dict[int, object] = {}
    # Score each file in row-batches to bound the (n_q × rows) score matrix; None =
    # whole file. Below k is pointless (a batch can't fill the top-K and isn't
    # smaller than the resident n_q × k state), so raise it to k with a warning.
    corpus_batch = cfg.params.corpus_batch_size
    if corpus_batch is not None and corpus_batch < k:
        logger.warning(
            "corpus_batch_size=%d is below k=%d; raising to k (a smaller batch can't "
            "fill the top-K and gives no memory benefit).", corpus_batch, k,
        )
        corpus_batch = k
    io_workers = max(1, io_workers if io_workers is not None else cfg.params.io_workers)
    itc = io_thread_count if io_thread_count is not None else cfg.params.io_thread_count
    if itc and itc > 0:
        import pyarrow as pa
        pa.set_io_thread_count(itc)
        logger.info("pyarrow IO thread pool set to %d (true S3 fetch concurrency)", itc)
    work: Queue = Queue()
    for item in mine:
        work.put(item)
    fq: Queue = Queue(maxsize=io_workers * 2)  # bounded → backpressure on readers

    def reader():
        while True:
            try:
                gidx, f = work.get_nowait()
            except Empty:
                return
            t0 = time.perf_counter()
            table = cstore.read_columns(f.read_path, read_cols)
            arr = dense_to_2d(table[dense_col])
            # carry the id column (combined to one contiguous array) to decode;
            # None when id_column isn't configured. Same row order as `arr`.
            ids = table[id_col].combine_chunks() if id_col else None
            t1 = time.perf_counter()
            # None when unfiltered; else a boolean mask over this file's rows,
            # ORIGINAL row order preserved (no compaction here — see consumer).
            # Timed separately from the read above: it's CPU-vectorized work over
            # the whole file, not IO wait, and lumping it into `read_secs` would
            # make an expensive filter look like slow S3, not slow filtering.
            keep = evaluate(filt, table) if filt else None
            t2 = time.perf_counter()
            fq.put((gidx, arr, ids, keep, t1 - t0, t2 - t1))

    for _ in range(io_workers):
        Thread(target=reader, daemon=True).start()

    # Timing split (debug): `io_wait` is real time the consumer blocked on an
    # empty queue == the GPU starved waiting for reads — the number that matters
    # here. `gpu_secs` is just CPU-side enqueue time (CUDA is async and overlaps
    # the next read), so it being tiny is itself evidence we're not compute-bound.
    # `read_secs` is summed per-file read latency across the reader threads.
    # `filter_secs` is summed per-file mask-evaluation time (0 when unfiltered),
    # also across the reader threads — kept apart from `read_secs` so a slow
    # filter doesn't masquerade as slow IO.
    io_wait = gpu_secs = read_secs = filter_secs = 0.0
    rows_seen = 0
    bytes_seen = 0  # decoded float32 bytes consumed (~= wire bytes for snappy-float32)
    wall0 = time.perf_counter()

    with tqdm(total=len(mine), unit="file", dynamic_ncols=True, desc="bf") as bar:
        for _ in range(len(mine)):
            w0 = time.perf_counter()
            gidx, arr, ids, keep, rsec, fsec = fq.get()
            io_wait += time.perf_counter() - w0
            read_secs += rsec
            filter_secs += fsec
            bar.update(1)
            # `orig_rows[i]` is the TRUE file row that arr[i] came from — identity
            # when unfiltered, else the surviving indices of `keep`. Filtering
            # compacts `arr` (fewer/smaller matmuls) but must never renumber rows:
            # both make_point_id(file, row) and corpus_ids[gidx][row] need the row
            # the vector actually occupies in the parquet file, not its position
            # in the filtered-down array.
            if keep is not None:
                orig_rows = np.nonzero(keep)[0]
                arr = arr[orig_rows]
            else:
                orig_rows = None
            if len(arr) == 0:
                continue
            rows_seen += len(arr)
            bytes_seen += int(arr.nbytes)
            if id_col:
                corpus_ids[gidx] = ids  # kept in RAM; indexed by row at decode
                # unfiltered — always the full per-file array, so `corpus_ids[gidx][row]`
                # resolves correctly regardless of `keep`.

            g0 = time.perf_counter()
            # zero-copy view of the (contiguous float32) array + one H2D copy;
            # torch.tensor() would add an extra host→host copy first.
            C = torch.from_numpy(arr).to(device, non_blocking=True)
            n_rows = C.shape[0]
            step = corpus_batch or n_rows  # None → whole file in one matmul
            for r0 in range(0, n_rows, step):
                Cb = C[r0 : r0 + step]
                scores = _scores(Q, Cb, metric)  # (n_q, ≤step)
                bk = min(k, Cb.shape[0])
                f_scores, f_local = torch.topk(scores, k=bk, dim=1)
                # rows are the TRUE file row for each entry of Cb (see orig_rows
                # above), so the encoding stays global_file_idx * MAX_ROWS_PER_FILE
                # + row regardless of batching or filtering.
                if orig_rows is not None:
                    rows = torch.from_numpy(orig_rows[r0 : r0 + Cb.shape[0]]).to(device)
                else:
                    rows = torch.arange(r0, r0 + Cb.shape[0], dtype=torch.int64, device=device)
                f_enc = (gidx * MAX_ROWS_PER_FILE + rows)[f_local]
                merged_s = torch.cat([top_scores, f_scores], dim=1)
                merged_e = torch.cat([top_enc, f_enc], dim=1)
                top_scores, idx = torch.topk(merged_s, k=k, dim=1)
                top_enc = merged_e.gather(1, idx)
            gpu_secs += time.perf_counter() - g0

            if bar.n % 200 == 0:
                postfix = f"io_wait={io_wait:.0f}s gpu={gpu_secs:.0f}s"
                if filt:
                    postfix += f" filter={filter_secs:.0f}s"
                bar.set_postfix_str(postfix, refresh=False)

    wall = time.perf_counter() - wall0
    gb = bytes_seen / 1e9
    wall_mbps = bytes_seen / 1e6 / max(wall, 1e-9)         # effective aggregate S3 throughput
    stream_mbps = bytes_seen / 1e6 / max(read_secs, 1e-9)  # avg single-connection throughput
    logger.info(
        "timing: %d files / %d rows / %.2f GB in %.1fs | consumer io_wait=%.1fs gpu=%.1fs | "
        "read latency avg=%.3fs/file (summed %.0fs over %d threads)%s",
        len(mine), rows_seen, gb, wall, io_wait, gpu_secs,
        read_secs / max(1, len(mine)), read_secs, io_workers,
        f" | filter eval avg={filter_secs / max(1, len(mine)):.3f}s/file "
        f"(summed {filter_secs:.0f}s over {io_workers} threads)" if filt else "",
    )
    # One machine-parseable line per run — for sweeping io_workers and plotting.
    # `wall_mbps` is the effective aggregate download rate; compare it to the
    # instance's *sustained* NIC baseline (g5.xlarge ≈ 310 MB/s) to tell whether
    # you're NIC-bound (plateaus there) or still latency-bound (keeps rising).
    # `filter_s` is 0.0 when unfiltered — always present so the line's schema
    # stays stable for scripts parsing it across both filtered and plain runs.
    logger.info(
        "bf-bench io_workers=%d files=%d rows=%d gb=%.3f wall_s=%.1f "
        "wall_mbps=%.1f stream_mbps=%.1f io_wait_s=%.1f gpu_s=%.1f filter_s=%.1f",
        io_workers, len(mine), rows_seen, gb, wall,
        wall_mbps, stream_mbps, io_wait, gpu_secs, filter_secs,
    )
    if io_wait > 3 * max(gpu_secs, 1e-6):
        logger.info(
            "IO-bound: GPU idle %.0f%% of the time waiting on reads — raise "
            "params.io_workers (currently %d).",
            100 * io_wait / max(io_wait + gpu_secs, 1e-9), io_workers,
        )

    # 3. decode the final top-K into hit ids. Either recompute make_point_id from
    #    (file_key, row) — only K*n_q ids, nothing corpus-wide — or, when an id
    #    column is configured, read it back from the in-RAM per-file arrays.
    if id_col is not None:
        def resolve_id(e: int) -> str:
            gidx = e // MAX_ROWS_PER_FILE
            row = e % MAX_ROWS_PER_FILE
            return str(corpus_ids[gidx][row].as_py())
    else:
        def resolve_id(e: int) -> str:
            return make_point_id(
                all_files[e // MAX_ROWS_PER_FILE].key, e % MAX_ROWS_PER_FILE
            )

    enc = top_enc.cpu().numpy()
    sc = top_scores.cpu().numpy()
    valid = sc > float("-inf")
    hit_ids, hit_scores = [], []
    for q in range(n_q):
        qe, qs = enc[q][valid[q]], sc[q][valid[q]]
        hit_ids.append([resolve_id(int(e)) for e in qe])
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
