"""The `merge` phase: inter-worker reduce over per-rank partials.

Each partial holds, per query, the top-K over one worker's disjoint slice of the
corpus (stride partition → no overlapping hits). So merging is just: concatenate
each query's candidates across partials and keep the global top-K. Carried
payload is identical across partials, so we take it from whichever appears first.

The reduce is **streamed, not accumulated**. Every partial holds all Q queries in
the identical row order (each rank scores the same queries against its own corpus
slice, and `load_queries` reads the query store deterministically), so the
partials are *row-aligned by query* — merging is a row-wise reduce across aligned
files, not a keyed group-by. We read the same batch of B queries from all W
partials in lockstep, fold to the global top-K in vectorized numpy, and write that
batch straight out with a ParquetWriter. Peak memory is ~(B × W × k) candidate
slots — independent of Q — so a 1M-query merge runs on a laptop. (The old version
held every candidate for every query in Python objects and built one giant table
at the end; that both OOMed and tripped `to_pylist` on the multi-row-group nested
columns a large partial gets written as.)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

from nova_bf.config import BruteForceConfig
from nova_bf.io import ParquetFile, Store
from nova_bf.results import RESERVED, partial_dir, result_name, warn_if_short

logger = logging.getLogger(__name__)

# Auto batch sizing keeps the (B × W × k) working set near this many candidate
# slots, so memory stays flat as workers/k grow (see ParamsConfig.merge_batch_size).
_TARGET_CANDIDATE_SLOTS = 20_000_000


def _resolve_batch_rows(explicit: int | None, n_rows: int, n_partials: int, k: int) -> int:
    if explicit is not None:
        return max(1, min(explicit, n_rows))
    per_row = max(1, n_partials * k)
    return max(1, min(_TARGET_CANDIDATE_SLOTS // per_row, n_rows))


# Prefetch downloads in parallel RANGES (not one stream per file): a single S3
# connection tops out ~30-50 MB/s, so whole-object copies of a few big partials
# never saturate the NIC. Chunking every file into range GETs and running many at
# once does. In-flight RAM is bounded by (concurrency × chunk).
_DL_CHUNK = 32 * 1024 * 1024  # 32 MiB range GETs
_DL_MAX_CONCURRENCY = 64      # ~enough connections to fill a 25 Gbps NIC; caps RAM


def _prefetch_local(store: Store, partials: list[ParquetFile], workers: int) -> tuple[str, list[str]]:
    """
    Bulk-copy S3 partials to a local temp dir via parallel ranged reads, so even a
    handful of multi-GB partials saturate the NIC (one stream per file does not).
    Returns (dir, local_paths) index-aligned with `partials`.
    """
    tmpdir = tempfile.mkdtemp(prefix="bf_merge_")

    # Plan destinations + sizes, then pre-allocate each local file so ranges can be
    # written to their absolute offset concurrently (os.pwrite is safe for
    # non-overlapping regions). Index-prefix keeps names unique across ranks.
    dsts, total_bytes = [], 0
    tasks: list[tuple[str, str, int, int]] = []  # (src, dst, offset, length)
    for idx, pf in enumerate(partials):
        size = store.fs.get_file_info(pf.read_path).size
        dst = os.path.join(tmpdir, f"{idx:05d}_{os.path.basename(pf.read_path)}")
        with open(dst, "wb") as fh:
            fh.truncate(size)
        dsts.append(dst)
        total_bytes += size
        for off in range(0, max(size, 1), _DL_CHUNK):
            tasks.append((pf.read_path, dst, off, min(_DL_CHUNK, size - off)))

    def fetch(task: tuple[str, str, int, int]) -> int:
        src, dst, off, length = task
        if length <= 0:  # empty partial
            return 0
        f = store.fs.open_input_file(src)
        try:
            buf = f.read_at(length, off)
        finally:
            f.close()
        fd = os.open(dst, os.O_WRONLY)
        try:
            os.pwrite(fd, memoryview(buf), off)
        finally:
            os.close(fd)
        return length

    conc = max(1, min(workers, _DL_MAX_CONCURRENCY))
    with ThreadPoolExecutor(max_workers=conc) as ex, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc="prefetch", dynamic_ncols=True
    ) as bar:
        for n in ex.map(fetch, tasks):
            bar.update(n)
    return tmpdir, dsts


def _topk_merge(
    score_lists: list[pa.ListArray], id_lists: list[pa.ListArray], k: int
) -> tuple[pa.ListArray, pa.ListArray]:
    """Fold W row-aligned (hit_scores, hit_ids) list columns into one global top-K.

    Each input list column is B rows of ≤k already-scored candidates. We scatter
    them into a dense (B, W·k) score/id grid (padding empties with -inf/""), take
    the top-k per row, and re-pack as variable-length lists (dropping the -inf pad,
    so a query with fewer than k total candidates keeps only its real hits).
    """
    n_partials = len(score_lists)
    b = len(score_lists[0])
    width = n_partials * k
    scores = np.full((b, width), -np.inf, dtype=np.float32)
    ids = np.empty((b, width), dtype=object)
    ids[:] = ""

    for w, (sl, il) in enumerate(zip(score_lists, id_lists)):
        lengths = sl.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            continue
        flat_s = sl.flatten().to_numpy(zero_copy_only=False)
        flat_i = il.flatten().to_numpy(zero_copy_only=False)
        row_idx = np.repeat(np.arange(b), lengths)
        starts = np.zeros(b, dtype=np.int64)
        np.cumsum(lengths[:-1], out=starts[1:])
        within = np.arange(total) - np.repeat(starts, lengths)  # position within each row
        col = w * k + within
        scores[row_idx, col] = flat_s
        ids[row_idx, col] = flat_i

    kk = min(k, width)
    if kk < width:
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
    else:
        part = np.broadcast_to(np.arange(width), (b, width)).copy()
    # partition is unordered; sort the kk survivors by score descending.
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_s = np.take_along_axis(scores, top_idx, axis=1)
    top_id = np.take_along_axis(ids, top_idx, axis=1)

    # -inf marks padding (a query with < k total candidates). Sorted descending, so
    # the real hits are a prefix of each row → boolean-mask flatten stays row-grouped.
    valid = np.isfinite(top_s)
    counts = valid.sum(axis=1).astype(np.int32)
    offsets = np.empty(b + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    off = pa.array(offsets, pa.int32())
    scores_arr = pa.ListArray.from_arrays(off, pa.array(top_s[valid], pa.float32()))
    ids_arr = pa.ListArray.from_arrays(off, pa.array(top_id[valid], pa.string()))
    return ids_arr, scores_arr


def run_merge(cfg: BruteForceConfig) -> str:
    out = Store(cfg.output.path)
    partials = out.list_parquets(subpath=partial_dir(cfg))
    if not partials:
        raise RuntimeError(
            f"no partial results under {cfg.output.path}/{partial_dir(cfg)}/ — "
            "run `bf compute --num-jobs N` first"
        )

    workers = max(1, cfg.params.io_workers)
    staged_dir: str | None = None
    if cfg.params.merge_prefetch and out.is_s3:
        logger.info("prefetching %d partials to local disk (%d workers)…", len(partials), workers)
        staged_dir, local_paths = _prefetch_local(out, partials, workers)
        gb = sum(os.path.getsize(p) for p in local_paths) / 1e9
        logger.info("staged %.1f GB to %s; merging from local disk", gb, staged_dir)
        readers = [pq.ParquetFile(p) for p in local_paths]
    else:
        if cfg.params.merge_prefetch:
            logger.info("merge_prefetch set but partials are already local — reading in place")
        readers = [pq.ParquetFile(f.read_path, filesystem=out.fs) for f in partials]

    try:
        return _reduce(cfg, out, partials, readers)
    finally:
        if staged_dir is not None:
            shutil.rmtree(staged_dir, ignore_errors=True)


def _reduce(
    cfg: BruteForceConfig, out: Store, partials: list[ParquetFile], readers: list[pq.ParquetFile]
) -> str:
    k = cfg.params.k
    n_rows = readers[0].metadata.num_rows
    for f, r in zip(partials, readers):
        if r.metadata.num_rows != n_rows:
            raise RuntimeError(
                f"partial {f.read_path} has {r.metadata.num_rows} rows but the first "
                f"partial has {n_rows}; partials must be row-aligned by query "
                "(same queries, same order). A truncated/mismatched partial can't be merged."
            )

    payload_cols = [c for c in readers[0].schema_arrow.names if c not in RESERVED]
    batch_rows = _resolve_batch_rows(cfg.params.merge_batch_size, n_rows, len(partials), k)
    logger.info(
        "merging %d partials (%d queries, k=%d) in batches of %d",
        len(partials), n_rows, k, batch_rows,
    )

    # Read query_id + payload only from partial 0; every partial contributes its
    # hit lists (and query_id, for a cheap per-batch alignment check).
    base_iter = readers[0].iter_batches(batch_size=batch_rows)
    hit_cols = ["query_id", "hit_ids", "hit_scores"]
    rest_iters = [r.iter_batches(batch_size=batch_rows, columns=hit_cols) for r in readers[1:]]

    path = f"{out.root.rstrip('/')}/{result_name(cfg)}"
    if not out.is_s3:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    sink = out.fs.open_output_stream(path)
    writer: pq.ParquetWriter | None = None

    # Running count of queries whose FINAL top-K came out short of k, tallied
    # per batch via a vectorized length check (same call _topk_merge already
    # uses internally) — never materializes a Python hit_ids list, which is
    # exactly what this streaming rewrite exists to avoid.
    short_count = 0

    total_batches = (n_rows + batch_rows - 1) // batch_rows
    try:
        with tqdm(total=total_batches, unit="batch", desc="merge", dynamic_ncols=True) as bar:
            for base in base_iter:
                rest = [next(it) for it in rest_iters]
                for other in rest:
                    if not base.column("query_id").equals(other.column("query_id")):
                        raise RuntimeError(
                            "partials are not row-aligned: a batch's query_id column "
                            "differs across partials. Re-run `bf compute` so every rank "
                            "writes the same queries in the same order."
                        )
                score_lists = [base.column("hit_scores")] + [o.column("hit_scores") for o in rest]
                id_lists = [base.column("hit_ids")] + [o.column("hit_ids") for o in rest]
                ids_arr, scores_arr = _topk_merge(score_lists, id_lists, k)

                lengths = ids_arr.value_lengths().to_numpy(zero_copy_only=False)
                short_count += int((lengths < k).sum())

                cols = {"query_id": base.column("query_id")}
                for c in payload_cols:
                    cols[c] = base.column(c)
                cols["hit_ids"] = ids_arr
                cols["hit_scores"] = scores_arr
                table = pa.table(cols)

                if writer is None:
                    writer = pq.ParquetWriter(sink, table.schema, compression="snappy")
                writer.write_table(table)
                bar.update(1)
    finally:
        if writer is not None:
            writer.close()
        sink.close()

    warn_if_short(short_count, n_rows, k, logger)
    logger.info("wrote %s (%d queries)", path, n_rows)
    return path
