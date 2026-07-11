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

from nova_bf.config import BruteForceConfig, SearchSpec
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


def _plan_prefetch(store: Store, partials: list[ParquetFile]) -> tuple[str, list[str], list[tuple[str, str, int, int]]]:
    """Plan (but don't run) a prefetch: allocate a temp dir + pre-size each
    local destination file, and return the flat (src, dst, offset, length)
    range-GET task list needed to fill them in. Pure bookkeeping, no IO —
    so several searches' plans can be built and their task lists concatenated
    before any fetching starts, letting them share ONE download pool (see
    `run_merge`) instead of one pool per search.
    """
    tmpdir = tempfile.mkdtemp(prefix="bf_merge_")

    # Pre-allocate each local file so ranges can be written to their absolute
    # offset concurrently (os.pwrite is safe for non-overlapping regions).
    # Index-prefix keeps names unique across ranks.
    dsts: list[str] = []
    tasks: list[tuple[str, str, int, int]] = []  # (src, dst, offset, length)
    for idx, pf in enumerate(partials):
        size = store.fs.get_file_info(pf.read_path).size
        dst = os.path.join(tmpdir, f"{idx:05d}_{os.path.basename(pf.read_path)}")
        with open(dst, "wb") as fh:
            fh.truncate(size)
        dsts.append(dst)
        for off in range(0, max(size, 1), _DL_CHUNK):
            tasks.append((pf.read_path, dst, off, min(_DL_CHUNK, size - off)))
    return tmpdir, dsts, tasks


def _fetch_range(store: Store, task: tuple[str, str, int, int]) -> int:
    """Worker function for one range-GET task (see `_plan_prefetch`) — a pure
    function of `(store, task)`, so the SAME thread pool can run tasks planned
    for any search without needing per-search state."""
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


def run_merge(cfg: BruteForceConfig) -> dict[str, str]:
    """Merges every search in `cfg.searches` — each spec's per-rank partials
    (written under its own `partial_dir`, see `compute.py`) are reduced
    independently into that spec's own final parquet. Returns
    `{spec.name: output_path}`.

    When `merge_prefetch` is on and the output is S3, every search's partial
    downloads are planned up front (see `_plan_prefetch`) and run on ONE
    shared thread pool, not one pool per search and not one search's downloads
    fully finishing before the next search's even start. A search with fewer
    or smaller partials frees its share of the pool the moment its own
    downloads land, instead of sitting on dedicated capacity a slower search
    could have used — the same reason `compute.py` uses a shared reader-thread
    pool across corpus files rather than reading them one at a time.
    """
    out = Store(cfg.output.path)

    partials_by_name: dict[str, list[ParquetFile]] = {}
    for spec in cfg.searches:
        partials = out.list_parquets(subpath=partial_dir(cfg, spec))
        if not partials:
            raise RuntimeError(
                f"no partial results under {cfg.output.path}/{partial_dir(cfg, spec)}/ "
                f"(search={spec.name!r}) — run `bf compute --num-jobs N` first"
            )
        partials_by_name[spec.name] = partials

    staged_dirs: list[str] = []
    readers_by_name: dict[str, list[pq.ParquetFile]] = {}
    try:
        if cfg.params.merge_prefetch and out.is_s3:
            _prefetch_all(cfg, out, partials_by_name, staged_dirs, readers_by_name)
        else:
            if cfg.params.merge_prefetch:
                logger.info("merge_prefetch set but partials are already local — reading in place")
            for spec in cfg.searches:
                readers_by_name[spec.name] = [
                    pq.ParquetFile(f.read_path, filesystem=out.fs) for f in partials_by_name[spec.name]
                ]

        return {
            spec.name: _reduce(cfg, spec, out, partials_by_name[spec.name], readers_by_name[spec.name])
            for spec in cfg.searches
        }
    finally:
        for d in staged_dirs:
            shutil.rmtree(d, ignore_errors=True)


def _prefetch_all(
    cfg: BruteForceConfig,
    out: Store,
    partials_by_name: dict[str, list[ParquetFile]],
    staged_dirs: list[str],
    readers_by_name: dict[str, list[pq.ParquetFile]],
) -> None:
    """Plan every search's prefetch, then run every search's download tasks
    on ONE shared pool — see `run_merge`'s docstring for why this beats a
    separate pool (or a fully sequential loop) per search."""
    plans: dict[str, tuple[str, list[str]]] = {}
    all_tasks: list[tuple[str, str, int, int]] = []
    for name, partials in partials_by_name.items():
        tmpdir, dsts, tasks = _plan_prefetch(out, partials)
        plans[name] = (tmpdir, dsts)
        staged_dirs.append(tmpdir)
        all_tasks += tasks

    total_bytes = sum(t[3] for t in all_tasks)
    conc = max(1, min(cfg.params.io_workers, _DL_MAX_CONCURRENCY))
    logger.info(
        "prefetching %d partials across %d searches to local disk (%d shared workers)…",
        sum(len(p) for p in partials_by_name.values()), len(partials_by_name), conc,
    )
    with ThreadPoolExecutor(max_workers=conc) as ex, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc="prefetch", dynamic_ncols=True
    ) as bar:
        for n in ex.map(lambda t: _fetch_range(out, t), all_tasks):
            bar.update(n)

    for name, (tmpdir, dsts) in plans.items():
        gb = sum(os.path.getsize(p) for p in dsts) / 1e9
        logger.info("search=%r: staged %.1f GB to %s; merging from local disk", name, gb, tmpdir)
        readers_by_name[name] = [pq.ParquetFile(p) for p in dsts]


def _reduce(
    cfg: BruteForceConfig, spec: SearchSpec, out: Store, partials: list[ParquetFile],
    readers: list[pq.ParquetFile],
) -> str:
    k = spec.k
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
        "search=%r: merging %d partials (%d queries, k=%d) in batches of %d",
        spec.name, len(partials), n_rows, k, batch_rows,
    )

    # Read query_id + payload only from partial 0; every partial contributes its
    # hit lists (and query_id, for a cheap per-batch alignment check).
    base_iter = readers[0].iter_batches(batch_size=batch_rows)
    hit_cols = ["query_id", "hit_ids", "hit_scores"]
    rest_iters = [r.iter_batches(batch_size=batch_rows, columns=hit_cols) for r in readers[1:]]

    path = f"{out.root.rstrip('/')}/{result_name(cfg, spec)}"
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

    warn_if_short(short_count, n_rows, k, spec.name, logger)
    logger.info("search=%r wrote %s (%d queries)", spec.name, path, n_rows)
    return path
