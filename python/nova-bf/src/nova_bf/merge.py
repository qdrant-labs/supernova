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
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

from nova_bf import manifest as run_manifest
from nova_bf.config import BruteForceConfig, SearchSpec
from nova_bf.io import ParquetFile, Store
from nova_bf.results import (
    CONFIG_KEY,
    JOB_RANK_KEY,
    NUM_JOBS_KEY,
    RESERVED,
    RUN_KEY,
    TIEBREAK_KEY,
    config_identity,
    partial_dir,
    provenance,
    result_name,
    warn_if_short,
)

logger = logging.getLogger(__name__)

# Auto batch sizing keeps the (B × W × k) working set near this many candidate
# slots, so memory stays flat as workers/k grow (see ParamsConfig.merge_batch_size).
_TARGET_CANDIDATE_SLOTS = 20_000_000


def _resolve_batch_rows(explicit: int | None, n_rows: int, n_partials: int, k: int) -> int:
    """Choose how many query rows one merge batch covers.

    Automatic sizing (`explicit is None`) targets `_TARGET_CANDIDATE_SLOTS`, so
    peak memory stays flat as the partial count and `k` grow.

    An EXPLICIT `params.merge_batch_size` is obeyed, bounded only by the query
    count. However, if the value is above the auto target, a warning is logged.
    """
    per_row = max(1, n_partials * k)
    ceiling = max(1, min(_TARGET_CANDIDATE_SLOTS // per_row, n_rows))
    if explicit is None:
        return ceiling
    want = max(1, min(explicit, n_rows))
    if want > ceiling:
        logger.warning(
            "params.merge_batch_size=%d holds %.1f M candidate slots per batch "
            "(%d partials x k=%d), above the ~%.1f M auto target — honoring it, "
            "since you set it. The grid scales with BOTH the partial count and "
            "k, so a value tuned for fewer shards asks for more here; drop the "
            "setting to let merge size itself (%d rows) if this OOMs.",
            explicit, want * per_row / 1e6, n_partials, k,
            _TARGET_CANDIDATE_SLOTS / 1e6, ceiling,
        )
    return want


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


def _string_ranks(ids: np.ndarray) -> np.ndarray:
    """Replace string IDs with dense integer ranks preserving their order.

    The distinct IDs in this block are sorted lexicographically and each ID is
    replaced by its position in that ordering. Comparing the resulting ranks is
    therefore exactly equivalent to comparing the full IDs, without truncation
    or hashing.

    IDs are converted to fixed-width byte strings when possible so `np.unique`
    can rank them with a compact native representation. Non-ASCII IDs fall back
    to fixed-width Unicode. The returned ranks are reshaped to match `ids`.
    """
    
    # Fixed-width first either way: `np.unique` over an OBJECT array compares
    # through Python, orders of magnitude slower than a fixed-width sort over
    # the same values.
    #
    # `astype("S")` is ASCII-only and raises on anything else, so non-ASCII ids
    # fall back to the wide form. The two orders agree wherever both apply:
    # bytewise order over UTF-8 IS code-point order, which is also the order
    # `compute` ranked on (pyarrow sorts strings bytewise) — so the fallback
    # changes what this costs, never what it decides.
    try:
        keys = ids.astype("S")
    except UnicodeEncodeError:
        keys = ids.astype("U")
    _, inv = np.unique(keys, return_inverse=True)
    return inv.reshape(ids.shape)


def _ambiguous_rows(scores: np.ndarray, top_s: np.ndarray) -> np.ndarray:
    """Return rows whose score-only top-k result does not uniquely determine the hits.

    A row is ambiguous if either:
      * two selected hits have the same score, so their relative order requires
        a tiebreak; or
      * the cutoff score is shared by both selected and unselected candidates,
        so the top-k membership requires a tiebreak.

    `-inf` entries are padding for missing candidates and are ignored.
    """
    # `> -inf`, not `isfinite`: `-inf` is the padding, but `+inf` is a real hit
    # that survives the write path below, so a tie between two of them still
    # has to be resolved rather than declared unambiguous.
    real = top_s > -np.inf
    amb = ((top_s[:, 1:] == top_s[:, :-1]) & real[:, :-1]).any(axis=1)

    tau = top_s[:, -1]
    fin_tau = tau > -np.inf
    if fin_tau.any():
        n_all = (scores == tau[:, None]).sum(axis=1)
        n_kept = (top_s == tau[:, None]).sum(axis=1)
        amb |= fin_tau & (n_all > n_kept)
    return amb


def _topk_merge(
    score_lists: list[pa.ListArray],
    id_lists: list[pa.ListArray],
    tie_lists: list[pa.ListArray] | None,
    k: int,
) -> tuple[pa.ListArray, pa.ListArray]:
    """Fold W row-aligned (hit_scores, hit_ids) list columns into one global top-K.

    Each input list column is B rows of ≤k already-scored candidates. We scatter
    them into a dense (B, W·k) score/id grid (padding empties with -inf/""), take
    the top-k per row, and re-pack as variable-length lists (dropping the -inf pad,
    so a query with fewer than k total candidates keeps only its real hits).

    Ties are resolved by the SAME rule each worker applied within itself, so the
    result does not depend on how many workers produced it. 
    """
    n_partials = len(score_lists)
    b = len(score_lists[0])
    width = n_partials * k
    scores = np.full((b, width), -np.inf, dtype=np.float32)
    ids = np.empty((b, width), dtype=object)
    ids[:] = ""
    # Only built when a numeric ordinate travelled with the partials. Skipped
    # otherwise, since it is the same size as the score grid.
    want_tie = tie_lists is not None
    ties = np.full((b, width), np.iinfo(np.int64).max, dtype=np.int64) if want_tie else None

    for w, (sl, il) in enumerate(zip(score_lists, id_lists)):
        lengths = sl.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            continue
        # Guard against out-of-index access
        # rather than an expected path.
        if lengths.max() > k:
            raise RuntimeError(
                f"partial {w} has a query with {int(lengths.max())} hits but k={k}; "
                "a partial must never hold more than k candidates per query. "
                "Re-run `bf compute` for this search."
            )
        flat_s = sl.flatten().to_numpy(zero_copy_only=False)
        flat_i = il.flatten().to_numpy(zero_copy_only=False)
        row_idx = np.repeat(np.arange(b), lengths)
        starts = np.zeros(b, dtype=np.int64)
        np.cumsum(lengths[:-1], out=starts[1:])
        within = np.arange(total) - np.repeat(starts, lengths)  # position within each row
        col = w * k + within
        scores[row_idx, col] = flat_s
        ids[row_idx, col] = flat_i
        if want_tie:
            tl = tie_lists[w]
            tie_lengths = tl.value_lengths().to_numpy(zero_copy_only=False)
            if not np.array_equal(tie_lengths.astype(np.int64), lengths):
                raise RuntimeError(
                    f"partial {w}'s hit_tie rows are split differently from its "
                    "hit_scores rows; the columns must line up row for row or ties "
                    "would be broken against the wrong hits. Re-run `bf compute` "
                    "for this search."
                )
            ties[row_idx, col] = tl.flatten().to_numpy(zero_copy_only=False)

    kk = min(k, width)
    if kk < width:
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
    else:
        part = np.broadcast_to(np.arange(width), (b, width)).copy()
    # partition is unordered; sort the kk survivors by score descending.
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_s = np.take_along_axis(scores, top_idx, axis=1)

    # The score-only cut above is unstable, so redo exactly the rows where a
    # real tie means it decided something arbitrarily.
    amb = _ambiguous_rows(scores, top_s)
    if amb.any():
        rows = np.flatnonzero(amb)
        tie = ties[rows] if want_tie else _string_ranks(ids[rows])
        # Last key is primary: score descending, then tiebreak ascending. The
        # -inf padding negates to +inf and so always sorts past every real hit.
        exact = np.lexsort((tie, -scores[rows]), axis=1)[:, :kk]
        top_idx[rows] = exact
        top_s[rows] = np.take_along_axis(scores[rows], exact, axis=1)

    top_id = np.take_along_axis(ids, top_idx, axis=1)

    # -inf marks padding (a query with < k total candidates). Sorted descending, so
    # the real hits are a prefix of each row → boolean-mask flatten stays row-grouped.
    # The test is `> -inf`, not `isfinite`: a `+inf` score is a REAL hit that a
    # single-node run emits, and dropping it here would make a sharded result
    # differ from an unsharded one on the same corpus.
    valid = top_s > -np.inf
    counts = valid.sum(axis=1).astype(np.int32)
    offsets = np.empty(b + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    off = pa.array(offsets, pa.int32())
    scores_arr = pa.ListArray.from_arrays(off, pa.array(top_s[valid], pa.float32()))
    ids_arr = pa.ListArray.from_arrays(
        off, pa.array(top_id[valid], pa.large_string()))
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

    # Every search in one `compute` run is written by the SAME set of ranks
    # (each rank writes one partial per search, back to back, in a single
    # invocation) — so with more than one search, every search must end up
    # with the identical partial COUNT. A mismatch means some rank died
    # partway through writing its per-search partials (crash/OOM/preemption
    # between two of its writes), silently leaving one search's merge short
    # a rank's worth of candidates with no other signal. Catch it here,
    # before reducing, rather than let it manifest as a suspiciously-low
    # top-K for just that search.
    if len(partials_by_name) > 1:
        counts = {name: len(partials) for name, partials in partials_by_name.items()}
        if len(set(counts.values())) > 1:
            raise RuntimeError(
                f"searches have mismatched partial counts: {counts} — every search in "
                "one `compute` run should have the same number of per-rank partials; "
                "this points to a rank that died partway through writing its per-search "
                "outputs (crash/OOM/preemption). Re-run the missing rank(s) with "
                "`bf compute --num-jobs N --job-rank R` before merging."
            )

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    staged_dirs: list[str] = []
    readers_by_name: dict[str, list[pq.ParquetFile]] = {}
    entries: list[dict] = []
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

        entries = [
            _reduce(cfg, spec, out, partials_by_name[spec.name], readers_by_name[spec.name])
            for spec in cfg.searches
        ]
    finally:
        for d in staged_dirs:
            shutil.rmtree(d, ignore_errors=True)

    # The run manifest for the reduce half — the compute manifests describe one
    # rank's slice each; this one describes the artifact people actually consume
    # (see manifest.py). Written after the staging cleanup so it records a
    # finished merge, and best-effort, so it cannot fail one.
    doc = run_manifest.base_manifest(cfg, "merge")
    doc.update({
        "started_at": started_at.isoformat(),
        "searches": entries,
        "counts": {
            # Partial COUNT is the merge's own sharding record: it is how many
            # ranks' candidates were actually folded in, which the final parquet
            # cannot tell you and a short merge (a dead rank) hinges on.
            "partials_merged": sum(e["partials"] for e in entries),
            "queries": max((e["queries"] for e in entries), default=0),
        },
        "timing": {"elapsed_seconds": round(time.perf_counter() - t0, 2)},
        "output_files": [e["output_file"] for e in entries],
    })
    run_manifest.write(out, run_manifest.manifest_name(cfg, "merge"), doc)
    return {e["name"]: e["output_path"] for e in entries}


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


def _validate_one_run(
    cfg: BruteForceConfig,
    spec: SearchSpec,
    partials: list[ParquetFile],
    readers: list[pq.ParquetFile],
) -> str | None:
    """Refuse to merge partials that did not come from ONE run, and refuse a
    run that is missing a rank. Returns the run fingerprint to carry onto the
    merged artifact.

    A search's partial directory is addressed by (queries stem, search name, k)
    alone, so any two runs agreeing on those three write into it. Files are
    named `rank<NNN>.parquet`, so a re-run overwrites only the ranks it has:
    32 ranks landing on a 64-rank run's leftovers produce a directory of 64
    partials, half of them stale. Nothing about the rows says so — the stale
    slices merge cleanly, double-counting the corpus regions they overlap and
    missing the ones nobody covered, and the output is a wrong top-K that looks
    entirely normal.

    Three checks, in the order a failure is most likely:

    1. every partial carries the same `run_fingerprint` (see
       `results.run_identity`) — this is the mixed-runs case;
    2. every partial's `config_fingerprint` matches the config THIS merge was
       handed — the same idea as the existing `tiebreak` check, extended to
       every other field that changes results (tiebreak keeps its own check,
       whose message is more specific than this one);
    3. the ranks present are exactly `0..num_jobs-1` — this is the missing-rank
       case, which the old "every search has the same partial count" check
       cannot see, because a rank that dies before writing anything leaves
       every search short by exactly one.
    """
    stamps = [(f, r.schema_arrow.metadata or {}) for f, r in zip(partials, readers)]

    def _get(meta: dict, key: bytes) -> str | None:
        value = meta.get(key)
        return value.decode() if value is not None else None

    runs = {f.read_path: _get(meta, RUN_KEY) for f, meta in stamps}
    present = {sha for sha in runs.values() if sha is not None}
    if not present:
        # Every partial predates run fingerprinting. Nothing to compare, and
        # refusing would strand hours of legitimately-produced GPU work over a
        # check that would have passed — so this is loud, not fatal.
        logger.warning(
            "search=%r: none of the %d partials carry a run fingerprint (they "
            "predate it) — cannot verify they came from a single run. Re-run "
            "`bf compute` if this directory may hold partials from more than one.",
            spec.name, len(partials),
        )
        return None
    unstamped = sorted(p for p, sha in runs.items() if sha is None)
    if unstamped or len(present) > 1:
        by_run: dict[str, list[str]] = {}
        for path, sha in runs.items():
            by_run.setdefault(sha or "(unstamped)", []).append(path)
        summary = "; ".join(
            f"{sha[:12] if sha != '(unstamped)' else sha}: {len(paths)} partial(s) "
            f"e.g. {sorted(paths)[0]}"
            for sha, paths in sorted(by_run.items())
        )
        raise RuntimeError(
            f"search={spec.name!r}: the partials under "
            f"{cfg.output.path}/{partial_dir(cfg, spec)}/ come from MORE THAN ONE "
            f"run — {summary}. Merging them would double-count the corpus where "
            "their slices overlap and miss it where neither covered, producing a "
            "wrong top-K that looks normal. Delete the directory and re-run "
            "`bf compute` for this search."
        )
    run_sha = present.pop()

    want_config = config_identity(cfg, spec)
    mismatched = [
        f.read_path for f, meta in stamps
        if (_get(meta, CONFIG_KEY) or want_config) != want_config
    ]
    if mismatched:
        raise RuntimeError(
            f"search={spec.name!r}: partial {mismatched[0]} was computed from a "
            "different config than this merge was given (metric/k/filter/rows, "
            "the corpus or queries paths and columns, or `allow_tf32` differ). "
            "Merge with the config that produced these partials, or re-run "
            "`bf compute`."
        )

    declared = {_get(meta, NUM_JOBS_KEY) for _, meta in stamps}
    if declared == {None}:
        return run_sha  # single-node partials, or pre-stamp: no rank set to check
    if len(declared) > 1:
        raise RuntimeError(
            f"search={spec.name!r}: partials disagree about how many ranks the "
            f"run had ({sorted(str(d) for d in declared)}) — they cannot be from "
            "one run. Delete the partial directory and re-run `bf compute`."
        )
    num_jobs = int(declared.pop())
    ranks = sorted(
        int(rank) for _, meta in stamps
        if (rank := _get(meta, JOB_RANK_KEY)) is not None
    )
    missing = sorted(set(range(num_jobs)) - set(ranks))
    if missing or len(ranks) != len(stamps) or len(set(ranks)) != len(ranks):
        raise RuntimeError(
            f"search={spec.name!r}: the run declared {num_jobs} ranks but this "
            f"directory holds {len(stamps)} partial(s) covering ranks {ranks}"
            + (f", missing {missing}" if missing else "")
            + ". Each missing rank's slice of the corpus is simply absent from "
            "the merged top-K, silently lowering every recall number computed "
            "against it. Re-run `bf compute --num-jobs "
            f"{num_jobs} --job-rank R` for the missing rank(s) before merging."
        )
    return run_sha


def _reduce(
    cfg: BruteForceConfig, spec: SearchSpec, out: Store, partials: list[ParquetFile],
    readers: list[pq.ParquetFile],
) -> dict:
    """Reduce one search's partials into its final parquet.

    Returns the run-manifest row for this search — the output path plus what
    the reduce actually saw (partial count, queries, dtypes carried off the
    partials, how many queries ended short of k). `run_merge` turns those into
    its `{name: path}` return value and into the manifest.
    """
    k = spec.k
    # BEFORE anything expensive: these partials must be one run's, complete.
    run_sha = _validate_one_run(cfg, spec, partials, readers)
    n_rows = readers[0].metadata.num_rows
    for f, r in zip(partials, readers):
        if r.metadata.num_rows != n_rows:
            raise RuntimeError(
                f"partial {f.read_path} has {r.metadata.num_rows} rows but the first "
                f"partial has {n_rows}; partials must be row-aligned by query "
                "(same queries, same order). A truncated/mismatched partial can't be merged."
            )

    # Storage dtypes come FROM a partial rather than being re-derived: merge
    # never opens the corpus, and the files being merged are the only thing
    # that describes what actually produced these rows.
    carried = readers[0].schema_arrow.metadata or {}
    carried_dtypes = {
        key: carried[f"nova_bf.{key}".encode()].decode()
        for key in ("corpus_dtype", "queries_dtype")
        if f"nova_bf.{key}".encode() in carried
    }

    # Which tie-break rule produced these partials, and does it match the config
    # merge was handed? Editing `params.tiebreak` between compute and merge would
    # otherwise reduce ties by a rule the partials were never built for.
    for f, r in zip(partials, readers):
        stamped = (r.schema_arrow.metadata or {}).get(TIEBREAK_KEY)
        stamped = stamped.decode() if stamped is not None else None
        if stamped is not None and stamped != cfg.params.tiebreak:
            raise RuntimeError(
                f"partial {f.read_path} was computed with params.tiebreak="
                f"{stamped!r}, but this merge was given {cfg.params.tiebreak!r}. "
                "Ties would be reduced by a rule the partials were not built for. "
                "Re-run `bf compute`, or merge with the config that produced them."
            )

    # `hit_tie` rides along only when `merge` cannot apply the rule from
    # `hit_ids` alone — see `results.build_result_table`. Every partial must
    # agree: half of them carrying it would mean two different rules.
    has_tie = ["hit_tie" in r.schema_arrow.names for r in readers]
    if any(has_tie) and not all(has_tie):
        missing = [f.read_path for f, h in zip(partials, has_tie) if not h]
        raise RuntimeError(
            "some partials carry a hit_tie ordinate and others do not "
            f"(missing from {missing[:3]}); they cannot have come from one run. "
            "Re-run `bf compute` for this search."
        )
    want_tie = all(has_tie)

    payload_cols = [c for c in readers[0].schema_arrow.names if c not in RESERVED]
    batch_rows = _resolve_batch_rows(cfg.params.merge_batch_size, n_rows, len(partials), k)
    logger.info(
        "search=%r: merging %d partials (%d queries, k=%d) in batches of %d",
        spec.name, len(partials), n_rows, k, batch_rows,
    )

    # Read query_id + payload only from partial 0; every partial contributes its
    # hit lists (and query_id, for a cheap per-batch alignment check).
    base_iter = readers[0].iter_batches(batch_size=batch_rows)
    hit_cols = ["query_id", "hit_ids", "hit_scores"] + (["hit_tie"] if want_tie else [])
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
                tie_lists = (
                    [base.column("hit_tie")] + [o.column("hit_tie") for o in rest]
                    if want_tie else None
                )
                ids_arr, scores_arr = _topk_merge(score_lists, id_lists, tie_lists, k)

                lengths = ids_arr.value_lengths().to_numpy(zero_copy_only=False)
                short_count += int((lengths < k).sum())

                cols = {"query_id": base.column("query_id")}
                for c in payload_cols:
                    cols[c] = base.column(c)
                cols["hit_ids"] = ids_arr
                cols["hit_scores"] = scores_arr
                table = pa.table(cols)
                # The merged file is the artifact people actually consume, so
                # it carries the same provenance the partials do. The storage
                # dtypes are read back FROM a partial rather than re-derived:
                # merge never opens the corpus, and taking them from the files
                # being merged is also the only reading that describes what
                # actually produced these rows.
                table = table.replace_schema_metadata(
                    # `run_sha` is the partials' own fingerprint, carried
                    # rather than recomputed: merge never lists the corpus, and
                    # the run that matters is the one these rows came from.
                    provenance(cfg, spec, carried_dtypes, run_sha=run_sha,
                               reducing=True)
                )

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
    entry = run_manifest.search_entry(spec)
    entry.update({
        "queries": n_rows,
        "output_file": result_name(cfg, spec),
        "output_path": path,
        "partials": len(partials),
        "partial_dir": partial_dir(cfg, spec),
        "merge_batch_rows": batch_rows,
        "hit_tie_column": want_tie,
        "queries_short_of_k": short_count,
        "corpus_dtype": carried_dtypes.get("corpus_dtype"),
        "queries_dtype": carried_dtypes.get("queries_dtype"),
    })
    return entry
