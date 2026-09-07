"""Inter-worker reduce over per-rank top-K partials.

Each partial contains the top-K for every query over one worker's disjoint
corpus slice. Partials are row-aligned by query, so merge streams them into a
running per-query top-K rather than performing a keyed group-by; `_fold`
verifies the alignment.

The reduce is partial-major: only the running state plus a byte-budgeted window
of partials is resident at once. This avoids batch-major reads, where Parquet
may materialize an entire row group from every open partial regardless of the
requested batch size.

`_topk_merge` is order-independent under the shared `(score, tiebreak)` ranking,
so partials may be folded as they arrive. Payload shared by all partials is
taken from the first one.
"""


from __future__ import annotations

import logging
import os
import re
import time
import traceback

from queue import Empty, Queue
from threading import Event, Semaphore, Thread
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

from nova_bf import manifest as run_manifest
from nova_bf.config import BruteForceConfig, SearchSpec
from nova_bf.io import ParquetFile, Store
from nova_bf.tiebreak import SENTINEL_KEY, build_ordinals
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

# Bound in-flight partial memory as a fraction of currently available RAM.
_MERGE_WINDOW_FRACTION = 0.35
_MERGE_WINDOW_MIN = 2
_MERGE_WINDOW_MAX = 16
_MERGE_WINDOW_FALLBACK_BYTES = 8 << 30

# Estimate parsed Arrow size from Parquet metadata; dictionary-encoded columns
# expand more when materialized.
_PARSE_EXPANSION_DICT = 3.5
_PARSE_EXPANSION_PLAIN = 1.3


def _hit_bytes_per_partial(readers: list[pq.ParquetFile], hit_cols: list[str],
                           ranged: bool = False) -> int:
    """Uncompressed bytes ONE partial's hit columns occupy once parsed."""
    worst = 0
    for r in readers:
        md = r.metadata
        hit = raw = 0
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            for c in range(rg.num_columns):
                col = rg.column(c)
                raw += col.total_compressed_size          # ~ the file on disk
                # nested columns appear as leaf paths ("hit_ids.list.element")
                if col.path_in_schema.split(".")[0] in hit_cols:
                    enc = getattr(col, "encodings", None)
                    dictish = (True if enc is None
                               else any("DICT" in str(e).upper() for e in enc))
                    by_encoding = col.total_uncompressed_size * (
                        _PARSE_EXPANSION_DICT if dictish else _PARSE_EXPANSION_PLAIN)
                    
                    # Use the larger metadata- or statistics-based estimate.
                    hit += max(by_encoding, _string_parsed_bytes(col) or 0)
        n = int(hit)
        if ranged:
            # Ranged reads also keep the full encoded file in memory.
            n += raw
        worst = max(worst, n)
    return worst


def _string_parsed_bytes(col) -> int | None:
    """Estimate parsed bytes for a fixed-width string column from statistics."""
    try:
        st = col.statistics
        if st is None or not st.has_min_max:
            return None
        lo, hi = st.min, st.max
        if not isinstance(hi, (str, bytes)) or not isinstance(lo, (str, bytes)):
            return None
        # Only fixed-width strings can be estimated safely from min/max.
        if len(lo) != len(hi):
            return None
        # large_string = character bytes + one int64 offset per value
        return int(col.num_values) * (len(hi) + 8)
    except Exception:
        # Statistics are optional; fall back to the encoding-based estimate.
        return None


def _merge_window(readers: list[pq.ParquetFile], hit_cols: list[str],
                  n_partials: int, ranged: bool = False) -> int:
    """How many partials may be in flight, from a byte budget."""
    avail = _MERGE_WINDOW_FALLBACK_BYTES
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    per = _hit_bytes_per_partial(readers, hit_cols, ranged)
    if per <= 0:
        logger.warning(
            "merge window: parquet metadata reported no bytes for %s — cannot "
            "size the window; using the %d-partial floor blind.",
            hit_cols, _MERGE_WINDOW_MIN,
        )
        return min(_MERGE_WINDOW_MIN, n_partials)
    budget = int(avail * _MERGE_WINDOW_FRACTION)
    fits = budget // per
    # Prefer the overlap floor when affordable, but never exceed the budget cap.
    n = min(max(fits, 1 if fits < _MERGE_WINDOW_MIN else _MERGE_WINDOW_MIN),
            _MERGE_WINDOW_MAX, n_partials)
    if fits < _MERGE_WINDOW_MIN:
        logger.warning(
            "merge window: one partial (%.2f GiB of hit columns) exceeds the "
            "%.1f GiB budget; dropping to %d reader(s) — no read/fold overlap, "
            "but a deeper window would only multiply the overshoot. Merge on a "
            "larger box if this still OOMs.",
            per / 2**30, budget / 2**30, n,
        )
    logger.info(
        "merge window: %d partial(s) in flight (%.2f GiB each x %d = %.1f GiB "
        "against %.1f GiB available)",
        n, per / 2**30, n, n * per / 2**30, avail / 2**30,
    )
    return n


_TARGET_CANDIDATE_SLOTS = 20_000_000


def _resolve_batch_rows(explicit: int | None, n_rows: int, n_partials: int, k: int) -> int:
    """Choose merge batch rows from the candidate-memory target.

    Each fold holds `B x 2k` candidates (state + one partial), independent of
    the total partial count. Explicit batch sizes are honored with a warning
    when they exceed the automatic target.
    """
    per_row = max(1, 2 * k)
    ceiling = max(1, min(_TARGET_CANDIDATE_SLOTS // per_row, n_rows))
    if explicit is None:
        return ceiling
    want = max(1, min(explicit, n_rows))
    if want > ceiling:
        logger.warning(
            "params.merge_batch_size=%d holds %.1f M candidate slots per batch "
            "(2 x k=%d per row; %d partials are folded one at a time), above the "
            "~%.1f M auto target — honoring it, since you set it. Drop the "
            "setting to let merge size itself (%d rows) if this OOMs.",
            explicit, want * per_row / 1e6, k, n_partials,
            _TARGET_CANDIDATE_SLOTS / 1e6, ceiling,
        )
    return want




def _id_tie_grid(scatter: list, rows: np.ndarray, b: int, width: int) -> np.ndarray:
    """Lexicographic ID ranks for selected candidate rows.

    Ranks all partials together with `build_ordinals`, matching compute-time
    tie-breaking and making ranks comparable across shards.
    """
    dest = np.full(b, -1, dtype=np.int64)
    dest[rows] = np.arange(len(rows))
    subs, places = [], []
    for row_idx, col, flat_ids in scatter:
        r = dest[row_idx]
        sel = np.flatnonzero(r >= 0)
        if not len(sel):
            continue
        # Avoid copying when every ID in the partial is selected
        subs.append(flat_ids if len(sel) == len(flat_ids)
                    else flat_ids.take(pa.array(sel, pa.int64())))
        places.append((r[sel], col[sel]))

    grid = np.full((len(rows), width), np.iinfo(np.int64).max, dtype=np.int64)
    if not subs:
        return grid
    try:
        ords = build_ordinals(subs)
    except ValueError as exc:
        # Null IDs cannot participate in deterministic ID tie-breaking.
        raise ValueError(
            f"a partial has null hit_ids, which have no ordering position and "
            f"cannot break a tie ({exc}). Re-run `bf compute` for this search "
            f"with an id column that has no nulls."
        ) from exc
    for (r, c), o in zip(places, ords):
        grid[r, c] = o.astype(np.int64)
    return grid

# Optional override for the merge fold backend for testing purposes;
# unset prefers CUDA when available.
_FOLD_ENV = "NOVA_BF_MERGE_FOLD"

# Which fold(s) actually executed the reduce for the manifest.
_FOLD_USED: set[str] = set()


def _reset_fold_used() -> None:
    _FOLD_USED.clear()


def _lane_rankable(scatter: list) -> bool:
    """Whether candidate IDs can use the fixed-width lane ranking path.

    Packed-key folding is worthwhile only when IDs can be ranked as uint64
    lanes; otherwise ranking falls back to the slower CPU string sort.
    """
    from nova_bf.tiebreak import _fixed_width

    return bool(scatter) and _fixed_width([a for _, _, a in scatter]) is not None


def _fold_device(forced_only: bool = False):
    """Return the requested/available fold device, or None for NumPy.

    `forced_only` reports only explicit packed-fold requests.
    """
    want = os.environ.get(_FOLD_ENV, "").strip().lower()
    if forced_only:
        return want not in ("", "numpy", "off", "0")
    if want in ("numpy", "off", "0"):
        return None
    try:
        import torch
    except Exception as exc:
        # Not just ImportError: a broken CUDA runtime raises OSError out of the
        # import itself.
        if want:
            raise RuntimeError(f"{_FOLD_ENV}={want!r} but torch is unusable: {exc}")
        return None
    if want in ("torch", "cpu"):
        return torch.device("cpu")
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{_FOLD_ENV}='cuda' but no CUDA device is available")
        return torch.device("cuda")
    if want:
        raise ValueError(f"{_FOLD_ENV}={want!r}; expected cuda, torch, cpu or numpy")
    dev = os.environ.get("NOVA_BF_DEVICE", "").strip().lower()
    if dev and dev != "cuda":
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def _fold_torch(scores: np.ndarray, tie: np.ndarray, tie_is_rank: bool,
                k: int, device) -> np.ndarray:
    """Return top-k column indices using compute's packed-key fold.

    Packed keys order by score, then lower tie rank, so no separate tie-repair
    pass is needed.
    """
    import torch

    from nova_bf.compute import _merge_topk
    from nova_bf.tiebreak import pack

    b, width = scores.shape
    s = torch.from_numpy(np.ascontiguousarray(scores)).to(device)
    o = torch.from_numpy(np.ascontiguousarray(tie)).to(device)
    # Compress arbitrary int64 tie keys to monotone 32-bit ranks for packing.
    if not tie_is_rank:
        o = _rank_dense(o)
    
    # Padding scores already lose; give them any packable ordinal.
    o = torch.where(s == float("-inf"), torch.zeros_like(o), o)
    key = pack(s, o)
    
    # Match NumPy semantics by forcing NaNs below all valid candidates.
    if torch.isnan(s).any():
        key = torch.where(torch.isnan(s),
                          torch.full_like(key, SENTINEL_KEY), key)
    enc = (torch.arange(width, device=device, dtype=torch.int64)
           .expand(b, width).contiguous())

    top_key, top_enc = key[:, :k].contiguous(), enc[:, :k].contiguous()
    if width > k:
        top_key, top_enc = _merge_topk(
            top_key, top_enc,
            [(key[:, k:].contiguous(), enc[:, k:].contiguous(), None)], k)
    # Sort survivors into final score/tiebreak order.
    order = torch.argsort(top_key, dim=1, descending=True, stable=True)
    return top_enc.gather(1, order).cpu().numpy()


def _rank_dense(t):
    """Convert tie values to dense 0-based ranks while preserving order.

    This compresses int64 tie keys into the 32-bit field used by `pack`.
    """
    import torch

    flat = t.reshape(-1)
    n = flat.numel()
    if n > 0xFFFFFFFF:
        raise ValueError(
            f"{n:,} candidates in one merge batch overflows the 32-bit tie-break "
            "field; lower params.merge_batch_size."
        )
    order = torch.argsort(flat, stable=True)
    rank = torch.empty_like(flat)
    rank[order] = torch.arange(n, device=flat.device, dtype=flat.dtype)
    return rank.reshape(t.shape)


def _ambiguous_rows(scores: np.ndarray, top_s: np.ndarray) -> np.ndarray:
    """Return rows whose score-only top-k result does not uniquely determine the hits.

    A row is ambiguous if either:
      * two selected hits have the same score, so their relative order requires
        a tiebreak; or
      * the cutoff score is shared by both selected and unselected candidates,
        so the top-k membership requires a tiebreak.

    `-inf` entries are padding for missing candidates and are ignored.
    """
    real = top_s > -np.inf
    amb = ((top_s[:, 1:] == top_s[:, :-1]) & real[:, :-1]).any(axis=1)

    tau = top_s[:, -1]
    fin_tau = tau > -np.inf
    if fin_tau.any():
        n_all = (scores == tau[:, None]).sum(axis=1)
        n_kept = (top_s == tau[:, None]).sum(axis=1)
        amb |= fin_tau & (n_all > n_kept)
    return amb


def _take_ids(scatter: list, sel: np.ndarray) -> pa.Array:
    """Gather selected IDs without concatenating/copying all partial buffers."""
    chunks = [a for _, _, a in scatter]
    if not chunks:
        return pa.array([], pa.large_string())
    values = chunks[0] if len(chunks) == 1 else pa.chunked_array(chunks)
    taken = values.take(pa.array(sel, pa.int64()))
    
    # Keep a stable large_string output schema across all batches.
    if taken.type != pa.large_string():
        taken = taken.cast(pa.large_string())
    if isinstance(taken, pa.ChunkedArray):
        # Normalize to the plain Array required by `ListArray.from_arrays`.
        taken = taken.combine_chunks()
    if isinstance(taken, pa.ChunkedArray):
        taken = (taken.chunk(0) if taken.num_chunks
                 else pa.array([], pa.large_string()))
    return taken


def _topk_numpy(scores, ties, scatter, want_tie, b, width, kk):
    """Portable top-k fold: fast score cut, then exact tie repair."""
    if kk < width:
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
    else:
        part = np.broadcast_to(np.arange(width), (b, width)).copy()

    # Sort the partition survivors by descending score.
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_s = np.take_along_axis(scores, top_idx, axis=1)

    # Re-rank only rows where the score-only cut crossed a tie.
    amb = _ambiguous_rows(scores, top_s)
    if amb.any():
        rows = np.flatnonzero(amb)
        tie = ties[rows] if want_tie else _id_tie_grid(scatter, rows, b, width)
        
        # Score descending, then tiebreak ascending.
        exact = np.lexsort((tie, -scores[rows]), axis=1)[:, :kk]
        top_idx[rows] = exact
        top_s[rows] = np.take_along_axis(scores[rows], exact, axis=1)
    return top_idx, top_s


def _topk_merge(
    score_lists: list[pa.ListArray],
    id_lists: list[pa.ListArray],
    tie_lists: list[pa.ListArray] | None,
    k: int,
) -> tuple[pa.ListArray, pa.ListArray, pa.ListArray | None]:
    """Merge row-aligned per-partial top-K lists into one global top-K.

    Ties are resolved by the SAME rule each worker applied within itself, so the
    result does not depend on how many workers produced it. 
    """
    n_partials = len(score_lists)
    b = len(score_lists[0])
    width = n_partials * k
    scores = np.full((b, width), -np.inf, dtype=np.float32)

    # Store indices into Arrow ID buffers instead of materializing Python strings.
    src = np.full((b, width), -1, dtype=np.int64)

    want_tie = tie_lists is not None
    ties = np.full((b, width), np.iinfo(np.int64).max, dtype=np.int64) if want_tie else None
    
    # Tracks where each partial's IDs land in the candidate grid.
    scatter: list = []
    base = 0

    for w, (sl, il) in enumerate(zip(score_lists, id_lists)):
        if sl.type.value_type != pa.float32():
            raise RuntimeError(
                f"partial {w}'s hit_scores are {sl.type.value_type}, not float32; "
                "merging would round every score into the float32 grid and change "
                "which candidates tie. Re-run `bf compute` for this search."
            )
        if sl.null_count:
            raise RuntimeError(
                f"partial {w}'s hit_scores column has {sl.null_count} null "
                "row(s); every query must carry a list of hits, empty if it "
                "matched nothing. Re-run `bf compute` for this search."
            )
        lengths = sl.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            continue
        # Guard against out-of-index access
        if lengths.max() > k:
            raise RuntimeError(
                f"partial {w} has a query with {int(lengths.max())} hits but k={k}; "
                "a partial must never hold more than k candidates per query. "
                "Re-run `bf compute` for this search."
            )
        # Reject null hit values before conversion: null scores become NaN, null ties 
        # can become INT64_MIN, and null IDs would propagate into the result.
        flat_s_arr, flat_ids = sl.flatten(), il.flatten()
        for col_name, child in (("hit_scores", flat_s_arr), ("hit_ids", flat_ids)):
            if child.null_count:
                raise RuntimeError(
                    f"partial {w}'s {col_name} has {child.null_count} null "
                    "value(s) inside its lists; every hit must carry a real "
                    f"{col_name[4:]}. Re-run `bf compute` for this search."
                )
        flat_s = flat_s_arr.to_numpy(zero_copy_only=False)
        
        # Scores and IDs must have identical row structure.
        id_lengths = il.value_lengths().to_numpy(zero_copy_only=False)
        if not np.array_equal(id_lengths.astype(np.int64), lengths):
            raise RuntimeError(
                f"partial {w}'s hit_ids rows are split differently from its "
                "hit_scores rows; the columns must line up row for row or every "
                "hit would be reported under the wrong id. Re-run `bf compute` "
                "for this search."
            )
        row_idx = np.repeat(np.arange(b), lengths)
        starts = np.zeros(b, dtype=np.int64)
        np.cumsum(lengths[:-1], out=starts[1:])
        within = np.arange(total) - np.repeat(starts, lengths)  # position within each row
        col = w * k + within
        scores[row_idx, col] = flat_s
        # `flat_ids` is in scatter order, so slot j of this partial is element
        # `base + j` of the partials concatenated end to end.
        src[row_idx, col] = base + np.arange(total, dtype=np.int64)
        base += total
        scatter.append((row_idx, col, flat_ids))
        if want_tie:
            tl = tie_lists[w]
            tie_lengths = tl.value_lengths().to_numpy(zero_copy_only=False)
            # Tie values must align with the same candidates.
            if not np.array_equal(tie_lengths.astype(np.int64), lengths):
                raise RuntimeError(
                    f"partial {w}'s hit_tie rows are split differently from its "
                    "hit_scores rows; the columns must line up row for row or ties "
                    "would be broken against the wrong hits. Re-run `bf compute` "
                    "for this search."
                )
            tl_flat = tl.flatten()
            if tl_flat.null_count:
                raise RuntimeError(
                    f"partial {w}'s hit_tie has {tl_flat.null_count} null "
                    "value(s) inside its lists; a null has no ordering position "
                    "and would outrank every real hit. Re-run `bf compute` for "
                    "this search."
                )
            ties[row_idx, col] = tl_flat.to_numpy(zero_copy_only=False)

    kk = min(k, width)
    device = _fold_device()
    if (device is not None and not want_tie and not _fold_device(forced_only=True)
            and not _lane_rankable(scatter)):
        # Avoid ranking every variable-width ID unless explicitly requested.
        device = None
    _FOLD_USED.add("numpy" if device is None else f"torch:{device.type}")
    if device is not None:
        tie = ties if want_tie else _id_tie_grid(scatter, np.arange(b), b, width)
        top_idx = _fold_torch(scores, tie, not want_tie, kk, device)
        top_s = np.take_along_axis(scores, top_idx, axis=1)
    else:
        top_idx, top_s = _topk_numpy(scores, ties, scatter, want_tie, b, width, kk)


    # Drop -inf padding; +inf remains a valid hit.
    valid = top_s > -np.inf
    
    # ListArray offsets are int32, so guard against silent overflow.
    counts = valid.sum(axis=1).astype(np.int64)
    total_hits = int(counts.sum())
    if total_hits > np.iinfo(np.int32).max:
        raise ValueError(
            f"{total_hits:,} hits in one merge batch overflows the int32 "
            f"ListArray offsets (limit {np.iinfo(np.int32).max:,})."
        )
    counts = counts.astype(np.int32)
    offsets = np.empty(b + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    off = pa.array(offsets, pa.int32())
    scores_arr = pa.ListArray.from_arrays(off, pa.array(top_s[valid], pa.float32()))
    
    # Gather only winning IDs from the original Arrow buffers.
    sel = np.take_along_axis(src, top_idx, axis=1)[valid]
    ids_arr = pa.ListArray.from_arrays(off, _take_ids(scatter, sel))
    
    ties_arr = None
    if want_tie:
        # Keep ties aligned with the same winners as scores and IDs.
        top_tie = np.take_along_axis(ties, top_idx, axis=1)
        ties_arr = pa.ListArray.from_arrays(
            off, pa.array(top_tie[valid], pa.int64()))
    return ids_arr, scores_arr, ties_arr

def run_merge(cfg: BruteForceConfig) -> dict[str, str]:
    """Merge each search's per-rank partials into its final Parquet output.

    `merge_ranged_reads` enables concurrent ranged fetches into memory; nothing
    is staged to local disk.
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

    # All searches from one compute run must have the same rank count.
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

    # Run fingerprints are search-specific, so they cannot prove cross-search
    # consistency. Record the run-global tie-break rule to catch incompatible partials.
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    
    # These readers provide schema/metadata only; `_reduce` streams data through
    # its own bounded read window.
    readers_by_name: dict[str, list[pq.ParquetFile]] = {
        spec.name: [pq.ParquetFile(f.read_path, filesystem=out.fs)
                    for f in partials_by_name[spec.name]]
        for spec in cfg.searches
    }
    # Search fingerprints are not comparable across searches, so use the stamped
    # run-global tie-break rule to reject incompatible partials.
    rules = {
        name: {(r.schema_arrow.metadata or {}).get(TIEBREAK_KEY) for r in readers}
        for name, readers in readers_by_name.items()
    }
    seen = {v for vs in rules.values() for v in vs if v is not None}
    if len(seen) > 1:
        pretty = {n: sorted(x.decode() for x in v if x) for n, v in rules.items()}
        raise RuntimeError(
            f"partials were computed under different tie-break rules: {pretty} — "
            "merging them puts hits decided by different rules in one artifact. "
            "Re-run `bf compute` so every search uses one `params.tiebreak`."
        )

    entries = [
        _reduce(cfg, spec, out, partials_by_name[spec.name], readers_by_name[spec.name])
        for spec in cfg.searches
    ]

    # Record the completed reduce and the number of partials actually folded.
    doc = run_manifest.base_manifest(cfg, "merge")
    doc.update({
        "started_at": started_at.isoformat(),
        "searches": entries,
        "counts": {
            "partials_merged": sum(e["partials"] for e in entries),
            "queries": max((e["queries"] for e in entries), default=0),
        },
        "timing": {"elapsed_seconds": round(time.perf_counter() - t0, 2)},
        "output_files": [e["output_file"] for e in entries],
    })
    run_manifest.write(out, run_manifest.manifest_name(cfg, "merge"), doc)
    return {e["name"]: e["output_path"] for e in entries}


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
        # Older partials may lack fingerprints; warn but continue with the
        # independent config/rank validation below.
        logger.warning(
            "search=%r: none of the %d partials carry a run fingerprint (they "
            "predate it) — cannot verify they came from a single run. Re-run "
            "`bf compute` if this directory may hold partials from more than one.",
            spec.name, len(partials),
        )
        run_sha = None
    elif (unstamped := sorted(p for p, sha in runs.items() if sha is None)) or len(present) > 1:
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
    else:
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
    ranks = []
    for f, meta in stamps:
        rank = _get(meta, JOB_RANK_KEY)
        if rank is None:
            continue
        try:
            r = int(rank)
        except ValueError:
            raise RuntimeError(
                f"search={spec.name!r}: partial {f.read_path} declares job_rank="
                f"{rank!r}, which is not an integer. Its metadata is corrupt; "
                "re-run `bf compute` for that rank."
            ) from None
        # Filename and metadata must identify the same rank.
        stem = f.read_path.rsplit("/", 1)[-1]
        if (m := re.fullmatch(r"rank(\d+)\.parquet", stem)) and int(m.group(1)) != r:
            raise RuntimeError(
                f"search={spec.name!r}: {stem} declares job_rank={r} — the file "
                "name and its metadata disagree, so one of them was rewritten "
                "and the rank set cannot be trusted. Delete the directory and "
                "re-run `bf compute` for this search."
            )
        ranks.append(r)
    ranks = sorted(ranks)
    missing = sorted(set(range(num_jobs)) - set(ranks))

    # Require exactly ranks 0..num_jobs-1; reject both missing and extra ranks.
    extra = sorted(set(ranks) - set(range(num_jobs)))
    if (missing or extra or len(ranks) != len(stamps)
            or len(set(ranks)) != len(ranks)):
        raise RuntimeError(
            f"search={spec.name!r}: the run declared {num_jobs} ranks but this "
            f"directory holds {len(stamps)} partial(s) covering ranks {ranks}"
            + (f", missing {missing}" if missing else "")
            + (f", and {extra} outside 0..{num_jobs - 1}" if extra else "")
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
    """Reduce one search's partials into its final Parquet and manifest entry."""
    k = spec.k
    _reset_fold_used()
    
    # Validate that all partials belong to the same complete compute run.
    run_sha = _validate_one_run(cfg, spec, partials, readers)
    n_rows = readers[0].metadata.num_rows
    for f, r in zip(partials, readers):
        if r.metadata.num_rows != n_rows:
            raise RuntimeError(
                f"partial {f.read_path} has {r.metadata.num_rows} rows but the first "
                f"partial has {n_rows}; partials must be row-aligned by query "
                "(same queries, same order). A truncated/mismatched partial can't be merged."
            )
    
    # Preserve the dtypes recorded by the compute partials.
    carried = readers[0].schema_arrow.metadata or {}
    carried_dtypes = {
        key: carried[f"nova_bf.{key}".encode()].decode()
        for key in ("corpus_dtype", "queries_dtype")
        if f"nova_bf.{key}".encode() in carried
    }

     # Compute and merge must use the same tie-break rule.
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

    # All partials must agree on whether an explicit tie ordinate is carried.
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
    # Prevent 0 row query file
    if n_rows == 0:
        raise RuntimeError(
            f"search={spec.name!r}: the partials under "
            f"{partial_dir(cfg, spec)}/ hold 0 queries, so there is nothing to "
            "merge. Re-run `bf compute` for this search."
        )
    batch_rows = _resolve_batch_rows(cfg.params.merge_batch_size, n_rows, len(partials), k)
    logger.info(
        "search=%r: merging %d partials (%d queries, k=%d) in batches of %d",
        spec.name, len(partials), n_rows, k, batch_rows,
    )

    # Partial-major reduce: fold a bounded window of partials into running
    # per-query top-K state, avoiding one open Parquet row group per worker
    hit_cols = ["hit_ids", "hit_scores"] + (["hit_tie"] if want_tie else [])
    n_batches = (n_rows + batch_rows - 1) // batch_rows

    # Running top-K state, bounded by n_rows * k candidates.
    state: list[tuple | None] = [None] * n_batches
    
    # Payload/query columns are identical across partials; retain them once.
    head: list[pa.Table | None] = [None] * n_batches
    
     # Reference query IDs used to verify row alignment as partials arrive.
    qref: list[pa.Array | None] = [None] * n_batches

    def _col(sl: pa.Table, name: str):
        """Return `name` as one contiguous Arrow Array."""
        ca = sl.column(name).combine_chunks()

        # `combine_chunks()` may still return a one-chunk ChunkedArray.
        return ca.chunk(0) if isinstance(ca, pa.ChunkedArray) else ca

    def _fold(idx: int, sl: pa.Table, keep_head: bool) -> None:
        if keep_head and head[idx] is None:
            # Copy retained query/payload columns so this slice does not pin the 
            # partial's full backing table.
            head[idx] = pa.table(
                {c: _col(sl, c) for c in ["query_id", *payload_cols]}
            )
        
        # Verify every partial has the same queries in the same row order.
        qid = _col(sl, "query_id")
        if qref[idx] is None:
            qref[idx] = qid
        elif not qref[idx].equals(qid):
            raise RuntimeError(
                "partials are not row-aligned: a batch's query_id column differs "
                "across partials. Re-run `bf compute` so every rank writes the "
                "same queries in the same order."
            )
        cur = state[idx]
        sc, ids = _col(sl, "hit_scores"), _col(sl, "hit_ids")
        ti = _col(sl, "hit_tie") if want_tie else None
        if cur is None:
            # Seed through the same fold so single- and multi-partial merges have 
            # identical normalization and tie semantics.
            ids0, sc0, ti0 = _topk_merge([sc], [ids], [ti] if want_tie else None, k)
            state[idx] = (ids0, sc0, ti0)
            return
        c_ids, c_sc, c_ti = cur
        ids2, sc2, ti2 = _topk_merge(
            [c_sc, sc], [c_ids, ids],
            [c_ti, ti] if want_tie else None, k,
        )
        state[idx] = (ids2, sc2, ti2)

    # Bound concurrent partial reads by the merge memory budget.
    ranged = bool(cfg.params.merge_ranged_reads)
    window_n = _merge_window(readers, hit_cols, len(partials), ranged)
    # Preserve the original URI scheme and divide ranged-read concurrency across 
    # all in-flight partials.
    src = Store(out.uri, ranged_get=ranged,
                ranged_get_concurrency=max(1, 24 // max(1, window_n)))
    q: Queue = Queue(maxsize=window_n)
    window = Semaphore(window_n)

    # Keep read and fold failures separate so data errors are not masked by I/O errors.
    errors: list[BaseException] = []
    fold_errors: list[BaseException] = []
    
    # Stop readers from starting unnecessary work after the reduce has failed.
    abort = Event()

    def _read(i: int, f: ParquetFile) -> None:
        try:
            window.acquire()
            if abort.is_set():
                # Preserve one queue item per reader without starting another read.
                q.put((i, None))
                return
            # Every partial supplies query IDs for alignment; payload comes from
            # partial 0 only.
            cols = hit_cols + ["query_id"] + (payload_cols if i == 0 else [])
            q.put((i, src.read_columns(f.read_path, cols)))
        except BaseException as exc:            # noqa: BLE001 - re-raised below
            errors.append(exc)
            q.put((i, None))

    # Prepare anything that can raise before starting reader threads; once readers 
    # exist, every started reader must be drained to release its window permit.
    short_count = 0
    path = f"{out.root.rstrip('/')}/{result_name(cfg, spec)}"
    if not out.is_s3:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    bar = tqdm(total=len(partials), unit="partial", desc=f"merge {spec.name}",
               dynamic_ncols=True)
    threads = [Thread(target=_read, args=(i, f), daemon=True)
               for i, f in enumerate(partials)]
    started = 0
    drained = 0
    failed = False
    try:
        # Track successful starts so only live readers are drained/joined.
        for t in threads:
            t.start()
            started += 1
        # Drain every started reader even after failure; otherwise readers can 
        # remain blocked on the queue or semaphore while holding partial buffers.
        for _ in range(started):
            i, tbl = q.get()
            drained += 1
            if tbl is None:                 # this reader failed or stood down
                failed = True
                abort.set()
                window.release()            # hand back ITS permit
                continue
            if failed:                      # already doomed: drop, keep draining
                del tbl
                window.release()
                continue
            try:
                for bi in range(n_batches):
                    sl = tbl.slice(bi * batch_rows, batch_rows)
                    if sl.num_rows:
                        _fold(bi, sl, keep_head=(i == 0))
            except BaseException as exc:     # noqa: BLE001 - re-raised below
                errors.append(exc)
                fold_errors.append(exc)
                failed = True
                abort.set()
                # Preserve traceback text, then release frame-held Arrow buffers.
                exc.add_note("".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__)).rstrip())
                exc.__traceback__ = None
                sl = None

            # `sl` is a view into `tbl`; release it before returning the window permit.
            sl = None
            del tbl
            window.release()                # slide the window forward
            try:
                bar.update(1)
            except Exception:
                pass
    finally:
        # Drain any readers left behind by an unexpected consumer-side exit.
        if drained < started:
            abort.set()
            while drained < started:
                try:
                    _i, _tbl = q.get(timeout=30)
                except Empty:
                    break                       # reported by the join below
                drained += 1
                del _tbl
                window.release()
        try:
            bar.close()
        except Exception:                       # noqa: BLE001
            pass
        
        # Use one shared shutdown deadline and join only threads that actually started.
        end = time.monotonic() + 30
        for t in threads[:started]:
            t.join(timeout=max(0.0, end - time.monotonic()))
        stuck = [t.name for t in threads[:started] if t.is_alive()]
        if stuck:
            logger.warning(
                "search=%r: %d reader thread(s) still running after the 30s "
                "shutdown grace (%s); they hold their partial's buffers until "
                "the process exits", spec.name, len(stuck), ", ".join(stuck[:4]))
    if failed and not errors:
        # Never write a result after an incomplete reduce.
        raise RuntimeError(
            f"search={spec.name!r}: the reduce failed but recorded no error; "
            "refusing to write a result built from incomplete partials."
        )
    if errors:
        primary = fold_errors[0] if fold_errors else errors[0]
        for extra in errors:
            if extra is not primary:
                logger.error("search=%r: additional merge failure: %r",
                             spec.name, extra)
        raise primary

    # All partials are folded; write each query batch and release its state
    sink = out.fs.open_output_stream(path)
    writer: pq.ParquetWriter | None = None
    body_ok = False
    try:
        for bi in range(n_batches):
            ids_arr, scores_arr, _ = state[bi]
            base = head[bi]
            lengths = ids_arr.value_lengths().to_numpy(zero_copy_only=False)
            short_count += int((lengths < k).sum())
            cols = {"query_id": _col(base, "query_id")}
            for c in payload_cols:
                cols[c] = _col(base, c)
            cols["hit_ids"] = ids_arr
            cols["hit_scores"] = scores_arr
            table = pa.table(cols)
            # Preserve the provenance carried by the compute partials.
            table = table.replace_schema_metadata(
                provenance(cfg, spec, carried_dtypes, run_sha=run_sha,
                           num_jobs=len(partials), reducing=True)
            )
            if writer is None:
                writer = pq.ParquetWriter(sink, table.schema, compression="snappy")
            writer.write_table(table)
            state[bi] = head[bi] = qref[bi] = None   # release as we go
        body_ok = True
    finally:
        # Close both resources without masking an error already in flight.
        close_err: BaseException | None = None
        try:
            if writer is not None:
                writer.close()
        except BaseException as exc:                # noqa: BLE001
            close_err = exc
        try:
            sink.close()
        except BaseException as exc:                # noqa: BLE001
            close_err = close_err or exc

        # A successful body is not enough: writer.close() commits the Parquet footer.
        wrote = body_ok and close_err is None
        if not wrote:
            # Never leave a partial result under the canonical output name.
            try:
                out.fs.delete_file(path)
                logger.error("search=%r: merge failed; removed the incomplete %s",
                             spec.name, path)
            except BaseException as exc:            # noqa: BLE001
                logger.error(
                    "search=%r: merge failed AND the incomplete %s could not be "
                    "removed (%r) — DELETE IT BY HAND; it is a truncated result "
                    "under the name a finished one would have",
                    spec.name, path, exc)
            if close_err is not None:
                logger.error("search=%r: also failed to close the output: %r",
                             spec.name, close_err)
        # Closing/committing the output is part of a successful write.
        if body_ok and close_err is not None:
            raise close_err

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
        "tiebreak_source": "hit_tie" if want_tie else "hit_ids",
        "run_sha": run_sha,
        "merge_fold": sorted(_FOLD_USED),
        "queries_short_of_k": short_count,
        "corpus_dtype": carried_dtypes.get("corpus_dtype"),
        "queries_dtype": carried_dtypes.get("queries_dtype"),
    })
    return entry
