"""Merge per-rank top-K partials into global per-query top-K results.

Each partial covers a disjoint corpus slice and contains row-aligned results
for the same queries. Partials are folded incrementally into a running top-K,
with query IDs checked for alignment as they arrive.

The reduce is partial-major and bounds the number of partials in flight to
overlap I/O with folding while limiting memory use. `_topk_merge` uses the
shared `(score, tiebreak)` ordering, so fold order does not affect the result.
Query payload columns are retained from the first partial.
"""


from __future__ import annotations

import json
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
import pyarrow.fs as fs
import pyarrow.parquet as pq

from tqdm import tqdm

from nova_bf import manifest as run_manifest
from nova_bf import thresholds as _thresholds
from nova_bf.config import BruteForceConfig, SearchSpec
from nova_bf.io import ParquetFile, Store
from nova_bf.tiebreak import SENTINEL_KEY, build_ordinals, _is_oom
from nova_bf.results import (
    CONFIG_KEY,
    FORCED_KEY,
    JOB_RANK_KEY,
    NUM_JOBS_KEY,
    RESERVED,
    EXCHANGE_KEY,
    RESCORE_KEY,
    RUN_KEY,
    TIEBREAK_KEY,
    config_identity,
    merge_forced,
    partial_dir,
    provenance,
    result_name,
    warn_if_short,
)

logger = logging.getLogger(__name__)
def _unclamp_decode_threads(cfg: BruteForceConfig) -> None:
    """Undo a launcher's OMP_NUM_THREADS clamp on PyArrow's decode pool.

    Ray/SkyPilot may set OMP_NUM_THREADS to the task's requested CPUs, often 1
    for GPU-only tasks. PyArrow inherits this for its global CPU pool, making
    reduce decoding single-threaded even when more CPUs are available.

    `_usable_cpu_count()` respects affinity and cgroup quotas, so widening is
    limited to this process's actual CPU allocation.

    Without an explicit `cpu_thread_count`, only pools attributable to
    OMP_NUM_THREADS are widened; deliberate Arrow pool settings are preserved.
    """
    from nova_bf.compute import _usable_cpu_count  # deferred: import cycle

    want = cfg.params.cpu_thread_count
    if not want or want <= 0:
        usable = _usable_cpu_count()

        # Widen only when Arrow matches the launcher's OMP_NUM_THREADS clamp.
        # A different width is treated as an intentional operator setting.
        omp = os.environ.get("OMP_NUM_THREADS", "").strip()
        try:
            clamped = bool(omp) and int(omp) == pa.cpu_count() < usable
        except ValueError:
            # Nested or otherwise non-integer OMP values cannot be matched
            # reliably; preserve the current pool and make that visible.
            clamped = False
            if pa.cpu_count() < usable:
                logger.warning(
                    "OMP_NUM_THREADS=%r is not a plain integer, so the decode "
                    "pool cannot be matched against it; leaving it at %d of %d "
                    "usable CPUs. Set params.cpu_thread_count to widen it.",
                    omp, pa.cpu_count(), usable,
                )

        if not clamped:
            return
        want = usable

    if want == pa.cpu_count():
        return

    try:
        pa.set_cpu_count(want)
    except Exception as exc:  # noqa: BLE001 - never fail a merge
        logger.warning("could not set pyarrow's decode pool to %d: %s", want, exc)
        return

    logger.info(
        "pyarrow decode pool -> %d thread(s) for the reduce "
        "(OMP_NUM_THREADS=%s, usable CPUs=%d)",
        want,
        os.environ.get("OMP_NUM_THREADS", "<unset>"),
        _usable_cpu_count(),
    )

def _decide_lanes(tbl, spec) -> bool:
    """Choose the ID representation from an already-decoded partial."""
    from nova_bf.tiebreak import _NO_GPU_ORDINALS, _fixed_width

    try:
        # Bound each probe: a 100k-query partial may be a single Arrow chunk,
        # and `_fixed_width` uses `np.diff`, which allocates an int64 array.
        probe_rows = max(256, min(8192, 2_000_000 // max(1, spec.k)))
        min_len, n_seen, W = None, 0, None

        for col in _sliced(tbl.column("hit_ids"), probe_rows):
            if len(col) == 0:
                continue

            lens = col.value_lengths().to_numpy(zero_copy_only=False)
            n_seen += len(lens)
            bmin = int(lens.min())
            min_len = bmin if min_len is None else min(min_len, bmin)

            if min_len < spec.k:
                break  # Not dense; device fold cannot apply.

            # `_fixed_width` inspects offsets only; character data is untouched.
            bw = _fixed_width([col.flatten()])
            if bw is None or (W is not None and bw != W):
                W = None
                break
            W = bw

        # Device fold requires full-k rows and fixed-width IDs.
        dense = n_seen > 0 and min_len == spec.k
        if not dense:
            logger.info(
                "search=%r: partial rows are not all k=%d hits (min %d), so "
                "the device fold cannot apply; keeping ids on Arrow",
                spec.name,
                spec.k,
                min_len if min_len is not None else 0,
            )

        return (
            dense
            and W is not None
            and not os.environ.get(_NO_GPU_ORDINALS)
        )

    except Exception as exc:  # noqa: BLE001 - must never fail a merge
        # Arrow fallback is correct; log failures so fast-path regressions show.
        logger.warning(
            "search=%r: the id density check failed (%s: %s); keeping ids "
            "on Arrow. The merge is correct but the device fold is off.",
            spec.name,
            type(exc).__name__,
            exc,
        )
        return False


def _sliced(column, rows: int):
    """Yield zero-copy slices of at most `rows` across Arrow chunks."""
    for chunk in column.chunks:
        for off in range(0, len(chunk), rows):
            yield chunk.slice(off, rows)


# How many partials may be read into memory at once.
_MERGE_WINDOW_DEFAULT = 2
# Concurrent range GETs per in-flight partial are this divided by the window,
# which bounds each file's own window of ranges. They all run on io's one
# process-wide fetch pool (`io._fetch_pool`, default 512 threads), which caps
# the total.
_RANGED_GET_POOL = 24

def _merge_window(cfg, n_partials: int) -> int:
    """Return the number of partials to keep in flight.

    NOVA_BF_MERGE_WINDOW overrides params.merge_window, then the default.
    """
    assert n_partials >= 1

    want = None
    raw = os.environ.get("NOVA_BF_MERGE_WINDOW", "").strip()

    if raw:
        try:
            want = int(raw)
        except ValueError:
            pass

        if want is not None and want < 1:
            want = None

        if want is None:
            logger.warning(
                "NOVA_BF_MERGE_WINDOW=%r is not a positive integer; "
                "using params.merge_window",
                raw,
            )

    if want is None:
        want = getattr(getattr(cfg, "params", None), "merge_window", None)

    if not (isinstance(want, int) and not isinstance(want, bool) and want >= 1):
        want = _MERGE_WINDOW_DEFAULT

    # Never keep more readers in flight than there are partials.
    n = min(want, n_partials)

    logger.info(
        "merge window: %d partial(s) in flight%s",
        n,
        "" if n == want
        else f" (requested {want}, clamped to {n_partials} partial(s))",
    )
    return n

_TARGET_CANDIDATE_SLOTS = 20_000_000

def _resolve_batch_rows(explicit: int | None, n_rows: int, k: int) -> int:
    """Choose batch rows from the target number of candidate slots.

    `2 * k`, NOT `n_partials * k`: a fold holds the running state plus ONE
    partial, so the candidate grid is `B x 2k` however many partials the merge
    has. That is why the partial count is not a parameter here -- it only ever
    appeared in the warning text below.
    """
    per_row = max(1, 2 * k)
    ceiling = max(1, min(_TARGET_CANDIDATE_SLOTS // per_row, n_rows))

    if explicit is None:
        return ceiling

    want = max(1, min(explicit, n_rows))
    if want > ceiling:
        logger.warning(
            "params.merge_batch_size=%d uses %.1fM candidate slots per fold "
            "(2 x k=%d per row), above the %.1fM automatic target; using it as "
            "requested. Drop the setting to let merge size itself (%d rows) if "
            "this runs out of memory.",
            explicit,
            want * per_row / 1e6,
            k,
            _TARGET_CANDIDATE_SLOTS / 1e6,
            ceiling,
        )

    return want



def _id_tie_grid(
    scatter: list,
    rows: np.ndarray,
    b: int,
    width: int,
) -> np.ndarray:
    """Build lexicographic ID ranks for selected candidate rows."""
    dest = np.full(b, -1, dtype=np.int64)
    dest[rows] = np.arange(len(rows))

    subs, places = [], []
    for row_idx, col, flat_ids in scatter:
        r = dest[row_idx]
        sel = np.flatnonzero(r >= 0)
        if not len(sel):
            continue

        # Reuse the full ID array when every candidate is selected.
        subs.append(
            flat_ids
            if len(sel) == len(flat_ids)
            else flat_ids.take(pa.array(sel, pa.int64()))
        )
        places.append((r[sel], col[sel]))

    grid = np.full(
        (len(rows), width),
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )
    if not subs:
        return grid

    try:
        ords = build_ordinals(subs)
    except ValueError as exc:
        raise ValueError(
            "a partial contains null hit_ids, which cannot break ties "
            "deterministically; re-run `bf compute` with non-null IDs"
        ) from exc

    for (r, c), o in zip(places, ords):
        grid[r, c] = o.astype(np.int64)

    return grid

## Optional fold-backend override; defaults to CUDA when available.
_FOLD_ENV = "NOVA_BF_MERGE_FOLD"

# Fold backends used by the current reduce.
_FOLD_USED: set[str] = set()


def _reset_fold_used() -> None:
    _FOLD_USED.clear()


def _lane_rankable(scatter: list) -> bool:
    """Whether candidate IDs support fixed-width lane ranking."""
    from nova_bf.tiebreak import _fixed_width

    return bool(scatter) and _fixed_width(
        [ids for _, _, ids in scatter]
    ) is not None

def _fold_device(forced_only: bool = False):
    """Return the requested/available fold device, or None for NumPy.

    With `forced_only=True`, return whether a non-NumPy backend was explicitly
    requested.
    """
    want = os.environ.get(_FOLD_ENV, "").strip().lower()

    if forced_only:
        return want not in ("", "numpy", "off", "0")

    if want in ("numpy", "off", "0"):
        return None

    try:
        import torch
    except Exception as exc:
        # Torch import can fail from runtime/library errors, not just ImportError.
        if want:
            raise RuntimeError(
                f"{_FOLD_ENV}={want!r} but torch is unusable: {exc}"
            ) from exc
        return None

    if want in ("torch", "cpu"):
        return torch.device("cpu")

    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{_FOLD_ENV}='cuda' but no CUDA device is available"
            )
        return torch.device("cuda")

    if want:
        raise ValueError(
            f"{_FOLD_ENV}={want!r}; expected cuda, torch, cpu or numpy"
        )

    dev = os.environ.get("NOVA_BF_DEVICE", "").strip().lower()
    if dev and dev != "cuda":
        return None

    return torch.device("cuda") if torch.cuda.is_available() else None

def _fold_torch(
    scores: np.ndarray,
    tie: np.ndarray,
    tie_is_rank: bool,
    k: int,
    device,
) -> np.ndarray:
    """Return top-k column indices using the packed-key fold."""
    import torch

    s = torch.from_numpy(np.ascontiguousarray(scores)).to(device)
    o = torch.from_numpy(np.ascontiguousarray(tie)).to(device)

    # Compress arbitrary tie values to monotone ranks for packing.
    if not tie_is_rank:
        o = _rank_dense(o)

    return _fold_packed(s, o, k).cpu().numpy()


def _fold_packed(s, o, k: int):
    """Return top-k column indices for device-resident score and tie grids."""
    import torch

    from nova_bf.compute import _merge_topk
    from nova_bf.tiebreak import pack

    device = s.device
    b, width = s.shape

    # Padding already loses by score; any packable ordinal is sufficient.
    o = torch.where(s == float("-inf"), torch.zeros_like(o), o)
    key = pack(s, o)

    # Match the NumPy path by ranking NaNs below valid candidates.
    key = torch.where(
        torch.isnan(s),
        torch.full_like(key, SENTINEL_KEY),
        key,
    )

    enc = (
        torch.arange(width, device=device, dtype=torch.int64)
        .expand(b, width)
        .contiguous()
    )

    top_key = key[:, :k].contiguous()
    top_enc = enc[:, :k].contiguous()

    if width > k:
        top_key, top_enc = _merge_topk(
            top_key,
            top_enc,
            [(key[:, k:].contiguous(), enc[:, k:].contiguous(), None)],
            k,
        )

    # Return winners in final score/tiebreak order.
    order = torch.argsort(top_key, dim=1, descending=True, stable=True)
    return top_enc.gather(1, order)


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

class _LazyIds:
    """Deferred ID representation for the running merge state.

    Fixed-width IDs remain as device lanes so later folds can reuse them
    directly; they are materialized as an Arrow ListArray only when needed.
    """

    __slots__ = ("lanes", "W", "counts", "_arr")

    def __init__(self, lanes, W: int, counts: np.ndarray):
        self.lanes = lanes    # (sum(counts), nlanes) int64
        self.W = W            # ID width in bytes
        self.counts = counts  # per-row hit counts, int32
        self._arr = None

    @property
    def dense_k(self) -> int | None:
        """Return the common row width, or None for ragged rows."""
        if len(self.counts) and self.counts.min() == self.counts.max():
            return int(self.counts[0])
        return None

    def __len__(self) -> int:
        # Row count, like the `pa.ListArray` this stands in for. Without it a
        # bare `len(ids)` raises TypeError, and `_validate_fold_inputs`' own
        # row-count check does exactly that -- so lanes state could not reach
        # the fold at all. Every other `_LazyIds` check in that function has an
        # `isinstance` carve-out; this one needs the object to answer instead.
        return len(self.counts)

    def value_lengths(self):
        return pa.array(self.counts, pa.int32())

    def materialize(self) -> pa.ListArray:
        if self._arr is None:
            off = np.empty(len(self.counts) + 1, dtype=np.int32)
            off[0] = 0
            np.cumsum(self.counts, out=off[1:])
            self._arr = pa.ListArray.from_arrays(
                pa.array(off, pa.int32()),
                self._bytes(),
            )
            self.lanes = None  # release device representation
        return self._arr

    def _bytes(self) -> pa.Array:
        """Materialize IDs from lanes, retrying on the host after a CUDA OOM."""
        from nova_bf.tiebreak import ids_from_lanes

        try:
            return ids_from_lanes(self.lanes, self.W)
        except Exception as exc:  # noqa: BLE001
            # Host retry only helps when device memory caused the failure.
            if not _is_oom(exc) or not self.lanes.is_cuda:
                raise

            logger.warning(
                "materialising %d rows of ids ran out of device memory (%s); "
                "retrying on the host",
                len(self.counts),
                exc,
            )

            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

            return ids_from_lanes(self.lanes.cpu(), self.W)


def _as_ids_array(x):
    """A `pa.ListArray` from either representation."""
    return x.materialize() if isinstance(x, _LazyIds) else x

def _validate_fold_inputs(score_lists, id_lists, tie_lists, k: int) -> None:
    """Validate partial structure before either fold backend.
    """
    if len(score_lists) != len(id_lists):
        raise RuntimeError(
            f"fold received {len(score_lists)} score inputs but "
            f"{len(id_lists)} ID inputs"
        )

    if tie_lists is not None and len(tie_lists) != len(score_lists):
        raise RuntimeError(
            f"fold received {len(score_lists)} score inputs but "
            f"{len(tie_lists)} tie inputs"
        )

    if not score_lists:
        return

    n_rows = len(score_lists[0])

    for w, (sl, il) in enumerate(zip(score_lists, id_lists)):
        if len(sl) != n_rows or len(il) != n_rows:
            raise RuntimeError(
                f"partial {w} has {len(sl)} score rows and {len(il)} ID rows; "
                f"expected {n_rows}"
            )

        if sl.type.value_type != pa.float32():
            raise RuntimeError(
                f"partial {w}'s hit_scores are {sl.type.value_type}, not float32; "
                "re-run `bf compute` for this search."
            )

        if sl.null_count:
            raise RuntimeError(
                f"partial {w}'s hit_scores has {sl.null_count} null row(s); "
                "use empty lists for queries with no hits."
            )

        if not isinstance(il, _LazyIds) and il.null_count:
            raise RuntimeError(
                f"partial {w}'s hit_ids has {il.null_count} null row(s); "
                "use empty lists for queries with no hits."
            )

        lengths = (
            sl.value_lengths()
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )

        if len(lengths) and lengths.max() > k:
            raise RuntimeError(
                f"partial {w} has a query with {int(lengths.max())} hits "
                f"but k={k}; a partial must never hold more than k candidates "
                "per query. Re-run `bf compute` for this search."
            )

        id_lengths = (
            il.value_lengths()
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        if not np.array_equal(id_lengths, lengths):
            raise RuntimeError(
                f"partial {w}'s hit_ids rows are split differently from its "
                "hit_scores rows; the columns must line up row for row or "
                "every hit would be reported under the wrong id. Re-run "
                "`bf compute` for this search."
            )

        children = [("hit_scores", sl.flatten())]
        if not isinstance(il, _LazyIds):
            # Lazy state was validated when it was originally folded.
            children.append(("hit_ids", il.flatten()))

        for name, child in children:
            if child.null_count:
                raise RuntimeError(
                    f"partial {w}'s {name} has {child.null_count} null "
                    "value(s) inside its lists; every hit must carry a real "
                    f"{name[4:]}. Re-run `bf compute` for this search."
                )

        if tie_lists is None:
            continue

        tl = tie_lists[w]
        if len(tl) != n_rows:
            raise RuntimeError(
                f"partial {w} has {len(tl)} tie rows; expected {n_rows}"
            )
        if tl.null_count:
            raise RuntimeError(
                f"partial {w}'s hit_tie has {tl.null_count} null row(s); "
                "use empty lists for queries with no hits."
            )
        tie_lengths = (
            tl.value_lengths()
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        if not np.array_equal(tie_lengths, lengths):
            raise RuntimeError(
                f"partial {w}'s hit_tie rows are split differently from its "
                "hit_scores rows; the columns must line up row for row or "
                "ties would be broken against the wrong hits. Re-run "
                "`bf compute` for this search."
            )
        tie_flat = tl.flatten()
        if tie_flat.null_count:
            raise RuntimeError(
                f"partial {w}'s hit_tie has {tie_flat.null_count} null "
                "value(s) inside its lists; a null has no ordering position "
                "and would outrank every real hit. Re-run `bf compute` for "
                "this search."
            )


def _dense_device_fold(score_lists, id_lists, k, device):
    """Fast fold for dense, fixed-width IDs without a host candidate grid.

    Returns None when the batch does not qualify. `sel` and `scatter` are
    returned only when every input still uses Arrow IDs.
    """
    import torch

    from nova_bf.tiebreak import (
        _NO_GPU_ORDINALS,
        _fixed_width,
        _lanes_on_device,
        _gpu_perm_from_lanes,
    )

    # This path bypasses build_ordinals, so honor its GPU-ranking kill switch.
    if os.environ.get(_NO_GPU_ORDINALS):
        return None

    n_inputs = len(score_lists)
    b = len(score_lists[0])
    width = n_inputs * k

    flat_scores, lane_parts, arrow_ids = [], [], []
    for sl, il in zip(score_lists, id_lists):
        lengths = sl.value_lengths().to_numpy(zero_copy_only=False)
        if len(lengths) != b or lengths.min() != k or lengths.max() != k:
            return None

        flat_scores.append(sl.flatten())

        if isinstance(il, _LazyIds):
            if il.dense_k != k or il.lanes is None:
                return None
            lane_parts.append(il.lanes)
            arrow_ids.append(None)
        else:
            lane_parts.append(None)
            arrow_ids.append(il.flatten())

    # Every input must use the same fixed ID width.
    widths = {il.W for il in id_lists if isinstance(il, _LazyIds)}
    present = [ids for ids in arrow_ids if ids is not None]
    if present:
        W = _fixed_width(present)
        if W is None:
            return None
        widths.add(W)

    if len(widths) != 1:
        return None

    W = widths.pop()
    nlanes = (W + 7) // 8

    scores = torch.cat(
        [
            torch.from_numpy(
                np.ascontiguousarray(a.to_numpy(zero_copy_only=False))
            ).to(device).view(b, k)
            for a in flat_scores
        ],
        dim=1,
    )

    lanes = torch.cat(
        [
            lp if lp is not None
            else _lanes_on_device([ids], W, b * k, device)
            for lp, ids in zip(lane_parts, arrow_ids)
        ],
        dim=0,
    )

    total = n_inputs * b * k
    perm = torch.from_numpy(_gpu_perm_from_lanes(lanes)).to(device)

    ordinals = torch.empty(total, dtype=torch.int64, device=device)
    ordinals[perm.long()] = torch.arange(
        total, dtype=torch.int64, device=device
    )
    del perm

    # Map input-major ordinals into the row-major candidate grid.
    tie = (
        ordinals.view(n_inputs, b, k)
        .permute(1, 0, 2)
        .reshape(b, width)
    )
    del ordinals

    top_idx = _fold_packed(scores, tie, k)
    del tie

    top_scores = scores.gather(1, top_idx)
    del scores

    # Keep winning ID lanes on-device for the next fold.
    lanes_win = (
        lanes.view(n_inputs, b, k, nlanes)
        .permute(1, 0, 2, 3)
        .reshape(b, width, nlanes)
        .gather(1, top_idx.unsqueeze(-1).expand(b, k, nlanes))
        .reshape(b * k, nlanes)
    )
    del lanes

    # Arrow gathering is only needed when every input still has Arrow IDs.
    if all(ids is not None for ids in arrow_ids):
        src = (
            torch.arange(total, dtype=torch.int64, device=device)
            .view(n_inputs, b, k)
            .permute(1, 0, 2)
            .reshape(b, width)
        )
        sel = src.gather(1, top_idx).cpu().numpy()
        scatter = [(None, None, ids) for ids in arrow_ids]
    else:
        sel, scatter = None, None

    return top_scores.cpu().numpy(), sel, scatter, (lanes_win, W)


def _topk_merge(
    score_lists: list[pa.ListArray],
    id_lists: list[pa.ListArray | _LazyIds],
    tie_lists: list[pa.ListArray] | None,
    k: int,
    lanes_mode: bool = False,
) -> tuple[pa.ListArray | _LazyIds, pa.ListArray, pa.ListArray | None]:
    """Merge row-aligned candidate lists into a per-query top-K."""
    if not score_lists:
        raise ValueError("fold requires at least one score input")

    # Validate once before either fold backend.
    _validate_fold_inputs(score_lists, id_lists, tie_lists, k)

    n_inputs = len(score_lists)
    b = len(score_lists[0])
    width = n_inputs * k

    # Dense fixed-width IDs can avoid building the host candidate grid.
    fast = None
    dev = None
    if tie_lists is None and b:
        dev = _fold_device()
        if dev is not None:
            try:
                fast = _dense_device_fold(score_lists, id_lists, k, dev)
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise

                logger.warning(
                    "merge fold on %s ran out of memory (%s); "
                    "falling back to the host path",
                    dev,
                    exc,
                )

                if dev.type == "cuda":
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass

    if fast is not None:
        _FOLD_USED.add(f"torch:{dev.type}")

        top_s, sel_all, scatter, lanes_w = fast
        valid = top_s > -np.inf

        return _assemble(
            top_s,
            sel_all,
            valid,
            scatter,
            b,
            None,
            None,
            lanes_win=lanes_w if lanes_mode else None,
            # Materialize lanes now if no Arrow IDs remain.
            lanes_fallback=(
                None if lanes_mode or scatter is not None else lanes_w
            ),
        )

    # The general path requires Arrow IDs.
    id_lists = [_as_ids_array(x) for x in id_lists]

    scores = np.full((b, width), -np.inf, dtype=np.float32)
    src = np.full((b, width), -1, dtype=np.int64)

    want_tie = tie_lists is not None
    ties = (
        np.full(
            (b, width),
            np.iinfo(np.int64).max,
            dtype=np.int64,
        )
        if want_tie
        else None
    )

    scatter: list = []
    base = 0

    # Input structure was validated above.
    for w, (sl, il) in enumerate(zip(score_lists, id_lists)):
        lengths = (
            sl.value_lengths()
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        total = int(lengths.sum())

        if total == 0:
            continue

        flat_s = sl.flatten().to_numpy(zero_copy_only=False)
        flat_ids = il.flatten()

        row_idx = np.repeat(np.arange(b), lengths)

        starts = np.zeros(b, dtype=np.int64)
        np.cumsum(lengths[:-1], out=starts[1:])
        within = np.arange(total) - np.repeat(starts, lengths)

        col = w * k + within
        scores[row_idx, col] = flat_s

        # `src` indexes the concatenated ID arrays stored in `scatter`.
        src[row_idx, col] = base + np.arange(total, dtype=np.int64)
        base += total
        scatter.append((row_idx, col, flat_ids))

        if want_tie:
            ties[row_idx, col] = (
                tie_lists[w]
                .flatten()
                .to_numpy(zero_copy_only=False)
            )

    device = _fold_device()
    if (
        device is not None
        and not want_tie
        and not _fold_device(forced_only=True)
        and not _lane_rankable(scatter)
    ):
        # Avoid Torch ranking for variable-width IDs unless explicitly requested.
        device = None

    _FOLD_USED.add(
        "numpy" if device is None else f"torch:{device.type}"
    )

    if device is not None:
        tie = (
            ties
            if want_tie
            else _id_tie_grid(
                scatter,
                np.arange(b),
                b,
                width,
            )
        )
        top_idx = _fold_torch(
            scores,
            tie,
            not want_tie,
            k,
            device,
        )
        top_s = np.take_along_axis(
            scores,
            top_idx,
            axis=1,
        )
    else:
        top_idx, top_s = _topk_numpy(
            scores,
            ties,
            scatter,
            want_tie,
            b,
            width,
            k,
        )

    # Drop -inf padding; +inf remains a valid score.
    valid = top_s > -np.inf
    sel = np.take_along_axis(
        src,
        top_idx,
        axis=1,
    )

    top_tie = (
        np.take_along_axis(
            ties,
            top_idx,
            axis=1,
        )
        if want_tie
        else None
    )

    out = _assemble(
        top_s,
        sel,
        valid,
        scatter,
        b,
        top_tie,
        want_tie,
    )

    if lanes_mode and not isinstance(out[0], _LazyIds):
        # Keep the running state in lane form when possible.
        try:
            out = (
                _lazy_from_arrow(out[0]),
                out[1],
                out[2],
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise

            logger.warning(
                "re-encoding winners as device lanes ran out of memory (%s); "
                "keeping Arrow IDs",
                exc,
            )

    return out


def _lazy_from_arrow(ids_arr: pa.ListArray) -> pa.ListArray | _LazyIds:
    """Re-encode an Arrow id list as device lanes, preserving row lengths.

    Returns the ARGUMENT UNCHANGED when the ids have no common width to pack
    into -- an all-empty batch has none to measure. Callers take either form
    (`_as_ids_array` accepts both), which is why this can decline rather than
    raise.
    """
    from nova_bf.tiebreak import _fixed_width, _lanes_on_device

    flat = ids_arr.flatten()
    W = _fixed_width([flat])
    if W is None:
        return ids_arr

    # Only import once needed
    import torch

    counts = ids_arr.value_lengths().to_numpy(zero_copy_only=False).astype(np.int32)
    dev = _fold_device() or torch.device("cpu")
    return _LazyIds(_lanes_on_device([flat], W, len(flat), dev), W, counts)

def _assemble(
    top_s,
    sel,
    valid,
    scatter,
    b,
    top_tie,
    want_tie,
    lanes_win=None,
    lanes_fallback=None,
):
    """Assemble winning IDs, scores, and optional ties."""
    counts = valid.sum(axis=1).astype(np.int64)
    total_hits = int(counts.sum())

    # Arrow ListArray offsets are int32.
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

    scores_arr = pa.ListArray.from_arrays(
        off,
        pa.array(top_s[valid], pa.float32()),
    )

    if lanes_win is not None or lanes_fallback is not None:
        lanes, W = (
            lanes_win
            if lanes_win is not None
            else lanes_fallback
        )

        # Row-major both sides: `lanes` is (b*k, nlanes) over (b, k), the same
        # order `top_s[valid]` yields. Reorder either and ids attach to the
        # wrong scores, silently.
        keep = valid.ravel()
        if not keep.all():
            import torch

            idx = torch.from_numpy(np.flatnonzero(keep)).to(lanes.device)
            lanes = lanes[idx]

        lazy = _LazyIds(lanes, W, counts)
        ids_arr = (
            lazy
            if lanes_win is not None
            else lazy.materialize()
        )
    else:
        ids_arr = pa.ListArray.from_arrays(
            off,
            _take_ids(scatter, sel[valid]),
        )

    ties_arr = None
    if want_tie:
        ties_arr = pa.ListArray.from_arrays(
            off,
            pa.array(top_tie[valid], pa.int64()),
        )

    return ids_arr, scores_arr, ties_arr
def preflight_searches(cfg: BruteForceConfig) -> None:
    """Check run-wide merge conditions using listings only.

    `run_merge` repeats these checks per search; this avoids duplicate failures
    when searches are merged concurrently.
    """
    out = Store(cfg.output.path)
    forced = merge_forced(cfg)

    counts: dict[str, int] = {}
    for spec in cfg.searches:
        counts[spec.name] = len(
            out.list_parquets(subpath=partial_dir(cfg, spec))
        )

    missing = sorted(
        name for name, count in counts.items()
        if count == 0
    )
    if missing:
        raise RuntimeError(
            f"no partial results for search(es) {missing} under "
            f"{cfg.output.path} — run `bf compute --num-jobs N` first"
        )

    if len(counts) > 1 and len(set(counts.values())) > 1:
        _refuse(
            forced,
            "<all>",
            f"searches have mismatched partial counts: {counts} — every search "
            "in one `compute` run should have the same number of per-rank "
            "partials; this points to a rank that died before writing all "
            "search outputs. Re-run the missing rank(s) with "
            "`bf compute --num-jobs N --job-rank R` before merging.",
        )

def run_merge(cfg: BruteForceConfig, only: set[str] | None = None) -> dict[str, str]:
    """Merge each search's per-rank partials into its final Parquet output.

    `only` restricts which searches are reduced, not which are validated, so
    cross-search consistency checks still see the complete run.
    """
    _unclamp_decode_threads(cfg)

    # Validate the requested search names before any storage I/O.
    if only is not None:
        unknown = only - {s.name for s in cfg.searches}
        if unknown:
            raise RuntimeError(
                f"--search named {sorted(unknown)}, which this config does not "
                f"define; it has {sorted(s.name for s in cfg.searches)}"
            )

    # Resolve force once and use the same decision throughout the merge.
    forced = merge_forced(cfg)
    if forced:
        logger.error(
            "MERGE FORCED: every provenance check below is advisory. Partials "
            "will be merged even if they come from different runs, a different "
            "config, a different tie-break rule, or an incomplete rank set. "
            "The output is stamped nova_bf.merge_forced=true and is NOT "
            "verified ground truth. DOUBLE COVERAGE is still refused -- a "
            "repeated rank, a rank out of range, an unstamped partial beside "
            "stamped ones, or partials disagreeing about num_jobs: those are "
            "provably wrong output, not merely unverified output."
        )

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

    # All searches from one compute run must have the same partial count.
    if len(partials_by_name) > 1:
        counts = {name: len(partials) for name, partials in partials_by_name.items()}
        if len(set(counts.values())) > 1:
            _refuse(
                forced,
                "<all>",
                f"searches have mismatched partial counts: {counts} — every search in "
                "one `compute` run should have the same number of per-rank partials; "
                "this points to a rank that died partway through writing its per-search "
                "outputs (crash/OOM/preemption). Re-run the missing rank(s) with "
                "`bf compute --num-jobs N --job-rank R` before merging."
            )

    # Readers provide metadata here; `_reduce` streams the actual data.
    readers_by_name: dict[str, list[pq.ParquetFile]] = {
        spec.name: [
            pq.ParquetFile(f.read_path, filesystem=out.fs)
            for f in partials_by_name[spec.name]
        ]
        for spec in cfg.searches
    }

    # Search fingerprints differ, so compare the run-global tie-break rule.
    rules = {
        name: {(r.schema_arrow.metadata or {}).get(TIEBREAK_KEY) for r in readers}
        for name, readers in readers_by_name.items()
    }
    seen = {v for vs in rules.values() for v in vs if v is not None}
    if len(seen) > 1:
        pretty = {n: sorted(x.decode() for x in v if x) for n, v in rules.items()}
        _refuse(
            forced,
            "<all>",
            f"partials were computed under different tie-break rules: {pretty} — "
            "merging them puts hits decided by different rules in one artifact. "
            "Re-run `bf compute` so every search uses one `params.tiebreak`."
        )

    todo = [s for s in cfg.searches if only is None or s.name in only]

    # Use one per-search manifest layout and write each manifest with its output.
    base = run_manifest.base_manifest(cfg, "merge")
    entries = []
    for spec in todo:
        t_search = time.perf_counter()
        started_search = datetime.now(timezone.utc)

        e = _reduce(
            cfg,
            spec,
            out,
            partials_by_name[spec.name],
            readers_by_name[spec.name],
            forced,
        )
        entries.append(e)

        run_manifest.write(
            out,
            run_manifest.manifest_name(cfg, "merge", search=e["name"]),
            {
                **base,
                # Stamp this search's actual completion time.
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": started_search.isoformat(),
                "searches": [e],
                "counts": {
                    "partials_merged": e["partials"],
                    "queries": e["queries"],
                },
                "output_files": [e["output_file"]],
                # Forced status belongs to this search's artifact.
                **({"merge_forced": True} if e.get("merge_forced") else {}),
                "timing": {
                    "elapsed_seconds": round(
                        time.perf_counter() - t_search, 2
                    ),
                    **e["timing"],
                },
            },
        )

    # Retire the legacy run-level manifest only when these outputs cover it.
    # Fan-out merges handle this once in the parent after all children succeed.
    drop_legacy_manifest(out, cfg, {e["output_file"] for e in entries})

    return {e["name"]: e["output_path"] for e in entries}
def drop_legacy_manifest(
    out: Store,
    cfg: BruteForceConfig,
    written: set[str],
) -> None:
    """Remove the legacy run-level manifest when it no longer covers unique outputs.

    `written` contains exact output filenames, since config changes such as `k`
    may preserve a search name while producing a different Parquet file.
    """
    name = run_manifest.manifest_name(cfg, "merge")
    root = out.root.rstrip("/")

    try:
        if out.fs.get_file_info(f"{root}/{name}").type == fs.FileType.NotFound:
            return

        with out.fs.open_input_stream(f"{root}/{name}") as f:
            doc = json.loads(f.read())

        orphans = []
        for entry in doc.get("searches") or []:
            target = entry.get("output_file")

            if not target or target in written:
                continue  # This merge rewrote the exact output.

            if (
                out.fs.get_file_info(f"{root}/{target}").type
                == fs.FileType.NotFound
            ):
                continue  # The recorded output no longer exists.

            # Keep the legacy record unless a per-search manifest covers this
            # exact output file; search names alone are insufficient.
            if _manifest_covers(
                out,
                cfg,
                entry.get("name"),
                target,
            ):
                continue

            orphans.append(target)

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not read %s to check what it still describes (%r); "
            "leaving it alone",
            name,
            exc,
        )
        return

    if orphans:
        logger.info(
            "keeping the legacy run-level manifest %s: it is still the only "
            "record for %s, which this merge did not replace",
            name,
            ", ".join(sorted(orphans)),
        )
        return

    _drop_manifests(
        out,
        [name],
        "merge now records one manifest per search",
    )


def _manifest_covers(out: Store, cfg: BruteForceConfig, name, target: str) -> bool:
    """Whether `name`'s per-search manifest names `target` among its outputs."""
    if not name:
        return False
    own = run_manifest.manifest_name(cfg, "merge", search=name)
    path = f"{out.root.rstrip('/')}/{own}"
    try:
        if out.fs.get_file_info(path).type == fs.FileType.NotFound:
            return False
        with out.fs.open_input_stream(path) as f:
            return target in (json.loads(f.read()).get("output_files") or [])
    except Exception:                               # noqa: BLE001
        return False                    # unreadable: assume it covers nothing


def _drop_manifests(out: Store, names: list[str], why: str) -> None:
    """Delete superseded manifests, ignoring ones already absent."""
    root = out.root.rstrip("/")

    for name in names:
        path = f"{root}/{name}"
        try:
            if out.fs.get_file_info(path).type == fs.FileType.NotFound:
                continue

            out.fs.delete_file(path)

        except FileNotFoundError:
            # Another process may have removed it first.
            continue

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "could not remove the superseded manifest %s (%r); it may "
                "describe a run that is no longer on disk",
                name,
                exc,
            )

        else:
            logger.info(
                "removed the superseded manifest %s: %s",
                name,
                why,
            )


# --- recorded escape hatch for merge provenance checks ------------------------

def _meta_get(meta: dict, key: bytes) -> str | None:
    value = meta.get(key)
    return value.decode() if value is not None else None


def _exchange_ids(stamps) -> set[str]:
    """Exchange run ids carried by these partials ("false"/absent excluded)."""
    ids = {_meta_get(meta, EXCHANGE_KEY) or "false" for _, meta in stamps}
    ids.discard("false")
    return ids


def _exchange_need(cfg: BruteForceConfig, spec: SearchSpec,
                   partials: list[ParquetFile], readers: list[pq.ParquetFile],
                   n_rows: int) -> "tuple[np.ndarray | None, bool, np.ndarray | None]":
    """Collect applied pruning thresholds and dropped NaN counts per query.

    Returns the maximum applied threshold (or None), whether any rank used
    a peer threshold, and the total dropped NaN hits. Merged results must
    satisfy these thresholds to guarantee exact top-k.
    """
    need: "np.ndarray | None" = None
    # NaN evictions on ANY rank count (a rank re-run with the exchange off,
    # the merge's own remedy, carries nan_hits but no applied_threshold).
    nan_total = np.zeros(n_rows, dtype=np.int64)
    for f, r in zip(partials, readers):
        names = r.schema_arrow.names
        cols = [c for c in ("applied_threshold", "nan_hits") if c in names]
        if not cols:
            continue
        tbl = r.read(columns=cols)
        if tbl.num_rows != n_rows:
            raise RuntimeError(
                f"partial {f.read_path} has {tbl.num_rows} rows but the first "
                f"partial has {n_rows}; partials must be row-aligned by query "
                "(same queries, same order). A truncated/mismatched partial can't be merged."
            )
        if "applied_threshold" in cols:
            if need is None:
                need = np.full(n_rows, -np.inf, dtype=np.float32)
            col = tbl.column("applied_threshold").to_numpy(zero_copy_only=False)
            # fmax: a NaN entry never raises the requirement; +inf cannot occur
            # (compute never applies one) but would only make the check stricter.
            np.fmax(need, col.astype(np.float32, copy=False), out=need)
        if "nan_hits" in cols:
            nan_total += tbl.column("nan_hits").to_numpy(zero_copy_only=False).astype(np.int64)
    active = _exchange_active(need)
    if (nan_total > 0).any():
        affected = int(((nan_total > 0) & (need > -np.inf)).sum()) if need is not None else 0
        logger.warning(
            "search=%r: %d of %d queries had NaN-scored corpus hits dropped on some "
            "rank (NaN/inf embeddings in the corpus; %d of them also have an applied "
            "peer threshold). Each such query's merged list is cut to its top "
            "(k - dropped), which is what a single-node run returns for it (the NaN "
            "hits took those slots), and that is what the threshold verification "
            "covers.",
            spec.name, int((nan_total > 0).sum()), n_rows, affected,
        )
    ids = _exchange_ids([(None, r.schema_arrow.metadata or {}) for r in readers])
    if len(ids) > 1 and active:
        logger.warning(
            "search=%r: partials carry %d different threshold-exchange ids (%s): "
            "those groups did not exchange thresholds with each other (different "
            "nova-bf code, or a rank re-run after a config change). Harmless for "
            "exactness, which is verified below, but the warm start was lost.",
            spec.name, len(ids), sorted(i[:12] for i in ids),
        )
    return need, active, nan_total


def _exchange_active(need: "np.ndarray | None") -> bool:
    """Did any rank actually prune against a peer's threshold? Only then do
    the partials depend on one another (and nothing may be forced)."""
    return need is not None and bool((need > -np.inf).any())


def _exchange_blame(partials: list[ParquetFile], readers: list[pq.ParquetFile],
                    rows: "np.ndarray", limit: "np.ndarray") -> list[int]:
    """Identify ranks whose pruning thresholds exceeded verified score limits.

    A limit of -inf flags a short result, implicating any rank that pruned.
    Other ranks remain valid because their thresholds reflect real scores.
    """
    ranks: set[int] = set()
    for i, (f, r) in enumerate(zip(partials, readers)):
        if "applied_threshold" not in r.schema_arrow.names:
            continue
        col = r.read(columns=["applied_threshold"]).column(0).to_numpy(zero_copy_only=False)
        if (~(limit >= col[rows])).any():
            rk = _meta_get(r.schema_arrow.metadata or {}, JOB_RANK_KEY)
            ranks.add(int(rk) if rk is not None else i)
    return sorted(ranks)


def _check_exchange_batch(spec: SearchSpec, need: "np.ndarray", nan_total: "np.ndarray",
                          row0: int, scores_arr: pa.ListArray) -> dict:
    """Check whether merged results preserve cross-worker pruning guarantees.

    Queries with a threshold must have at least k - N hits, where N is the
    number of NaN-scored hits dropped. The (k - N)-th score must meet the
    threshold. Queries with N >= k cannot be verified.

    Returns batch-relative rows, score limits, and verification counts.
    """
    n = len(scores_arr)
    sl = need[row0:row0 + n]
    nn = nan_total[row0:row0 + n]
    kk = spec.k - nn                       # how many hits the result must vouch for
    # Every query with ANY applied threshold is checked, +inf included (an
    # applied +inf pruned everything, so it must be caught, not skipped).
    has = sl > -np.inf
    checked = has & (kk >= 1)
    empty = {"rows": np.zeros(0, np.int64), "limit": np.zeros(0, np.float32),
             "short": 0, "low": 0, "checked": int(checked.sum()),
             "skipped": int((has & ~checked).sum())}
    if not checked.any():
        return empty
    col = scores_arr.combine_chunks() if isinstance(scores_arr, pa.ChunkedArray) else scores_arr
    offsets = col.offsets.to_numpy()
    lens = np.diff(offsets)
    # `values` is the WHOLE child array even for a sliced ListArray; rebase so
    # the last row's segment ends at its own offset, not at the child's end.
    vals = col.values.to_numpy(zero_copy_only=False)[offsets[0]:offsets[-1]]
    offsets = offsets - offsets[0]
    # The score each row can vouch for: its kk-th best.
    vouch = np.full(n, np.inf, dtype=np.float32)
    nonempty = lens > 0
    if nonempty.any():
        vouch[nonempty] = np.fmin.reduceat(vals, offsets[:-1][nonempty])
    for i in np.flatnonzero(checked & (nn > 0) & nonempty):
        row = vals[offsets[i]:offsets[i + 1]]
        row = np.sort(row[~np.isnan(row)])[::-1]
        vouch[i] = row[kk[i] - 1] if len(row) >= kk[i] else -np.inf
    short = checked & (lens < kk)
    low = checked & ~short & ~(vouch >= sl)
    bad = short | low
    rows = np.flatnonzero(bad)
    limit = np.where(short[rows], np.float32(-np.inf), vouch[rows]).astype(np.float32)
    return {"rows": rows, "limit": limit, "short": int(short.sum()), "low": int(low.sum()),
            "checked": int(checked.sum()), "skipped": int((has & ~checked).sum()),
            "example": (int(rows[0]), int(lens[rows[0]]), int(kk[rows[0]]),
                        float(vouch[rows[0]]), float(sl[rows[0]])) if len(rows) else None}


def _truncate_lists(arr: pa.ListArray, keep: "np.ndarray") -> pa.ListArray:
    """Each row of `arr` cut to its first `keep[i]` elements (rows already
    shorter are left alone)."""
    col = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    offsets = col.offsets.to_numpy().astype(np.int64)
    lens = np.diff(offsets)
    new_lens = np.minimum(lens, np.maximum(keep, 0))
    if (new_lens == lens).all():
        return col
    starts = offsets[:-1]
    new_offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(new_lens, out=new_offsets[1:])
    total = int(new_offsets[-1])
    # Flat index of every kept element: row start + position within the row.
    idx = (np.repeat(starts, new_lens)
           + np.arange(total, dtype=np.int64)
           - np.repeat(new_offsets[:-1], new_lens))
    values = col.values.take(pa.array(idx, pa.int64()))
    return pa.ListArray.from_arrays(pa.array(new_offsets.astype(np.int32)), values)


def _exchange_verify(cfg: BruteForceConfig, spec: SearchSpec, need: "np.ndarray",
                     nan_total: "np.ndarray", batches, batch_rows: int, blame,
                     quiet: bool = False) -> int:
    """Run `_check_exchange_batch` over every batch, then raise ONCE
    (unforceably) naming every rank at fault across all batches, so the
    operator re-runs the right set in one go. Returns the number of queries
    verified."""
    all_rows: list[np.ndarray] = []
    all_limit: list[np.ndarray] = []
    short = low = checked = skipped = 0
    example = None
    for bi, scores_arr in batches:
        res = _check_exchange_batch(spec, need, nan_total, bi * batch_rows, scores_arr)
        short += res["short"]; low += res["low"]
        checked += res["checked"]; skipped += res["skipped"]
        if len(res["rows"]):
            all_rows.append(res["rows"] + bi * batch_rows)
            all_limit.append(res["limit"])
            if example is None:
                r0, ln, kk, vo, th = res["example"]
                example = (bi * batch_rows + r0, ln, kk, vo, th)
    if skipped and not quiet:
        logger.warning(
            "search=%r: %d queries with an applied peer threshold dropped k or more "
            "NaN-scored hits and cannot be verified at all.", spec.name, skipped,
        )
    if not all_rows:
        return checked
    rows = np.concatenate(all_rows); limit = np.concatenate(all_limit)
    culprits = blame(rows, limit) if blame is not None else []
    r0, ln, kk, vo, th = example
    why = (f"only {ln} of the {kk} hits it must vouch for survived"
           if ln < kk else f"{kk}-th score {vo:.6g} < applied threshold {th:.6g}")
    remedy = (
        f"re-run ONLY rank(s) {culprits} (`bf compute --num-jobs N --job-rank R`, "
        "same config, with `threshold_exchange: false`) and merge again; the other "
        "partials are valid"
        if culprits else
        f"delete {_thresholds.SUBDIR}/ and the partials, and re-run `bf compute`"
    )
    raise RuntimeError(
        f"search={spec.name!r}: THRESHOLD-EXCHANGE VERIFICATION FAILED for "
        f"{len(rows)} of {checked} checked queries ({short} short, {low} below "
        f"threshold; e.g. row {r0}: {why}). Rank(s) {culprits or '?'} pruned against "
        "thresholds that were not lower bounds for this corpus/queries (leftover "
        f"files under {cfg.output.path}/{_thresholds.SUBDIR}/ from an earlier run with "
        "different content under the same paths, or a rank re-scored with a different "
        f"GPU/batch size under dot/euclidean/TF32). No output was written: {remedy}. "
        "Refused even under NOVA_BF_MERGE_FORCE."
    )


def _refuse(forced: bool, spec_name: str, message: str) -> None:
    """Raise unless this merge was explicitly forced."""
    if not forced:
        raise RuntimeError(message)

    logger.error(
        "MERGE CHECK FORCED for search=%r. The merge would otherwise have "
        "refused: %s This artifact is stamped nova_bf.merge_forced=true and "
        "is NOT verified ground truth.",
        spec_name,
        message,
    )


def _inputs_forced(readers: list[pq.ParquetFile]) -> bool:
    """Return whether any input was produced by a forced merge."""
    return any(
        (r.schema_arrow.metadata or {}).get(FORCED_KEY) == b"true"
        for r in readers
    )



def _validate_one_run(
    cfg: BruteForceConfig,
    spec: SearchSpec,
    partials: list[ParquetFile],
    readers: list[pq.ParquetFile],
    forced: bool = False,
    exchange_active: bool = False,
) -> str | None:
    """Validate that partials form one complete, non-overlapping run.

    Returns the run fingerprint to carry onto the merged artifact. Missing
    legacy metadata is warned about where it cannot be verified; known double
    coverage is always refused.
    """
    if len(partials) != len(readers):
        raise RuntimeError(
            f"search={spec.name!r}: found {len(partials)} partials but opened "
            f"{len(readers)} readers"
        )
    if not partials:
        raise RuntimeError(f"search={spec.name!r}: no partials to merge")

    stamps = [
        (f, r.schema_arrow.metadata or {})
        for f, r in zip(partials, readers)
    ]

    def _get(meta: dict, key: bytes) -> str | None:
        value = meta.get(key)
        return value.decode() if value is not None else None

    # Partials that pruned against each other's thresholds are only exact as
    # a complete, single-run, single-config set: none of the checks below may
    # be forced for them (a forced merge would be WRONG, not just unverified).
    if exchange_active:
        forced = False

    # All stamped partials must belong to the same run.
    runs = {f.read_path: _get(meta, RUN_KEY) for f, meta in stamps}
    present = {sha for sha in runs.values() if sha is not None}

    if not present:
        logger.warning(
            "search=%r: none of the %d partials carry a run fingerprint; "
            "cannot verify they came from a single run. Re-run `bf compute` "
            "if this directory may contain partials from multiple runs.",
            spec.name,
            len(partials),
        )
        run_sha = None
    elif any(sha is None for sha in runs.values()) or len(present) > 1:
        by_run: dict[str, list[str]] = {}
        for path, sha in runs.items():
            by_run.setdefault(sha or "(unstamped)", []).append(path)

        summary = "; ".join(
            f"{sha[:12] if sha != '(unstamped)' else sha}: "
            f"{len(paths)} partial(s), e.g. {sorted(paths)[0]}"
            for sha, paths in sorted(by_run.items())
        )

        _refuse(
            forced,
            spec.name,
            f"search={spec.name!r}: the partials under "
            f"{cfg.output.path}/{partial_dir(cfg, spec)}/ come from MORE THAN "
            f"ONE run — {summary}. Merging them could double-count overlapping "
            "corpus slices and omit others, producing a wrong top-K that looks "
            "normal. Delete the directory and re-run `bf compute` for this search.",
        )
        run_sha = None
    else:
        run_sha = next(iter(present))

    # Verify config fingerprints where available. Legacy unstamped partials are
    # allowed, but explicitly remain unverified.
    want_config = config_identity(cfg, spec)
    configs = {
        f.read_path: _get(meta, CONFIG_KEY)
        for f, meta in stamps
    }

    unstamped_configs = sorted(
        path for path, sha in configs.items() if sha is None
    )
    if unstamped_configs:
        logger.warning(
            "search=%r: %d of %d partial(s) lack a config fingerprint; their "
            "config cannot be verified.",
            spec.name,
            len(unstamped_configs),
            len(stamps),
        )

    mismatched = sorted(
        path
        for path, sha in configs.items()
        if sha is not None and sha != want_config
    )
    if mismatched:
        _refuse(
            forced,
            spec.name,
            f"search={spec.name!r}: partial {mismatched[0]} was computed from "
            "a different config than this merge was given "
            "(metric/k/filter/rows, corpus or query paths/columns, or "
            "`allow_tf32` differ). Merge with the config that produced these "
            "partials, or re-run `bf compute`.",
        )

    # Rank identity requires job_rank; completeness additionally requires num_jobs.
    ranks: list[int] = []
    for f, meta in stamps:
        rank = _get(meta, JOB_RANK_KEY)
        if rank is None:
            continue

        try:
            r = int(rank)
        except ValueError:
            raise RuntimeError(
                f"search={spec.name!r}: partial {f.read_path} declares "
                f"job_rank={rank!r}, which is not an integer. Its metadata is "
                "corrupt; re-run `bf compute` for that rank."
            ) from None

        if r < 0:
            raise RuntimeError(
                f"search={spec.name!r}: partial {f.read_path} declares "
                f"job_rank={r}; job_rank must be non-negative."
            )

        stem = f.read_path.rsplit("/", 1)[-1]
        if (m := re.fullmatch(r"rank(\d+)\.parquet", stem)) and int(m.group(1)) != r:
            raise RuntimeError(
                f"search={spec.name!r}: {stem} declares job_rank={r} — the "
                "filename and metadata disagree, so the rank set cannot be "
                "trusted. Delete the directory and re-run `bf compute`."
            )

        ranks.append(r)

    ranks.sort()
    dupes = sorted({
        a for a, b in zip(ranks, ranks[1:])
        if a == b
    })

    declared = {_get(meta, NUM_JOBS_KEY) for _, meta in stamps}
    sharded = declared != {None}

    # Mixed stamped/unstamped ranks are ambiguous. Fully legacy, unsharded
    # directories are allowed with a warning because distinctness is unknowable.
    unstamped_ranks = len(ranks) != len(stamps) and (bool(ranks) or sharded)

    if not ranks and not sharded and len(stamps) > 1:
        logger.warning(
            "search=%r: none of the %d partials carry a job_rank, so distinct "
            "corpus coverage cannot be verified. These partials predate the "
            "stamp; re-run `bf compute` if that guarantee is required.",
            spec.name,
            len(stamps),
        )

    # Duplicate or ambiguous rank coverage can count corpus slices twice, so it
    # is never forceable.
    if dupes or unstamped_ranks:
        raise RuntimeError(
            f"search={spec.name!r}: the partial directory covers rank(s) "
            + (f"{dupes} more than once" if dupes else "")
            + (" and " if dupes and unstamped_ranks else "")
            + (
                f"{len(stamps) - len(ranks)} partial(s) carry no job_rank and "
                "cannot be shown not to duplicate another rank"
                if unstamped_ranks else ""
            )
            + f" (ranks present: {ranks}). Those corpus slices could be counted "
            "TWICE, producing duplicate document ids and fewer than k distinct "
            "results. This is refused even under NOVA_BF_MERGE_FORCE. Delete "
            "the duplicate/stray partials and re-run the affected rank(s)."
        )

    if not sharded:
        return run_sha

    # Different num_jobs values describe different corpus partitions and can
    # therefore overlap even when their rank numbers differ.
    if len(declared) > 1:
        raise RuntimeError(
            f"search={spec.name!r}: partials disagree about how many ranks the "
            f"run had ({sorted(str(d) for d in declared)}) — their rank numbers "
            "refer to DIFFERENT corpus partitions and those partitions can "
            "overlap. This is refused even under NOVA_BF_MERGE_FORCE. Delete "
            "the partial directory and re-run `bf compute`."
        )

    raw_num_jobs = next(iter(declared))
    try:
        num_jobs = int(raw_num_jobs)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"search={spec.name!r}: num_jobs={raw_num_jobs!r} is not an integer; "
            "the partial metadata is corrupt."
        ) from None

    if num_jobs < 1:
        raise RuntimeError(
            f"search={spec.name!r}: num_jobs={num_jobs} must be positive; "
            "the partial metadata is corrupt."
        )

    expected = set(range(num_jobs))
    actual = set(ranks)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if extra:
        raise RuntimeError(
            f"search={spec.name!r}: rank(s) {extra} are outside "
            f"0..{num_jobs - 1} (ranks present: {ranks}); their corpus slices "
            "are not part of this run. Delete the stray partials and re-run "
            "`bf compute`."
            + (
                f" This directory is ALSO missing rank(s) {missing}; fix both "
                "before merging."
                if missing else ""
            )
        )

    # Missing coverage is incomplete but non-duplicated, so it may be forced —
    # UNLESS the ranks pruned against each other's thresholds: then the
    # present partials dropped candidates on the strength of the missing
    # rank's witnesses, and merging without it is WRONG, not merely short.
    if missing and exchange_active:
        raise RuntimeError(
            f"search={spec.name!r}: rank(s) {missing} are missing and the present "
            "partials were computed with params.threshold_exchange, so they pruned "
            "candidates that only the missing rank(s) can vouch for. Merging without "
            "them would return a WRONG top-K (not just an incomplete one); this is "
            "refused even under NOVA_BF_MERGE_FORCE. Re-run `bf compute --num-jobs "
            f"{num_jobs} --job-rank R` for the missing rank(s) first."
        )
    if missing:
        _refuse(
            forced,
            spec.name,
            f"search={spec.name!r}: the run declared {num_jobs} ranks but this "
            f"directory holds {len(stamps)} partial(s) covering ranks {ranks}, "
            f"missing {missing}. Each missing rank's corpus slice is absent "
            "from the merged top-K, silently lowering recall computed against "
            "it. Re-run `bf compute --num-jobs "
            f"{num_jobs} --job-rank R` for the missing rank(s) before merging.",
        )

    return run_sha


def _reduce(
    cfg: BruteForceConfig, spec: SearchSpec, out: Store, partials: list[ParquetFile],
    readers: list[pq.ParquetFile], forced: bool = False,
) -> dict:
    """Reduce one search's partials into its final Parquet and manifest entry."""
    k = spec.k
    _reset_fold_used()
    n_rows = readers[0].metadata.num_rows
    exchange_need, exchange_active, exchange_nan = _exchange_need(cfg, spec, partials, readers, n_rows)
    exchange_verified = 0
    nan_cut_queries = 0
    # Partials that pruned against each other's thresholds are only exact as
    # a complete, single-run, single-config set: nothing may be forced for
    # them, and the output must not be stamped as forced either. Partials
    # that carry the column but never applied a finite threshold (all -inf:
    # the exchange was on but no peer was ever seen) are ordinary partials.
    if exchange_active:
        forced = False
    # The reduced artifact records that its partials pruned against each
    # other (and were verified): the single exchange id, or "mixed".
    _xids = _exchange_ids([(None, r.schema_arrow.metadata or {}) for r in readers])
    exchange_stamp: str | bool = (
        (_xids.pop() if len(_xids) == 1 else "mixed") if exchange_active else False
    )
    # Tiles-mode provenance comes from the partials, never from this merge's
    # own config or environment: "tiles" when every partial carries it,
    # "mixed" when only some do.
    _rescore = {(r.schema_arrow.metadata or {}).get(RESCORE_KEY) for r in readers}
    rescore_stamp = (None if _rescore == {None}
                     else "tiles" if _rescore == {b"tiles"} else "mixed")
    # Validate that all partials belong to the same complete compute run.
    run_sha = _validate_one_run(cfg, spec, partials, readers, forced, exchange_active)
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
            _refuse(forced, spec.name,
                f"partial {f.read_path} was computed with params.tiebreak="
                f"{stamped!r}, but this merge was given {cfg.params.tiebreak!r}. "
                "Ties would be reduced by a rule the partials were not built for. "
                "Re-run `bf compute`, or merge with the config that produced them."
            )

    # All partials must agree on whether an explicit tie ordinate is carried.
    has_tie = ["hit_tie" in r.schema_arrow.names for r in readers]
    if any(has_tie) and not all(has_tie):
        missing = [f.read_path for f, h in zip(partials, has_tie) if not h]
        _refuse(forced, spec.name,
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
    batch_rows = _resolve_batch_rows(cfg.params.merge_batch_size, n_rows, k)
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
            ids0, sc0, ti0 = _topk_merge([sc], [ids], [ti] if want_tie else None,
                                         k, lanes_mode=lanes_mode)
            state[idx] = (ids0, sc0, ti0)
            return
        c_ids, c_sc, c_ti = cur
        # `lanes_mode` is fixed for the search (see `_reduce`), so the state
        # keeps the same id representation on every fold: the next fold ranks
        # it without re-packing and without an Arrow `Take`, and the strings
        # are built once, at write time.
        ids2, sc2, ti2 = _topk_merge(
            [c_sc, sc], [c_ids, ids],
            [c_ti, ti] if want_tie else None, k, lanes_mode=lanes_mode,
        )
        state[idx] = (ids2, sc2, ti2)

    ranged = bool(cfg.params.merge_ranged_reads)
    window_n = _merge_window(cfg, len(partials))
    # Keep one ID representation for the running state throughout the search.
    lanes_eligible = not want_tie and _fold_device() is not None
    lanes_mode = False
    lanes_decided = not lanes_eligible
    if not lanes_eligible:
        logger.info("merge state ids: arrow")

    inputs_forced = _inputs_forced(readers)
    # Divide ranged-read concurrency across in-flight partials. The pool is a
    # budget, not a ceiling: a window wider than the pool still gives each
    # reader at least one GET, so `merge_window: 64` really does run 64.
    per_file = max(1, _RANGED_GET_POOL // max(1, window_n))
    src = Store(out.uri, ranged_get=ranged, ranged_get_concurrency=per_file)
    # Give the first partial extra concurrency because folding cannot begin
    # until it has arrived. The constraint is GETs per file, not bandwidth.
    # Invalid overrides fall back to the default pool size.
    raw_first = os.environ.get("NOVA_BF_FIRST_GETS", "").strip()
    try:
        first_gets = int(raw_first) if raw_first else _RANGED_GET_POOL
        if first_gets < 1:
            raise ValueError(first_gets)
    except ValueError:
        logger.warning("NOVA_BF_FIRST_GETS=%r is not a positive integer; "
                       "using %d", raw_first, _RANGED_GET_POOL)
        first_gets = _RANGED_GET_POOL
    src_first = (Store(out.uri, ranged_get=ranged, ranged_get_concurrency=first_gets)
                 if ranged and first_gets != per_file else src)
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
            reader = src_first if i == 0 else src
            q.put((i, reader.read_columns(f.read_path, cols)))
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
    # PHASE TIMERS.
    t_io = 0.0          # consumer blocked waiting for a partial to arrive
    t_fold = 0.0        # folding a partial into the running top-K state
    t_write = 0.0       # the final parquet write
    t_reduce0 = time.perf_counter()
    try:
        # Track successful starts so only live readers are drained/joined.
        for t in threads:
            t.start()
            started += 1
        # Drain every started reader even after failure; otherwise readers can 
        # remain blocked on the queue or semaphore while holding partial buffers.
        for _ in range(started):
            _t = time.perf_counter()
            i, tbl = q.get()
            t_io += time.perf_counter() - _t
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
            # Fix the id representation from the first partial to arrive, before
            # anything is folded, so every fold in this search sees one setting.
            if not lanes_decided:
                lanes_mode = _decide_lanes(tbl, spec)
                lanes_decided = True
                logger.info("merge state ids: %s",
                            "device lanes" if lanes_mode else "arrow")
            try:
                _t = time.perf_counter()
                for bi in range(n_batches):
                    sl = tbl.slice(bi * batch_rows, batch_rows)
                    if sl.num_rows:
                        _fold(bi, sl, keep_head=(i == 0))
                t_fold += time.perf_counter() - _t
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
    t_write0 = time.perf_counter()
    # Did a previous, possibly GOOD, result already exist under this name? It
    # only decides the WORDING of the failure messages below; what actually
    # survives a failed merge is settled by whether the commit ran, and the
    # measurements for that are with the cleanup in the `finally`.
    try:
        existed = out.fs.get_file_info(path).type != fs.FileType.NotFound
    except Exception:
        existed = False
    if exchange_need is not None:
        # Verify EVERY batch before the output is opened: a failure here is
        # wrong ground truth, and it must not destroy a previous good result
        # under the same name (local streams truncate on open; a closed S3
        # stream commits). The in-loop check below stays as a guard.
        _blame = lambda rows, limit: _exchange_blame(partials, readers, rows, limit)  # noqa: E731
        exchange_verified = _exchange_verify(
            cfg, spec, exchange_need, exchange_nan,
            ((bi, state[bi][1]) for bi in range(n_batches)), batch_rows, _blame)
    sink = out.fs.open_output_stream(path)
    writer: pq.ParquetWriter | None = None
    body_ok = False
    try:
        for bi in range(n_batches):
            ids_state, scores_arr, _ = state[bi]
            base = head[bi]
            # Materialise the strings ONCE, here, if the state carried lanes.
            ids_arr = _as_ids_array(ids_state)
            if exchange_need is not None:
                # Before this batch is committed: a failure here takes the
                # normal failed-merge path (output removed, manifest dropped).
                _exchange_verify(cfg, spec, exchange_need, exchange_nan,
                                 [(bi, scores_arr)], batch_rows, _blame, quiet=True)
            # A query whose ranks dropped N NaN-scored hits: each rank kept
            # only its top (k - its NaNs) real candidates, so places k-N+1..k
            # of the merged list would be filled from other ranks' leftovers
            # rather than true neighbours. Cut to the top (k - N), which is
            # exactly what a single-node run returns for that query (its NaN
            # hits dropped, list short) and what the verification covers.
            nn = exchange_nan[bi * batch_rows: bi * batch_rows + len(scores_arr)]
            if (nn > 0).any():
                keep = (k - nn).astype(np.int64)
                before = ids_arr.value_lengths().to_numpy(zero_copy_only=False)
                # Rare path (NaN rows in the corpus); the `take` copies the
                # child arrays, a few seconds per 110k x 1000 batch.
                ids_arr = _truncate_lists(ids_arr, keep)
                scores_arr = _truncate_lists(scores_arr, keep)
                nan_cut_queries += int((before > np.maximum(keep, 0)).sum())
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
                           reducing=True, num_jobs=len(partials),
                           forced=forced, inputs_forced=inputs_forced,
                           threshold_exchange=exchange_stamp,
                           two_pass_rescore=rescore_stamp)
            )
            if writer is None:
                writer = pq.ParquetWriter(sink, table.schema, compression="snappy")
            writer.write_table(table)
            state[bi] = head[bi] = qref[bi] = None   # release as we go
        body_ok = True
    finally:
        # Close both resources without masking an error already in flight.
        # Close the writer and the stream separately: `writer.close()` writes
        # the Parquet footer, `sink.close()` commits the object-store upload.
        writer_err: BaseException | None = None
        sink_err: BaseException | None = None
        try:
            if writer is not None:
                writer.close()
        except BaseException as exc:                # noqa: BLE001
            writer_err = exc
        try:
            sink.close()
        except BaseException as exc:                # noqa: BLE001
            sink_err = exc
        close_err = writer_err or sink_err

        # A successful body is not enough: writer.close() commits the Parquet footer.
        wrote = body_ok and close_err is None
        # Local streams truncate on open; on S3 the object changes only if the
        # commit runs, so a failed `sink.close()` leaves a previous result whole.
        committed = not out.is_s3 or sink_err is None
        if not wrote and committed:
            # Remove the incomplete output committed under the final name.
            try:
                out.fs.delete_file(path)
                removed = True
            except FileNotFoundError:
                removed = True          # nothing to clean up; same end state
            except BaseException as exc:            # noqa: BLE001
                removed = False
                logger.error(
                    "search=%r: merge failed AND the incomplete %s could not be "
                    "removed (%r) — DELETE IT BY HAND; it is a truncated result "
                    "under the name a finished one would have",
                    spec.name, path, exc)
            if removed:
                logger.error("search=%r: merge failed; removed the incomplete %s",
                             spec.name, path)
            else:
                logger.error(
                    "search=%r: its manifest has been removed too, so nothing "
                    "claims %s is a finished result", spec.name, path)
            # Either way: remove any manifest that would now describe a
            # missing or incomplete output.
            _drop_manifests(
                out, [run_manifest.manifest_name(cfg, "merge", search=spec.name)],
                "the output it described was removed by a failed merge")
            if existed:
                logger.error(
                    "search=%r: %s HELD A PREVIOUS RESULT and this run "
                    "overwrote it before failing — it is gone, not stale. "
                    "Re-run the merge; there is nothing to fall back on.",
                    spec.name, path)
        elif not wrote:
            # S3, and the commit never ran. DO NOT DELETE: there is no object
            # of ours to remove, and if a previous result is there it is whole.
            logger.error(
                "search=%r: merge failed before committing %s (%r). Nothing "
                "was uploaded under that name by this run%s",
                spec.name, path, sink_err,
                (" — the object already there is the PREVIOUS result, intact. "
                 "Check its modification time before trusting it as the result "
                 "of THIS run." if existed else "."))
        if not wrote and close_err is not None:
            logger.error("search=%r: also failed to close the output: %r",
                         spec.name, close_err)
        if writer_err is not None:
            # Prevent `ParquetWriter.__del__` from retrying a failed close.
            try:
                writer.is_open = False
            except Exception:                       # noqa: BLE001
                pass
        # Closing/committing the output is part of a successful write.
        if body_ok and close_err is not None:
            raise close_err

    warn_if_short(short_count, n_rows, k, spec.name, logger)

    t_write = time.perf_counter() - t_write0
    t_reduce = time.perf_counter() - t_reduce0
    # `other_s` is the RESIDUAL, and it is printed because the three phases do
    # NOT sum to the reduce.
    t_other = t_reduce - t_io - t_fold - t_write

    def _read_bytes(i: int, r: pq.ParquetFile) -> int:
        want = None if ranged else set(
            hit_cols + ["query_id"] + (payload_cols if i == 0 else []))
        return sum(
            col.total_compressed_size
            for g in range(r.metadata.num_row_groups)
            for col in (r.metadata.row_group(g).column(c)
                        for c in range(r.metadata.row_group(g).num_columns))
            
            if want is None or col.path_in_schema in want
            or col.path_in_schema.split(".")[0] in want
        )
    _gb = sum(_read_bytes(i, r) for i, r in enumerate(readers)) / 1e9
    logger.info(
        "merge-bench search=%r partials=%d queries=%d k=%d gb=%.2f "
        "reduce_s=%.1f io_wait_s=%.1f fold_s=%.1f write_s=%.1f other_s=%.1f "
        "fold=%s read_mbps=%.0f",
        spec.name, len(partials), n_rows, k, _gb,
        t_reduce, t_io, t_fold, t_write, t_other,
        # `_FOLD_USED`, not `_fold_device()`. The latter reports the device
        # that was AVAILABLE; `_topk_merge` then declines it and falls back to
        # NumPy whenever the ids are not fixed-width lane-rankable, so the
        # field said `cuda` in precisely the case it exists to detect -- "the
        # kernel is slow" versus "there was no kernel".
        ",".join(sorted(_FOLD_USED)) or "none",
        (_gb * 1000 / t_reduce) if t_reduce else 0.0,
    )
    logger.info("search=%r wrote %s (%d queries)", spec.name, path, n_rows)
    entry = run_manifest.search_entry(spec)
    entry.update({
        "queries": n_rows,
        "output_file": result_name(cfg, spec),
        "output_path": path,
        "partials": len(partials),
        "partial_dir": partial_dir(cfg, spec),
        "merge_batch_rows": batch_rows,
        **({"threshold_exchange_verified_queries": exchange_verified}
           if exchange_need is not None else {}),
        **({"nan_truncated_queries": nan_cut_queries} if nan_cut_queries else {}),
        "tiebreak_source": "hit_tie" if want_tie else "hit_ids",
        "run_sha": run_sha,
        "merge_fold": sorted(_FOLD_USED),
        # The decode width this reduce actually used.
        "cpu_thread_count": pa.cpu_count(),
        "queries_short_of_k": short_count,
        # The phase split behind the `merge-bench` line, which used to live only
        # in the log. The writer adds this search's `elapsed_seconds` around it.
        "timing": {"io_wait_seconds": round(t_io, 2),
                   "fold_seconds": round(t_fold, 2),
                   "write_seconds": round(t_write, 2)},
        "corpus_dtype": carried_dtypes.get("corpus_dtype"),
        "queries_dtype": carried_dtypes.get("queries_dtype"),
        # Only when true, so a clean merge's manifest says nothing about it.
        **({"merge_forced": True} if forced or inputs_forced else {}),
    })
    return entry
