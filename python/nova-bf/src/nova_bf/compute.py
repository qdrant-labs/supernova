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
the final top-K. Large files can be scored in row-batches (`params.dense_batch_size`/
`sparse_batch_size`) to bound the per-file score matrix `(n_queries × rows)` on the GPU.

If a search's `filter` is set (see `nova_bf.filters`), each file's payload columns
are read alongside the vector column and evaluated into a keep-mask *before*
scoring, so filtered-out rows never reach the GPU. The mask compacts `arr` but
never renumbers rows — `orig_rows` tracks each surviving row's true file-row
number, since both `make_point_id` and the `id_column` lookup are keyed on it.
Mask evaluation is timed separately from the read (`filter_secs`, reported
alongside `read_secs`/`io_wait`/`gpu_secs`) so a slow filter is distinguishable
from slow IO in the end-of-run timing log.

`run_compute` runs every `SearchSpec` in `cfg.searches` — one or more
independent searches (own vector_type/metric/k/filter), e.g. dense-unfiltered
AND sparse-filtered against the same corpus in one invocation. Each corpus
file is read and decoded ONCE per vector_type any spec needs, not once per
spec. This is purely a sharing of REDUNDANT work, never a fusion of scores:
each search still gets its own independent ranked list, not a blended hybrid
ranking.

Per vector_type, EVERY search sharing that vector_type shares ONE batch grid
(see `run_compute`'s `has_baseline` and `_process_shared_batch`): the GPU
transfer/CSR build and each distinct metric's score matrix are computed ONCE
per batch, and every search merges its own top-K from those same columns,
masking them down to its own filter's surviving rows first if it has one.
Masking a raw, per-row-independent score matrix is exact, not an
approximation, so a filtered search's metric never needs to already be used
by anyone else — computing one more metric on a batch that's already
resident on the GPU is cheap, unlike a second transfer.

What rows make up that shared grid depends on whether any search of the
vector_type is unfiltered:

- If ANY search is unfiltered (or has an explicit-but-empty filter — see
  `_is_unfiltered`), the shared grid is the WHOLE file, uncompacted: an
  unfiltered search needs every row anyway, so every OTHER (filtered) search
  of the vector_type rides along for free, masking down to its own rows
  afterward.
- Otherwise (every search of the vector_type has an active filter — none
  unfiltered), the shared grid is the UNION of every DISTINCT active
  filter's surviving rows (`_union_keep`), compacted/transferred/scored
  ONCE, with each search then masking down further to its own filter's
  subset of that union. A per-query filter (see `_is_per_query`) has no
  SINGLE row-subset to offer here (different queries need different rows
  from the same batch) — it instead contributes a cheap, safe OVER-
  approximation ("does at least one query's own leaf admit this row" — Front
  B, see `_row_union_from_gpu_leaves`), so it still benefits from (and
  contributes to) compaction; its own FINE per-query masking still applies
  afterward, via `masked_fill`, same as when it rode the whole-file grid.
  This never does MORE row-scoring than treating each distinct filter
  independently would — in the worst case (fully disjoint filters) it's
  exactly the same total rows, just one transfer/launch instead of several —
  and strictly less whenever two filters' surviving rows overlap. The
  tradeoff: unlike treating each filter independently (which bounds each
  one's own transfer to its own, typically smaller, surviving-row count),
  the union's peak transfer size is bounded by `params.dense_batch_size`/
  `sparse_batch_size` alone when left at their None default — several large,
  mostly-disjoint filters (or one loose per-query over-approximation) can
  produce a union nearly the size of the whole file. Set that batch size
  explicitly if you have many such filters (see `ParamsConfig`).

Either way, `_process_shared_batch` does the per-search masking: an
unfiltered search uses the shared grid as-is; a UNIFORMLY filtered one
narrows it to its own filter's surviving COLUMNS (a gather), with two
searches sharing an identical filter memoizing that lookup once (keyed by
the `Filter` object itself, frozen+hashable) rather than repeating it. A
PER-QUERY filter (`match_from_query`/`range_from_query`/
`match_text_from_query` — see `nova_bf.filters`) can't be expressed as a
column gather, since different queries keep different rows from the SAME
columns: instead every column stays, and individual `(query, row)` cells get
invalidated to `-inf` via `masked_fill` before the per-query `torch.topk` —
needing no changes to the id-encoding scheme, since no column is ever
dropped. `nova_bf.filters.evaluate()`'s result is `(rows,)` for a uniform
filter (unchanged cost/shape from before per-query filters existed) or
`(n_queries, rows)` the instant any condition, in any of `must`/`should`/
`must_not`, is per-query.

A per-query filter's `(n_queries, rows)` `cell_mask` is built one of two
ways, decided once per distinct filter (`_gpu_eligible`): if every leaf is a
static condition, `match_from_query` (scalar or list/MatchAny), or
`range_from_query`, it's evaluated GPU-natively (`_gpu_evaluate`) straight
from small per-file corpus arrays (vocab codes or raw numeric values,
transferred to the GPU ONCE per file, then sliced per batch — see
`_corpus_leaf_array`/`_build_gpu_leaf_state`) and small per-query GPU state
built ONCE at setup — no `(n_queries, rows)` array is ever materialized on
the CPU or shipped over PCIe. A filter with a `match_text`/
`match_text_from_query` leaf anywhere falls back to `nova_bf.filters.
evaluate()`'s CPU/numpy path unchanged (torch has no string tensor type) —
its `(n_queries, rows)` result is the one CPU-fallback mask still held for a
whole file's batch loop, so it's bit-packed along the query axis
(`np.packbits`, 8 queries/byte) right after `evaluate()` returns, cutting
that footprint 8x; each batch slice unpacks only its own row-slice
(`np.unpackbits`) back to one bool per query, right before use.
"""

from __future__ import annotations

import logging
import os
import re
import time

from collections import Counter
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread

import numpy as np

from tqdm import tqdm

from nova_bf.config import BruteForceConfig, Filter, FilterCondition, SearchSpec
from nova_bf.filters import _condition_mask, _match_any_membership, evaluate
from nova_bf.ids import make_point_id
from nova_bf.io import ParquetFile, Store, dense_to_2d, sparse_to_coo_parts
from nova_bf.results import build_result_table, partial_dir, result_name, warn_if_short

logger = logging.getLogger(__name__)

PREFETCH_QUEUE_SIZE = 4
# Per-file id encoding: global_file_idx * MAX_ROWS_PER_FILE + row. Collision-free
# and reversible as long as no single file has more rows than this.
MAX_ROWS_PER_FILE = 100_000_000


def filter_corpus_files(
    files: list[ParquetFile], include: str | None, exclude: str | None
) -> list[ParquetFile]:
    """Keep only corpus files whose path matches `include` (if set) and does not
    match `exclude` (if set) — both are `re.search`-ed against the full path.
    `path` globs recursively, so this is how you skip unintended siblings (a
    `prepared/` folder, a staging dir) that share the prefix."""
    kept = files
    if include is not None:
        rx = re.compile(include)
        kept = [f for f in kept if rx.search(f.read_path)]
    if exclude is not None:
        rx = re.compile(exclude)
        kept = [f for f in kept if not rx.search(f.read_path)]
    dropped = len(files) - len(kept)
    if dropped:
        logger.info(
            "corpus filter: kept %d of %d files (dropped %d via include/exclude)",
            len(kept), len(files), dropped,
        )
    if files and not kept:
        logger.warning("corpus include/exclude removed ALL %d files — check the patterns", len(files))
    return kept


def _next_in_order(want_gidx: int, pending: dict[int, tuple], fetch) -> tuple:
    """Returns the `(gidx, ...)` item for `want_gidx`, buffering (in
    `pending`) any item `fetch()` delivers out of turn until it's needed.
    `fetch()` returns the next available item in ARRIVAL order, which may
    not be `want_gidx`'s turn yet — used so items produced by an unordered
    pool of worker threads can still be consumed in a fixed, deterministic
    order. `run_compute`'s consumer loop uses this to fold corpus files into
    the running top-K merge in ascending `gidx` order regardless of which
    reader thread's file arrives first — the merge is commutative in the
    SCORES it produces, but not in which of several EXACTLY tied candidates
    a run picks, so without this, that pick varied nondeterministically run
    to run for the identical corpus and queries."""
    if want_gidx in pending:
        return pending.pop(want_gidx)
    while True:
        item = fetch()
        if item[0] == want_gidx:
            return item
        pending[item[0]] = item


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


def _to_query_array(values: list) -> np.ndarray:
    """A per-query filter column's values -> a numpy array. Plain scalars
    (equality/range/text values) get numpy's own dtype inference (float64
    for numbers, object/unicode for strings) so `filters.py`'s broadcast
    comparisons work natively. A per-query MatchAny column (each row a
    list of acceptable values) instead gets an explicit `dtype=object`
    array with each list assigned as one element — `np.array([[...], [...]])`
    would otherwise try to interpret same-length inner lists as a 2D array,
    which is never what a per-query list-of-alternatives means.

    Scans every value (not just the first) to decide which encoding applies:
    a MatchAny column can legitimately have `None` (no restriction) for some
    query and a real list for another, and checking only `values[0]` would
    misclassify the whole column whenever THAT one row happens to be null —
    `np.array([None, [...], [...]])` raises `ValueError` (inhomogeneous
    shape) rather than producing the object array `filters.py` expects."""
    if any(isinstance(v, (list, tuple)) for v in values):
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        return arr
    return np.array(values)


def load_queries(
    store: Store, qcfg, filter_cols: list[str] = (),
) -> tuple[np.ndarray, list[str], dict[str, list], dict[str, np.ndarray]]:
    cols = [qcfg.dense_column]
    if qcfg.id_column:
        cols.append(qcfg.id_column)
    cols += [c for c in qcfg.payload_fields if c not in cols]
    cols += [c for c in filter_cols if c not in cols]

    embs: list[np.ndarray] = []
    ids: list[str] = []
    payload: dict[str, list] = {c: [] for c in qcfg.payload_fields}
    filter_vals: dict[str, list] = {c: [] for c in filter_cols}
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
        for c in filter_cols:
            filter_vals[c] += d[c]
    Q = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), np.float32)
    return Q, ids, payload, {c: _to_query_array(v) for c, v in filter_vals.items()}


def _build_query_vocab(indices: np.ndarray) -> np.ndarray:
    """Sorted distinct token ids used by the query set. A corpus token id
    absent from this vocab can never match any query's nonzero support, so
    dropping it later (see `_vocab_lookup`) is exact, not lossy.

    Deliberately NOT a dense `remap[token_id] -> compact_col` array: real
    hashed sparse schemes (e.g. fastembed's BM25) scatter token ids across the
    full uint32 range, so a dense array sized by `max(token_id)` can demand
    tens of GB for a handful of distinct tokens. `_vocab_lookup` does the same
    mapping via `np.searchsorted` against this sorted array instead."""
    return np.unique(indices)


def _vocab_lookup(vocab: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """`ids` -> compact column position in `vocab`, or -1 if not present."""
    if len(vocab) == 0:
        return np.full(len(ids), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(vocab, ids), len(vocab) - 1)
    return np.where(vocab[pos] == ids, pos, -1).astype(np.int64)


def _coalesce_by_row_col(
    row_ids: np.ndarray, col_ids: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge duplicate (row, col) pairs by SUMMING their values, returning
    arrays sorted by row then col with every (row, col) pair appearing at
    most once.

    A repeated sparse index within one row (e.g. a hash collision in a real
    hashed sparse embedder) must contribute the SUM of its values, not just
    the last one seen — the two properties this guarantees (sorted-by-row,
    distinct-per-row columns) are also exactly torch's CSR tensor invariants,
    so this is the one place both the "sum duplicates" correctness and the
    "valid CSR" requirement are satisfied together, instead of only sorting
    (which leaves duplicate columns per row — still invalid CSR, since
    `check_invariants=True` rejects it, and relies on undefined/backend-
    specific behavior to sum them at matmul time).

    `np.lexsort`'s LAST key is primary: `(col_ids, row_ids)` sorts by row
    first, then col within each row — the reverse of the argument order.
    """
    if len(row_ids) == 0:
        return row_ids, col_ids, values
    perm = np.lexsort((col_ids, row_ids))
    row_ids, col_ids, values = row_ids[perm], col_ids[perm], values[perm]
    is_new = np.empty(len(row_ids), dtype=bool)
    is_new[0] = True
    is_new[1:] = (row_ids[1:] != row_ids[:-1]) | (col_ids[1:] != col_ids[:-1])
    group = np.cumsum(is_new) - 1
    merged = np.zeros(int(group[-1]) + 1, dtype=np.float64)
    np.add.at(merged, group, values.astype(np.float64))
    return row_ids[is_new], col_ids[is_new], merged.astype(values.dtype)


def _sparse_rows_to_dense(row_offsets: np.ndarray, indices: np.ndarray, values: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """CSR parts (indices already all within `vocab`) -> dense (n_rows, len(vocab)).

    Uses `np.add.at` (accumulate), not a plain fancy-index assignment, so a row
    with a repeated token id sums its contributions instead of silently
    keeping only the last one written — matching the corpus side, where the
    same repeated-index case is summed via `_coalesce_by_row_col`.

    Uses `np.searchsorted` directly rather than `_vocab_lookup`: `vocab` is
    built from this exact `indices` array (`_build_query_vocab`), so every
    element is provably present — the -1-for-absent branch can never fire
    here, only in `_sparse_batch_to_csr`'s corpus-side lookup."""
    n_rows = len(row_offsets) - 1
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(row_offsets))
    cols = np.searchsorted(vocab, indices)
    Q = np.zeros((n_rows, len(vocab)), dtype=np.float32)
    np.add.at(Q, (row_ids, cols), values)
    return Q


def load_queries_sparse(
    store: Store, qcfg, filter_cols: list[str] = (),
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list], dict[str, np.ndarray]]:
    """Sparse analog of `load_queries`: reads the struct<indices,values> column
    from every query file, then densifies once over the query set's own
    vocabulary (see `_build_query_vocab`) — queries are few enough that a dense
    (n_q, vocab_size) matrix is cheap, same as loading Q fully upfront today."""
    cols = [qcfg.sparse_column]
    if qcfg.id_column:
        cols.append(qcfg.id_column)
    cols += [c for c in qcfg.payload_fields if c not in cols]
    cols += [c for c in filter_cols if c not in cols]

    counts_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    ids: list[str] = []
    payload: dict[str, list] = {c: [] for c in qcfg.payload_fields}
    filter_vals: dict[str, list] = {c: [] for c in filter_cols}
    for f in store.list_parquets():
        table = store.read_columns(f.read_path, cols)
        row_offsets, idx, val = sparse_to_coo_parts(table[qcfg.sparse_column])
        counts_parts.append(np.diff(row_offsets))
        indices_parts.append(idx)
        values_parts.append(val)
        d = table.to_pydict()
        n = len(row_offsets) - 1
        if qcfg.id_column:
            ids += [str(x) for x in d[qcfg.id_column]]
        else:
            ids += [make_point_id(f.key, r) for r in range(n)]
        for c in qcfg.payload_fields:
            payload[c] += d[c]
        for c in filter_cols:
            filter_vals[c] += d[c]

    indices = np.concatenate(indices_parts) if indices_parts else np.zeros(0, np.int64)
    values = np.concatenate(values_parts) if values_parts else np.zeros(0, np.float32)
    counts = np.concatenate(counts_parts) if counts_parts else np.zeros(0, np.int64)
    n_q = len(counts)
    row_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    vocab = _build_query_vocab(indices)
    Q = _sparse_rows_to_dense(row_offsets, indices, values, vocab)
    return Q, vocab, ids, payload, {c: _to_query_array(v) for c, v in filter_vals.items()}


def _compact_sparse_rows(
    row_offsets: np.ndarray, indices: np.ndarray, values: np.ndarray,
    norms: np.ndarray | None, keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Row-level filter compaction for sparse CSR parts — the nnz-granular
    analog of the dense path's `arr = arr[orig_rows]`. `keep` is a per-row mask
    (from `filters.evaluate`); dropped rows' nonzeros are dropped too, but
    surviving rows' TRUE file-row numbers are preserved via `orig_rows`, same
    invariant the dense path relies on for id resolution."""
    orig_rows = np.nonzero(keep)[0]
    nnz_keep = np.repeat(keep, np.diff(row_offsets))
    new_indices = indices[nnz_keep]
    new_values = values[nnz_keep]
    new_row_offsets = np.concatenate(([0], np.cumsum(np.diff(row_offsets)[orig_rows]))).astype(np.int64)
    new_norms = norms[orig_rows] if norms is not None else None
    return new_row_offsets, new_indices, new_values, new_norms, orig_rows


def _sparse_batch_to_csr(
    row_offsets: np.ndarray, indices: np.ndarray, values: np.ndarray,
    r0: int, r1: int, vocab: np.ndarray, device: str,
):
    """This batch's RAW (unnormalized) CSR rows, remapped into the query
    vocabulary (out-of-vocab entries dropped — see `_build_query_vocab`).

    Deliberately takes no `norms`/metric argument and never scales `b_val` —
    this builder is shared across every search of this vector_type that scores
    the same rows via `_process_shared_batch`, including a mix of `cosine` and
    `dot` searches, so it must stay metric-agnostic BY CONSTRUCTION. Cosine
    normalization is applied by the caller as a post-hoc divide on the score
    matrix (`raw / row_norms`, mathematically identical to pre-scaling these
    values by `1/row_norm` before the matmul, since a per-row scalar commutes
    with it) — never inside this function. A `norms` parameter here once let a
    run-wide "does ANY search need cosine" flag silently corrupt a co-scheduled
    `dot` search's scores (see `test_sparse_dot_spec_not_corrupted_by_
    coscheduled_cosine_spec`); removing the parameter entirely, rather than
    just remembering not to pass it, is what prevents that class of bug from
    coming back.

    `check_invariants=False` below skips torch's own validation that each
    row's column indices are sorted and distinct — a real, enforced CSR
    invariant (violating it is undefined behavior per torch's own sparse
    tensor docs, not just a missed optimization). The source parquet's
    per-row token order isn't guaranteed sorted OR unique (two original
    token ids can remap to the same vocab column — impossible here since
    `_vocab_lookup` is injective, but a row can also carry the same raw
    token id twice, e.g. a hash collision in a real hashed embedder), so
    `_coalesce_by_row_col` both sorts AND merges duplicate columns before
    construction, so skipping torch's redundant check stays safe."""
    import torch

    lo, hi = int(row_offsets[r0]), int(row_offsets[r1])
    b_idx = indices[lo:hi]
    b_val = values[lo:hi]
    counts = np.diff(row_offsets[r0 : r1 + 1])

    row_ids = np.repeat(np.arange(r1 - r0, dtype=np.int64), counts)
    b_idx = _vocab_lookup(vocab, b_idx)
    keep_nnz = b_idx >= 0
    b_idx, b_val, row_ids = b_idx[keep_nnz], b_val[keep_nnz], row_ids[keep_nnz]

    row_ids, b_idx, b_val = _coalesce_by_row_col(row_ids, b_idx, b_val)
    new_counts = np.bincount(row_ids, minlength=r1 - r0)
    crow = np.concatenate(([0], np.cumsum(new_counts))).astype(np.int64)

    Cb = torch.sparse_csr_tensor(
        torch.from_numpy(crow), torch.from_numpy(b_idx), torch.from_numpy(b_val),
        size=(r1 - r0, len(vocab)), check_invariants=False,
    )
    return Cb.to(device, non_blocking=True)


def _sparse_scores(Q, Cb):
    """`Cb` (RAW, unnormalized sparse CSR, batch × vocab) against dense `Q`
    (n_q × vocab) via the documented, stable `sparse_csr @ dense -> dense` path
    (cuSPARSE SpMM) — deliberately not sparse-sparse matmul, which torch has no
    stable binding for. Returns the raw dot-product score matrix; the caller
    applies any metric-specific transform (e.g. dividing by each row's L2 norm
    for cosine) afterward — see `SparseBatchSlice.score`, which shares one
    `Cb` across every search scoring it regardless of metric, so this
    function itself must stay metric-agnostic."""
    import torch

    return torch.matmul(Cb, Q.T).T


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


def _sparse_file_norms(row_offsets: np.ndarray, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each row's true (untruncated) L2 norm, over the FULL row before any
    query-vocab truncation or filtering — see `_sparse_batch_to_csr`. A row's
    true value at a given token id is the SUM of every occurrence of that id
    (a repeated raw index — e.g. a hash collision — isn't two separate
    dimensions), so duplicates must be coalesced before squaring:
    sum-of-squares-of-parts is not the same as square-of-the-summed-value
    whenever a row repeats a token id. Only computed when some spec in the
    run needs cosine similarity for sparse vectors."""
    n_rows = len(row_offsets) - 1
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(row_offsets))
    m_rows, _, m_vals = _coalesce_by_row_col(row_ids, indices, values)  # indices already int64
    sumsq = np.bincount(m_rows, weights=m_vals.astype(np.float64) ** 2, minlength=n_rows)
    return np.sqrt(sumsq).astype(np.float32)


class DenseCorpusBatch:
    """Decoded dense corpus vectors for one file: `(n_rows, dim)`. Exposes the
    same `.n_rows`/`.nbytes`/`.compact`/`.transfer` surface as
    `SparseCorpusBatch` so `run_compute`'s per-file loop never branches on
    vector_type."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    @property
    def n_rows(self) -> int:
        return self.arr.shape[0]

    @property
    def nbytes(self) -> int:
        return int(self.arr.nbytes)

    def compact(self, keep: np.ndarray) -> tuple["DenseCorpusBatch", np.ndarray]:
        """`keep` is a per-row mask; returns the compacted batch plus each
        surviving row's TRUE file-row number (`orig_rows`) — both
        `make_point_id` and an `id_column` lookup are keyed on that true row,
        not on position in the compacted array."""
        orig_rows = np.nonzero(keep)[0]
        return DenseCorpusBatch(self.arr[orig_rows]), orig_rows

    def transfer(self, r0: int, r1: int, device: str) -> "DenseBatchSlice":
        import torch

        Cb = torch.from_numpy(self.arr[r0:r1]).to(device, non_blocking=True)
        return DenseBatchSlice(Cb)


@dataclass
class DenseBatchSlice:
    Cb: object  # torch.Tensor, (n_rows, dim)

    @property
    def n_rows(self) -> int:
        return self.Cb.shape[0]

    def score(self, Q, metric: str):
        return _scores(Q, self.Cb, metric)


class SparseCorpusBatch:
    """Decoded sparse corpus CSR parts for one file, plus each row's true
    (untruncated) L2 norm (`norms`, `None` unless some spec needs cosine —
    see `_sparse_file_norms`). `vocab`/`need_row_norms` are fixed RUN-WIDE
    (see `run_compute`'s `need_sparse_norms`) and carried through `.compact()`
    unchanged, so `.transfer()` needs no extra args beyond the `(r0, r1,
    device)` every corpus batch takes. This means a batch/filter-group made
    up entirely of `dot` searches still moves `row_norms` to the GPU whenever
    ANY search in the run needs cosine — a negligible extra transfer (one
    float per row in the batch) that a `dot` search's own `.score()` never
    reads, so it costs a little bandwidth, never correctness."""

    def __init__(self, row_offsets, indices, values, norms, vocab: np.ndarray, need_row_norms: bool):
        self.row_offsets = row_offsets
        self.indices = indices
        self.values = values
        self.norms = norms
        self.vocab = vocab
        self.need_row_norms = need_row_norms

    @property
    def n_rows(self) -> int:
        return len(self.row_offsets) - 1

    @property
    def nbytes(self) -> int:
        # on-disk width (uint32 index + float32 value per nnz), not the
        # in-memory int64 index array torch requires — keeps this comparable
        # to the dense path's wire-byte estimate.
        return len(self.indices) * 8

    def compact(self, keep: np.ndarray) -> tuple["SparseCorpusBatch", np.ndarray]:
        row_offsets, indices, values, norms, orig_rows = _compact_sparse_rows(
            self.row_offsets, self.indices, self.values, self.norms, keep
        )
        return (
            SparseCorpusBatch(row_offsets, indices, values, norms, self.vocab, self.need_row_norms),
            orig_rows,
        )

    def transfer(self, r0: int, r1: int, device: str) -> "SparseBatchSlice":
        import torch

        Cb = _sparse_batch_to_csr(self.row_offsets, self.indices, self.values, r0, r1, self.vocab, device)
        row_norms = (
            torch.from_numpy(self.norms[r0:r1]).to(device, non_blocking=True)
            if self.need_row_norms else None
        )
        return SparseBatchSlice(Cb, row_norms)


@dataclass
class SparseBatchSlice:
    Cb: object  # torch.Tensor, sparse CSR (n_rows, vocab)
    row_norms: object  # torch.Tensor | None

    @property
    def n_rows(self) -> int:
        return self.Cb.shape[0]

    def score(self, Q, metric: str):
        raw = _sparse_scores(Q, self.Cb)
        if metric == "cosine":
            return raw / self.row_norms.clamp_min(1e-12)[None, :]
        return raw


def _merge_topk(top_scores, top_enc, batch_scores, batch_rows, gidx: int, k: int):
    """Fold one batch's `(n_q, n_candidate_rows)` score matrix into a spec's
    running `(top_scores, top_enc)` top-k state: top-k the batch on its own
    (bounded by however many candidate rows it actually has), append to the
    running state, and re-top-k down to `k`. `batch_rows` must be the TRUE
    file row for each of `batch_scores`'s columns, so the encoding stays
    `global_file_idx * MAX_ROWS_PER_FILE + row` regardless of batching,
    compaction, or masking."""
    import torch

    bk = min(k, batch_scores.shape[1])
    f_scores, f_local = torch.topk(batch_scores, k=bk, dim=1)
    f_enc = (gidx * MAX_ROWS_PER_FILE + batch_rows)[f_local]
    merged_s = torch.cat([top_scores, f_scores], dim=1)
    merged_e = torch.cat([top_enc, f_enc], dim=1)
    new_top_scores, idx = torch.topk(merged_s, k=k, dim=1)
    return new_top_scores, merged_e.gather(1, idx)


def _resolve_vt_batch_size(configured: int | None, k_floor: int, vt: str) -> int | None:
    """`configured` (`params.dense_batch_size`/`sparse_batch_size`) is fine
    as-is whenever it's unset or already at/above `k_floor` (the largest `k`
    among EVERY search of this vector_type) — a batch that size can already
    fill any of these searches' own top-K in one pass. Below that, never
    raise it: every search of the vector_type shares one batch grid
    regardless of filter (see `run_compute`'s `has_baseline` and
    `_union_keep`), so raising `configured` for one search's large `k` would
    silently blow past a DIFFERENT, unrelated search's own memory bound —
    exactly the OOM footgun `dense_batch_size`/`sparse_batch_size` exists to
    prevent. Warn instead: the larger-`k` search just takes extra merge
    rounds to fill its own top-K (more of them the further `configured` sits
    below `k_floor`), at no extra GPU memory cost to anyone."""
    if configured is None or configured >= k_floor:
        return configured
    logger.warning(
        "params.%s_batch_size=%d is below k=%d, the largest among searches "
        "sharing this vector_type's batch pass — keeping your configured "
        "value (it's a memory bound, so it takes priority); the larger-k "
        "search(es) will need extra merge rounds to fill their own top-K "
        "instead (more of them the further below k=%d your batch size is), "
        "at no extra GPU memory cost.",
        vt, configured, k_floor, k_floor,
    )
    return configured


def _pack_query_axis(mask: np.ndarray) -> np.ndarray:
    """Bit-pack a `(n_queries, rows)` boolean mask along the query axis (8
    queries/byte, `np.packbits` default `bitorder="big"`) — shrinks the one
    CPU-fallback per-query mask (a filter with a `match_text`/
    `match_text_from_query` leaf, ineligible for Front A's GPU-native path)
    still held on the CPU for a whole file's batch loop, 8x. Rows aren't the
    packed axis, so slicing by `true_rows` (`keeps[f][:, true_rows]`) stays a
    plain column slice — no unpacking needed just to select rows. Inverse:
    `_unpack_query_axis`."""
    return np.packbits(mask, axis=0)


def _unpack_query_axis(packed: np.ndarray, n_queries: int) -> np.ndarray:
    """Inverse of `_pack_query_axis`: expand a packed `(ceil(n_queries / 8),
    rows)` byte array back to one `bool` per query, `(n_queries, rows)`.
    `count=n_queries` trims the padding bits `packbits` adds when
    `n_queries` isn't a multiple of 8 — without it, the result would carry
    up to 7 extra all-`False` phantom queries, mismatching every real
    per-query tensor it's later combined with. `unpackbits` itself returns
    `uint8` 0/1, not `bool` — cast explicitly, since `~` on a `uint8` tensor
    flips all 8 bits (`0 -> 255`) rather than negating logically."""
    return np.unpackbits(packed, axis=0, count=n_queries).astype(bool)


def _union_keep(filters: list[Filter], keeps: dict[Filter | None, np.ndarray | None]) -> np.ndarray:
    """OR-reduce of every DISTINCT active filter's keep-mask in `filters` —
    the shared row-set for a vector_type where no search is unfiltered (see
    `run_compute`). Never called with `None` in `filters` (that's the
    `has_baseline` case, handled by leaving the whole file uncompacted
    instead), so every `keeps[f]` here is a real `np.ndarray`, not `None`.
    `filters` is never empty (a vt only reaches this function once
    `vt_spec_idxs` has established it has at least one spec, each with a
    real filter).

    A UNIFORM filter's (or a GPU-eligible per-query filter's — Front B, see
    `_row_union_from_gpu_leaves`) `keeps[f]` is already `(rows,)`, used as
    -is. A CPU-fallback per-query filter's (`match_text`/
    `match_text_from_query` leaf present) is bit-packed along the query axis,
    `(ceil(n_queries / 8), rows)` (see `run_compute`'s reader thread) —
    reduced to `(rows,)` via `.any(axis=0)` first: still EXACT, not a
    heuristic, since a packed byte is 0 iff every query bit it holds is 0, so
    byte-truthiness IS "does any query this byte covers want this row"; OR-ing
    that across bytes is exactly "does any query want this row", simply read
    off the packed array rather than approximated."""
    parts = [m.any(axis=0) if m.ndim == 2 else m for m in (keeps[f] for f in filters)]
    return np.logical_or.reduce(parts)


def _process_batch_group(
    batch, member_idxs: list[int], specs: list[SearchSpec], spec_Q, spec_top_scores, spec_top_enc,
    batch_size: int | None, gidx: int, device: str, orig_rows: np.ndarray | None, select,
) -> float:
    """The shared per-vector_type primitive behind `_process_shared_batch`:
    iterate `batch` in `batch_size`-row slices, transfer each slice once
    (`batch.transfer`), score it once per
    DISTINCT metric among `member_idxs` (`score_cache` — every member needing
    that metric reads the same tensor), and merge each member's own top-k
    from those shared columns via `_merge_topk`.

    `orig_rows` maps a slice position to its TRUE file-row number: `None`
    means position IS the true row — `batch` is the raw, whole file, because
    some search of this vector_type is unfiltered and needs every row; an
    array means `batch` was already compacted by the caller (to the union of
    every active filter's surviving rows — see `run_compute`'s `has_baseline`
    and `_union_keep`) and this maps back to true rows for `_merge_topk`'s
    encoding.

    `select(m, rows, true_rows, cache) -> (sel_rows, sel_cols, cell_mask)` is
    the per-member filtering strategy (see `_process_shared_batch`):
    `sel_rows is None` skips the merge entirely for this member/slice (e.g.
    a filter keeping zero rows here); `sel_cols` is either `None` (member is
    unfiltered or per-query-filtered — use the slice's rows unchanged) or a
    column-index tensor used to mask both the score matrix and `rows` down
    to that member's own (uniform) filter's surviving columns; `cell_mask`
    is either `None` (no per-(query,row) masking needed) or a `(n_queries,
    len(rows))` boolean tensor applied via `masked_fill` — a per-query
    filter's own rows vary BY QUERY, so unlike a uniform filter it can't be
    expressed as one shared column selection; every column stays, and
    individual (query, row) cells get invalidated instead. `true_rows`
    indexes a filter's keep-mask (sized to the whole file, not to this
    possibly-already-compacted batch): a plain `slice(r0, r1)` when
    `orig_rows is None` (position IS the true row — a cheap view, no copy),
    or `orig_rows`'s corresponding array slice otherwise. `cache` is a fresh
    dict per r0-slice for `select` to memoize per-filter lookups shared
    across members — it does not persist across slices, since the mask is
    slice-relative.

    Returns elapsed wall-clock seconds spent in this loop (folded into the
    caller's `gpu_secs`). A caller that pre-compacts `batch` must do so
    BEFORE calling this function — its own timer starts only once this loop
    begins, so CPU-side compaction time never counts as GPU time (see the
    `io_wait`/`gpu_secs` split docs at the top of this module)."""
    import torch

    n_rows = batch.n_rows
    if n_rows == 0:
        return 0.0
    t0 = time.perf_counter()
    step = batch_size or n_rows
    for r0 in range(0, n_rows, step):
        r1 = min(r0 + step, n_rows)
        sl = batch.transfer(r0, r1, device)
        if orig_rows is None:
            # No CPU round-trip: build the row-index tensor directly on
            # device, and use a plain slice (a view, not a gather-copy) to
            # index a filter's keep-mask below — position already IS the
            # true row here, so there's nothing to remap.
            rows = torch.arange(r0, r0 + sl.n_rows, dtype=torch.int64, device=device)
            true_rows = slice(r0, r0 + sl.n_rows)
        else:
            true_rows = orig_rows[r0 : r0 + sl.n_rows]
            rows = torch.from_numpy(true_rows).to(device, non_blocking=True)

        score_cache: dict[str, object] = {}
        cache: dict[object, object] = {}  # keyed by whatever select() memoizes on (e.g. Filter)
        for m in member_idxs:
            s = specs[m]
            if s.metric not in score_cache:
                score_cache[s.metric] = sl.score(spec_Q[m], s.metric)
            scores = score_cache[s.metric]

            sel_rows, sel_cols, cell_mask = select(m, rows, true_rows, cache)
            if sel_rows is None:
                continue
            sel_scores = scores if sel_cols is None else scores[:, sel_cols]
            if cell_mask is not None:
                sel_scores = sel_scores.masked_fill(~cell_mask, float("-inf"))

            spec_top_scores[m], spec_top_enc[m] = _merge_topk(
                spec_top_scores[m], spec_top_enc[m], sel_scores, sel_rows, gidx, s.k
            )
    return time.perf_counter() - t0


def _is_unfiltered(f: Filter | None) -> bool:
    """`None` is the common no-filter case; an explicit-but-empty `filter: {}`
    (`Filter(must=(), should=(), must_not=())`) is semantically the same —
    `evaluate()` keeps every row either way — so both must be treated
    identically wherever "does this spec have an active filter" decides
    `run_compute`'s `has_baseline` or `_process_shared_batch`'s per-member
    masking. `f.fields()` is empty iff every condition group is empty,
    without duplicating that check against `Filter`'s own fields."""
    return f is None or not f.fields()


def _is_per_query(f: Filter | None) -> bool:
    """Does `f` have any per-query condition (`match_from_query`/
    `range_from_query`/`match_text_from_query`) anywhere? A per-query
    filter has no single EXACT row-subset to offer a shared batch grid
    (different queries need different corpus rows from the same batch), so
    it no longer forces `run_compute`'s `has_baseline` by itself (see
    `_row_union_from_gpu_leaves`/`_union_keep` — Front B): it instead
    contributes a cheap, safe over-approximation to union-compaction like
    any other filter. Its FINE per-query masking still applies afterward —
    `_process_shared_batch`'s `select` masks it via `cell_mask` (a full
    `(n_queries, rows)` invalidation) rather than a column gather."""
    return f is not None and f.is_per_query()


def _gpu_eligible(f: Filter | None) -> bool:
    """Whether `f`'s per-query evaluation can run GPU-natively (Front A — see
    this module's docstring) instead of `filters.py`'s CPU/numpy path:
    needs at least one per-query leaf (a purely uniform filter already gets
    the cheap column-gather path — see `_process_shared_batch` — so there's
    nothing to speed up here) and no `match_text`/`match_text_from_query`
    leaf ANYWHERE in the filter — torch has no string tensor type, so
    regex/substring matching stays CPU-only regardless of what else is in
    the filter (see `filters._match_text_from_query_mask`'s own word-level
    dedup instead)."""
    if f is None or not _is_per_query(f):
        return False
    return not any(
        cond.match_text is not None or cond.match_text_from_query is not None
        for cond in f.all_conditions()
    )


def _not_null_mask(vals: np.ndarray) -> np.ndarray:
    """Which entries of `vals` are non-null: `None` (object-dtype arrays) or
    `nan` (any float dtype) both mean "no restriction"/"never matches" —
    same convention `filters.py`'s per-query masks already use. A plain
    int64/bool array has no null representation at all, so every entry
    counts as non-null."""
    if vals.dtype == object:
        return np.array([v is not None and v == v for v in vals], dtype=bool)
    if vals.dtype.kind == "f":
        return ~np.isnan(vals)
    return np.ones(len(vals), dtype=bool)


def _is_null_scalar(v) -> bool:
    """One MatchAny list ELEMENT is null (`None`, or float `nan`) — same
    "no restriction" convention `_not_null_mask` applies to a whole array,
    used here to drop a null nested INSIDE a query's own list (as opposed
    to the list itself being `None`) before vocab-building — `np.unique`
    can't sort `None`/`nan` against a real value any more than it can for a
    whole column (see `_encode_against_vocab`)."""
    return v is None or (isinstance(v, float) and v != v)


def _encode_against_vocab(vocab: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """`vals` -> position in `vocab`, or -1 for null or absent-from-vocab —
    the corpus-side half of Front A's `match_from_query` GPU path (see
    `_corpus_leaf_array`). Excludes nulls from the `searchsorted` call
    itself, same reason `filters._match_any_from_query_mask` does: `vocab`'s
    dtype has no defined ordering against `None`/`nan` mixed into an
    object array."""
    not_null = _not_null_mask(vals)
    codes = np.full(len(vals), -1, dtype=np.int64)
    if not_null.any():
        codes[not_null] = _vocab_lookup(vocab, vals[not_null])
    return codes


def _safe_extreme(vals: np.ndarray, kind: str) -> float | None:
    """`min`/`max` of `vals`, excluding null entries — `None` if every entry
    is null (nothing to extremize). Used by `_row_union_from_gpu_leaves` so
    one query's null per-query bound (which never matches for THAT query,
    see `_range_from_query_mask`) can't poison a plain `.min()`/`.max()`
    with `nan`."""
    not_null = _not_null_mask(vals)
    if not not_null.any():
        return None
    return float(vals[not_null].min() if kind == "min" else vals[not_null].max())


def _row_union_from_gpu_leaves(
    f: Filter, leaf_arrays: dict[FilterCondition, np.ndarray],
    query_filter_vals: dict[str, np.ndarray], n_rows: int,
) -> np.ndarray | None:
    """`(rows,)` SAFE OVER-APPROXIMATION of "does at least one query's
    per-query leaf admit this row" for a GPU-eligible per-query filter
    (Front B — see `run_compute`'s `has_baseline`/`vt_union_filters`), or
    `None` if no such leaf exists to build one from (caller substitutes
    all-`True`: no tightening possible, but always correct).

    Built ONLY from `f.must`'s `match_from_query`/`range_from_query` leaves
    — deliberately NEVER `should` or `must_not`: a `must` leaf is a
    NECESSARY condition for the row regardless of anything else in the
    filter, so whichever query actually wants a row necessarily satisfies
    THAT query's own version of every one of these leaves — ORing each
    leaf's "some query's own value admits this row" mask together is
    therefore provably a superset of the true "some query wants this row"
    set, however many `must` conditions (per-query or static) there are.
    `should`/`must_not` don't have this property: a `should` branch's
    satisfaction doesn't require any ONE specific condition to hold (a row
    could be admitted via a totally different, possibly static, branch),
    and a `must_not` leaf's own predicate holding is what EXCLUDES a row,
    not what admits one — using either here could silently exclude a row
    some query genuinely wants, so a filter whose only per-query leaf(s)
    live outside `must` gets no tightening at all here (falls back to
    all-`True` via the `None` return), never a WRONG one.

    `match_from_query`'s test (`arr != -1`) is the same expression whether
    the leaf is scalar or MatchAny: either way, `vocab` (see
    `_build_gpu_leaf_state`) is built from exactly the values SOME query's
    leaf references, so a corpus value present in it (a non-`-1` code) is
    necessarily wanted by whichever query(ies) contributed that value.
    `range_from_query` unions each configured bound independently across
    every query (the loosest — min for a lower bound, max for an upper
    one) rather than per-query per-bound; combining a query's OWN gt/lt
    pair would be tighter, but this cheaper cross-query union is still a
    valid superset (each query's own bound is at least as tight as the
    extreme, so a row a query's own bounds admit is also admitted by the
    extreme)."""
    mask = None
    for cond in f.must:
        if cond.match_from_query is not None:
            part = leaf_arrays[cond] != -1
        elif cond.range_from_query is not None:
            arr = leaf_arrays[cond]
            r = cond.range_from_query
            part = np.ones(n_rows, dtype=bool)
            if r.gt is not None:
                lo = _safe_extreme(query_filter_vals[r.gt], "min")
                if lo is not None:
                    part &= arr > lo
            if r.gte is not None:
                lo = _safe_extreme(query_filter_vals[r.gte], "min")
                if lo is not None:
                    part &= arr >= lo
            if r.lt is not None:
                hi = _safe_extreme(query_filter_vals[r.lt], "max")
                if hi is not None:
                    part &= arr < hi
            if r.lte is not None:
                hi = _safe_extreme(query_filter_vals[r.lte], "max")
                if hi is not None:
                    part &= arr <= hi
        else:
            continue
        mask = part if mask is None else (mask | part)
    return mask


def _build_gpu_leaf_state(
    cond: FilterCondition, query_filter_vals: dict[str, np.ndarray], device: str,
) -> tuple[object, np.ndarray | None]:
    """Setup-time (once per DISTINCT `match_from_query`/`range_from_query`
    `FilterCondition` across every GPU-eligible filter in the run — see
    `run_compute` — NOT per `Filter` and NOT per file): builds and transfers
    this ONE leaf's small per-query GPU state — vocab-encoded query codes, a
    MatchAny membership matrix, or per-bound numeric arrays — the per-query
    analog of `Q_gpu_by_vt`'s one-time transfer. Keying by the condition
    itself (not by the enclosing filter) means two different filters that
    happen to share an identical per-query leaf build and transfer it only
    ONCE, same sharing `leaf_arrays`/`leaf_gpu` already give the corpus side.

    Returns `(query_gpu_entry, vocab_or_none)`: a `match_from_query` entry is
    `("scalar", codes_gpu)` or `("list", membership_gpu)`, paired with the
    vocab needed again per file to encode the CORPUS side against the SAME
    vocab (see `_corpus_leaf_array`) — a corpus value never seen in that
    vocab can't match any query's `match_from_query` leaf, by construction.
    A `range_from_query` entry is a `{bound_name: qbounds_gpu}` dict, with no
    vocab (`None`) since it needs no factorization."""
    import torch

    if cond.match_from_query is not None:
        qvals = query_filter_vals[cond.match_from_query]
        is_list = any(isinstance(v, (list, tuple, np.ndarray)) for v in qvals)
        if is_list:
            # Vocab = union of every query's OWN list values (not the
            # corpus's distinct values) — a corpus value absent from every
            # query's list can never match anyone, so mapping it to -1 below
            # (see _gpu_cond_mask's `valid` masking) is exact, not lossy.
            # Null elements NESTED INSIDE a list (as opposed to the list
            # itself being None) are dropped before np.unique, same reason
            # _encode_against_vocab excludes nulls from its own vocab lookup.
            flat = [
                v for lst in qvals if lst is not None
                for v in lst if not _is_null_scalar(v)
            ]
            vocab = np.unique(np.asarray(flat)) if flat else np.empty(0)
            membership = _match_any_membership(vocab, qvals)
            return ("list", torch.from_numpy(membership).to(device, non_blocking=True)), vocab
        not_null = _not_null_mask(qvals)
        vocab = np.unique(qvals[not_null]) if not_null.any() else np.empty(0, dtype=qvals.dtype)
        # -2: a null per-query value never matches anything — kept distinct
        # from -1 (corpus null/absent-from-vocab) so the two sentinels can
        # never accidentally compare equal.
        codes = np.full(len(qvals), -2, dtype=np.int64)
        if not_null.any():
            codes[not_null] = _vocab_lookup(vocab, qvals[not_null])
        return ("scalar", torch.from_numpy(codes).to(device, non_blocking=True)), vocab

    r = cond.range_from_query
    bounds: dict[str, object] = {}
    for name in ("gt", "gte", "lt", "lte"):
        colname = getattr(r, name)
        if colname is None:
            continue
        qb = query_filter_vals[colname].astype(np.float64, copy=False)
        bounds[name] = torch.from_numpy(qb).to(device, non_blocking=True)
    return bounds, None


def _corpus_leaf_array(cond: FilterCondition, table, vocab_for_cond: np.ndarray | None) -> np.ndarray:
    """CPU-side, once per file (see `run_compute`'s reader thread): this
    condition's corpus-side array for GPU-native evaluation (Front A) —
    `(rows,)` int64 vocab codes (-1 for null/absent) for a `match_from_query`
    leaf (scalar OR MatchAny — both compare/gather against one corpus scalar
    per row), `(rows,)` float64 raw values for `range_from_query` (a null
    row becomes NaN, which already compares False against every bound — see
    `_gpu_cond_mask`), or `(rows,)` boolean — this file's already-evaluated
    result — for a static `match`/`range` leaf, reusing
    `filters._condition_mask` UNCHANGED so the leaves Front A doesn't need
    to move at all can never diverge from the CPU reference path."""
    if cond.match_from_query is not None:
        corpus_vals = table[cond.field].to_numpy(zero_copy_only=False)
        return _encode_against_vocab(vocab_for_cond, corpus_vals)
    if cond.range_from_query is not None:
        return table[cond.field].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    return _condition_mask(cond, table)


def _gpu_cond_mask(cond: FilterCondition, leaf_gpu: dict, rows, query_gpu: dict):
    """torch-native per-condition mask for one GPU-eligible leaf — the Front
    A analog of `filters._condition_mask`. `leaf_gpu[cond]` is this
    condition's corpus-side array (built once per file, transferred once —
    see `run_compute`); indexing it by `rows` (this slice's true row
    positions, already a GPU `LongTensor` — the SAME tensor `_merge_topk`'s
    id-encoding uses) narrows it to the current batch slice with no extra
    host transfer or `true_rows` bookkeeping needed here. `query_gpu[cond]`
    is this condition's per-query GPU state, built once at setup (see
    `_build_gpu_leaf_state`)."""
    import torch

    arr = leaf_gpu[cond][rows]
    if cond.match_from_query is not None:
        kind, qstate = query_gpu[cond]
        if kind == "scalar":
            return qstate[:, None] == arr[None, :]
        # MatchAny: qstate is the (n_queries, n_distinct) membership matrix;
        # arr is this slice's corpus vocab codes (-1 for null/absent). An
        # empty vocab (every query's list was null/empty) means n_distinct
        # is 0 — nothing to gather, and no valid corpus code can exist
        # either, so short-circuit before indexing a zero-width dimension
        # (arr.clamp(min=0) would otherwise still try to gather column 0
        # from it and raise IndexError).
        if qstate.shape[1] == 0:
            return torch.zeros((qstate.shape[0], arr.shape[0]), dtype=torch.bool, device=arr.device)
        # Clamp before gathering to avoid a -1 (null/absent) code wrapping
        # to the LAST column, then explicitly zero those columns out,
        # mirroring filters._match_any_from_query_mask's `valid_cols`
        # handling.
        valid = arr >= 0
        gathered = qstate[:, arr.clamp(min=0)]
        return gathered & valid[None, :]
    if cond.range_from_query is not None:
        ops = {"gt": torch.gt, "gte": torch.ge, "lt": torch.lt, "lte": torch.le}
        mask = None
        for name, qbounds in query_gpu[cond].items():
            part = ops[name](arr[None, :], qbounds[:, None])
            mask = part if mask is None else (mask & part)
        return mask
    # Static match/range: leaf_gpu[cond] is already the (rows,) boolean
    # result of filters._condition_mask, computed once per file — no
    # per-query broadcast needed for this leaf.
    return arr


def _gpu_evaluate(f: Filter, leaf_gpu: dict, rows, query_gpu: dict, device: str):
    """torch-native analog of `filters.evaluate()`, for a GPU-eligible
    per-query filter (see `_gpu_eligible`) — identical must/should/must_not
    combination logic, operating on GPU-resident tensors so a batch slice
    never needs a host round trip. Always returns a `(n_queries,
    len(rows))` bool tensor: eligibility requires at least one per-query
    leaf, so unlike `filters.evaluate()` there's no plain-`(rows,)` case to
    handle here."""
    import torch

    n = rows.numel()
    keep = torch.ones(n, dtype=torch.bool, device=device)
    for cond in f.must:
        keep = keep & _gpu_cond_mask(cond, leaf_gpu, rows, query_gpu)
    if f.should:
        any_match = torch.zeros(n, dtype=torch.bool, device=device)
        for cond in f.should:
            any_match = any_match | _gpu_cond_mask(cond, leaf_gpu, rows, query_gpu)
        keep = keep & any_match
    for cond in f.must_not:
        keep = keep & ~_gpu_cond_mask(cond, leaf_gpu, rows, query_gpu)
    return keep


def _process_shared_batch(
    batch, member_idxs: list[int], specs: list[SearchSpec], spec_Q, spec_top_scores, spec_top_enc,
    keeps: dict[Filter | None, np.ndarray | None], filter_is_per_query: dict[Filter | None, bool],
    filter_is_gpu_eligible: dict[Filter | None, bool], leaf_gpu: dict[FilterCondition, object],
    query_gpu: dict[FilterCondition, object], filter_share_count: dict[Filter | None, int],
    batch_size: int | None, gidx: int, device: str, orig_rows: np.ndarray | None,
) -> float:
    """Every search of this vector_type shares one batch grid: `orig_rows`
    is `None` when some search is unfiltered (`batch` is the raw, whole
    file — see `run_compute`'s `has_baseline`), or the true-row map produced
    by compacting `batch` to the union of every active filter's surviving
    rows otherwise (see `_union_keep` — a per-query filter contributes its
    own safe OVER-approximation to that union, Front B, rather than an
    exact row-subset). Three cases per member, decided by
    `filter_is_gpu_eligible[s.filter]` then `filter_is_per_query[s.filter]`:

    - Unfiltered: use the shared slice as-is.
    - Per-query filter (either GPU-eligible via Front A, or the CPU fallback
      for a `match_text`/`match_text_from_query` leaf — see `_gpu_eligible`):
      builds this member's `(n_queries, batch_rows)` `cell_mask` — from
      GPU-resident tensors via `_gpu_evaluate` (`leaf_gpu`/`query_gpu`, no
      CPU-side mask, no per-slice host transfer) or from `keeps[s.filter]`
      (computed once per file in `filters.evaluate()`) — cached per filter
      in `cache` only when `filter_share_count[s.filter] > 1`: a filter used
      by exactly one spec has nothing to share, so skipping the cache entry
      lets that tensor be collected as soon as this member's `sel_scores`
      is built instead of living until the end of this r0-slice's member
      loop. Every query needs a potentially different row-subset here, so
      there's no shared column selection to make — instead every column
      stays, and `cell_mask` gets applied via `masked_fill` in
      `_process_batch_group`.
    - Uniform filter: mask down to its own filter's surviving COLUMNS, via a
      `local_idx` cached per filter (in `_process_batch_group`'s per-slice
      `cache`) so two members sharing an identical filter don't recompute
      `nonzero` twice — indexing `keeps[s.filter]` by `true_rows` (each
      slice position's TRUE file row), not by position, since the shared
      batch may already be a compacted subset of the file.

    See `_process_batch_group` for the shared loop body."""
    import torch

    def select(m: int, rows, true_rows, cache: dict[object, object]):
        s = specs[m]
        if _is_unfiltered(s.filter):
            return rows, None, None
        if filter_is_gpu_eligible[s.filter] or filter_is_per_query[s.filter]:
            cell_mask = cache.get(s.filter)
            if cell_mask is None:
                if filter_is_gpu_eligible[s.filter]:
                    cell_mask = _gpu_evaluate(s.filter, leaf_gpu, rows, query_gpu, device)
                else:
                    # `keeps[s.filter]` is bit-packed along the query axis
                    # (see `run_compute`'s reader thread / `_pack_query_axis`)
                    # — row-slice the packed bytes first (cheap: rows aren't
                    # the packed axis), THEN unpack, so only this batch
                    # slice's bytes ever get expanded back to
                    # one-bool-per-query, not the whole file's.
                    packed_np = keeps[s.filter][:, true_rows]
                    cell_np = _unpack_query_axis(packed_np, spec_Q[m].shape[0])
                    cell_mask = torch.from_numpy(cell_np).to(device, non_blocking=True)
                if filter_share_count[s.filter] > 1:
                    cache[s.filter] = cell_mask
            if not cell_mask.any():
                return None, None, None
            return rows, None, cell_mask
        local_idx = cache.get(s.filter)
        if local_idx is None:
            local_np = np.nonzero(keeps[s.filter][true_rows])[0]
            local_idx = torch.from_numpy(local_np).to(device, non_blocking=True)
            cache[s.filter] = local_idx
        if local_idx.numel() == 0:
            return None, None, None
        return rows[local_idx], local_idx, None

    return _process_batch_group(
        batch, member_idxs, specs, spec_Q, spec_top_scores, spec_top_enc,
        batch_size, gidx, device, orig_rows=orig_rows, select=select,
    )


def _validate_query_filter_cols(qstore: Store, filter_cols: list[str]) -> None:
    """Fail fast with a clear message if a per-query filter condition
    (`match_from_query`/`range_from_query`/`match_text_from_query`) names a
    queries column that doesn't exist — rather than letting a typo surface
    deep inside `load_queries`/`load_queries_sparse` as a generic pyarrow
    `ArrowInvalid` about a missing `FieldRef`. Peeks at the FIRST queries
    file's schema only (metadata read, no row data) since every queries
    file in a run is assumed to share one schema."""
    if not filter_cols:
        return
    files = qstore.list_parquets()
    if not files:
        return  # let the empty-queries-store case surface downstream as usual
    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(files[0].read_path, filesystem=qstore.fs).schema_arrow.names)
    missing = sorted(c for c in filter_cols if c not in schema_names)
    if missing:
        raise ValueError(
            f"queries file is missing column(s) referenced by a per-query filter "
            f"(match_from_query/range_from_query/match_text_from_query): {missing} "
            f"— available columns: {sorted(schema_names)}"
        )


def run_compute(
    cfg: BruteForceConfig,
    num_jobs: int | None = None,
    job_rank: int | None = None,
    io_workers: int | None = None,
    io_thread_count: int | None = None,
    max_files: int | None = None,
) -> dict[str, str]:
    """Runs every search in `cfg.searches` — independent vector_type/metric/k/
    filter combinations — against the corpus in ONE pass: each file is read
    and decoded once per vector_type actually needed (not once per spec), and
    every distinct filter is evaluated once per file regardless of how many
    searches share it. Per vector_type, every search then shares one GPU
    batch grid via `_process_shared_batch` — the whole file when some search
    is unfiltered, otherwise the union of every active filter's surviving
    rows (`_union_keep`) — see this module's docstring. Only each search's
    own scoring and top-K accumulation stay independent. Returns
    `{spec.name: output_path}`.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is required for `compute`: install nova-bf[compute]")

    job_rank = _resolve_rank(num_jobs, job_rank)
    specs = cfg.searches
    vts_needed = sorted({s.vector_type for s in specs})  # ["dense"] / ["sparse"] / both
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU detected — brute force on CPU will be slow.")

    # 1. queries — loaded once per DISTINCT vector_type needed across all specs
    #    (queries files are small, so no need to dedupe further than that).
    qstore = Store(cfg.queries.path)
    # Union of every spec's per-query filter columns — same "read exactly (and
    # only) what's referenced" guarantee corpus-side filter_cols already makes,
    # extended to the query side (see Filter.query_fields()).
    query_filter_cols = sorted({c for s in specs if s.filter for c in s.filter.query_fields()})
    _validate_query_filter_cols(qstore, query_filter_cols)
    Q_np_by_vt: dict[str, np.ndarray] = {}
    query_vocab = None  # sparse only: sorted distinct query token ids (see _build_query_vocab)
    query_ids: list[str] | None = None
    payload: dict[str, list] | None = None
    query_filter_vals: dict[str, np.ndarray] | None = None
    for vt in vts_needed:
        if vt == "sparse":
            Q_np, query_vocab, q_ids, q_payload, q_filter_vals = load_queries_sparse(
                qstore, cfg.queries, query_filter_cols
            )
            if len(query_vocab) == 0 and len(q_ids) > 0:
                logger.warning(
                    "sparse query vocabulary is empty (every query has zero nonzero entries) — "
                    "every corpus row will score 0; check queries.sparse_column is correct."
                )
        else:
            Q_np, q_ids, q_payload, q_filter_vals = load_queries(qstore, cfg.queries, query_filter_cols)
        Q_np_by_vt[vt] = Q_np
        if query_ids is None:
            query_ids, payload, query_filter_vals = q_ids, q_payload, q_filter_vals
        elif q_ids != query_ids:
            # Both loaders read the same query store via the same deterministic
            # store.list_parquets() order, so q_ids should be IDENTICAL, not just
            # equal length — checking exact identity (not just len()) catches a
            # row-order mismatch too (e.g. a future loader change that filters or
            # reorders rows differently per column), which a length-only check
            # would silently let through and misattribute query N's dense vector
            # to query M's sparse vector (or id/payload) in the output.
            raise RuntimeError(
                f"queries.{'sparse_column' if vt == 'sparse' else 'dense_column'} produced "
                f"query ids that don't match a different vector_type's load for the same "
                f"query set (first mismatch at row "
                f"{next((i for i, (a, b) in enumerate(zip(q_ids, query_ids)) if a != b), min(len(q_ids), len(query_ids)))}"
                f"; {len(q_ids)} vs {len(query_ids)} total rows) — the two columns must "
                "agree on row count and order"
            )
    n_q = len(query_ids)
    for s in specs:
        dim = Q_np_by_vt[s.vector_type].shape[1]
        logger.info(
            "search=%r queries=%d %s=%d metric=%s k=%d device=%s%s",
            s.name, n_q, "vocab" if s.vector_type == "sparse" else "dim", dim,
            s.metric, s.k, device,
            f" rank={job_rank}/{num_jobs}" if num_jobs else "",
        )
        if s.filter is not None:
            logger.info("search=%r filter: %s", s.name, s.filter.model_dump(exclude_defaults=True))

    # 2. corpus files (global, deterministic order); this worker takes a stride
    #    slice so its global indices stay stable for id decoding. Shared across
    #    every spec — they must all see the identical file set/order/truncation.
    cstore = Store(cfg.corpus.path)
    all_files = cstore.list_parquets()
    # Restrict the recursive glob to the intended shards (see corpus.include/exclude).
    # Applied BEFORE striding so the global file index — and thus make_point_id — is
    # over the filtered set, consistent between compute workers and merge.
    all_files = filter_corpus_files(all_files, cfg.corpus.include, cfg.corpus.exclude)
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

    # 3. per-spec GPU state: each spec gets its own running (top_scores, top_enc)
    #    pair, but its (possibly cosine-normalized) query matrix is shared across
    #    every spec with the same vector_type — normalization is deterministic
    #    given vector_type alone, so re-normalizing per spec would just hold N
    #    identical n_q x dim/vocab copies resident on the GPU for no reason.
    Q_gpu_by_vt = {
        vt: torch.tensor(Q_np_by_vt[vt], dtype=torch.float32, device=device) for vt in vts_needed
    }
    normalized_Q_by_vt: dict[str, object] = {}
    spec_Q, spec_top_scores, spec_top_enc = [], [], []
    for s in specs:
        if s.metric == "cosine":
            if s.vector_type not in normalized_Q_by_vt:
                normalized_Q_by_vt[s.vector_type] = torch.nn.functional.normalize(
                    Q_gpu_by_vt[s.vector_type], dim=1
                )
            Qv = normalized_Q_by_vt[s.vector_type]
        else:
            Qv = Q_gpu_by_vt[s.vector_type]
        spec_Q.append(Qv)
        spec_top_scores.append(torch.full((n_q, s.k), float("-inf"), device=device))
        spec_top_enc.append(torch.zeros((n_q, s.k), dtype=torch.int64, device=device))

    # Per vector_type: does any spec have no filter (or an explicit-but-empty
    # one — see `_is_unfiltered`) OR a per-query filter (see `_is_per_query` —
    # it has no single row-subset to offer, since different queries need
    # different rows)? If so every spec of that vector_type shares the whole
    # file, uncompacted (`has_baseline`); otherwise every spec has a uniform
    # active filter, so the shared grid is instead the UNION of those
    # filters' surviving rows (`_union_keep`, computed fresh per file below).
    # See this module's docstring for the full rationale.
    spec_filter = [s.filter for s in specs]
    distinct_filters: list[Filter | None] = list(dict.fromkeys(spec_filter))
    # Computed once per distinct filter (not per file/slice) — the hot
    # per-member `select` closure below does a cheap dict lookup instead of
    # recomputing `Filter.is_per_query()` (a must/should/must_not walk).
    filter_is_per_query: dict[Filter | None, bool] = {f: _is_per_query(f) for f in distinct_filters}
    # Which of those per-query filters can skip filters.py's CPU/numpy path
    # entirely and evaluate GPU-natively instead (Front A — see
    # _gpu_eligible); each such filter's match_from_query/range_from_query
    # leaves get their small per-query GPU state (vocab codes, membership
    # matrices, bound tensors) built ONCE here, not per file — the
    # per-query analog of Q_gpu_by_vt's one-time transfer above. Keyed by
    # the FilterCondition itself (not by Filter), so two different filters
    # sharing an identical per-query leaf build/transfer it only once —
    # same sharing leaf_arrays/leaf_gpu already give the corpus side below.
    filter_is_gpu_eligible: dict[Filter | None, bool] = {f: _gpu_eligible(f) for f in distinct_filters}
    gpu_query_gpu: dict[FilterCondition, object] = {}
    gpu_vocabs: dict[FilterCondition, np.ndarray] = {}
    for f in distinct_filters:
        if not filter_is_gpu_eligible[f]:
            continue
        for cond in f.all_conditions():
            if cond.match_from_query is None and cond.range_from_query is None:
                continue  # a static leaf inside an eligible filter needs no per-query GPU state
            if cond in gpu_query_gpu:
                continue
            qgpu, vocab = _build_gpu_leaf_state(cond, query_filter_vals, device)
            gpu_query_gpu[cond] = qgpu
            if vocab is not None:
                gpu_vocabs[cond] = vocab
    # How many specs share each distinct filter — used by _process_shared_batch
    # to skip caching a per-query cell_mask when nobody else will reuse it
    # (Front A's memory-shrinking complement).
    filter_share_count: dict[Filter | None, int] = Counter(spec_filter)

    vt_spec_idxs: dict[str, list[int]] = {vt: [] for vt in vts_needed}
    for i, s in enumerate(specs):
        vt_spec_idxs[s.vector_type].append(i)

    vt_configured_batch = {"dense": cfg.params.dense_batch_size, "sparse": cfg.params.sparse_batch_size}
    has_baseline: dict[str, bool] = {}
    vt_batch_size: dict[str, int | None] = {}
    vt_union_filters: dict[str, list[Filter]] = {}
    for vt, idxs in vt_spec_idxs.items():
        # Front B: a per-query filter no longer forces the whole file — see
        # keeps[f]'s row-level union for a per-query filter (built in the
        # reader thread below) and _row_union_from_gpu_leaves's docstring.
        has_baseline[vt] = any(_is_unfiltered(specs[i].filter) for i in idxs)
        # Every search of this vector_type shares one batch grid regardless of
        # which branch below applies, so k_floor spans ALL of them, not just a
        # same-filter subset — see _resolve_vt_batch_size's docstring.
        k_floor = max(specs[i].k for i in idxs)
        vt_batch_size[vt] = _resolve_vt_batch_size(vt_configured_batch[vt], k_floor, vt)
        if has_baseline[vt]:
            logger.info(
                "vector_type=%r: %d search(es) share one full-file batch pass (batch_size=%s)",
                vt, len(idxs), vt_batch_size[vt],
            )
        else:
            vt_union_filters[vt] = list(dict.fromkeys(spec_filter[i] for i in idxs))
            logger.info(
                "vector_type=%r: no unfiltered search — %d search(es) share one batch pass "
                "over the union of %d distinct filter(s) (batch_size=%s)",
                vt, len(idxs), len(vt_union_filters[vt]), vt_batch_size[vt],
            )
    # Only a filter that actually appears in SOME vt's union ever reaches
    # _union_keep (a filter whose vt has has_baseline=True never does) — so
    # only these need the per-file row-level union computed at all (Front B).
    filters_needing_row_union: set[Filter] = {f for fs in vt_union_filters.values() for f in fs}

    # Prefetch corpus files with a POOL of reader threads so many S3 GETs are in
    # flight at once — otherwise the GPU sits idle behind one file's latency at a
    # time. pyarrow releases the GIL during IO, so threads parallelize. Reader
    # threads may FINISH in any order, but the consumer below folds files into
    # the running top-K in a FIXED order (ascending `gidx`) regardless of
    # arrival order — see the comment above the consumer loop for why: the
    # merge is commutative in the SCORES it produces, but not in which of
    # several EXACTLY tied candidates a run picks.
    dense_col, sparse_col = cfg.corpus.dense_column, cfg.corpus.sparse_column
    id_col = cfg.corpus.id_column  # None → derive make_point_id(file_key, row) at decode
    # Union of every spec's filter fields — read_cols below stays exactly (and only)
    # the columns some spec actually references, same guarantee the single-search
    # path always made.
    filter_cols = sorted({c for s in specs if s.filter for c in s.filter.fields()})
    read_cols = list(dict.fromkeys(
        ([dense_col] if "dense" in vts_needed else [])
        + ([sparse_col] if "sparse" in vts_needed else [])
        + ([id_col] if id_col else [])
        + filter_cols
    ))
    need_sparse_norms = any(s.vector_type == "sparse" and s.metric == "cosine" for s in specs)
    # Corpus id strings carried through to decode, kept in RAM per file (only when
    # id_column is set). gidx → pyarrow string array aligned with that file's rows.
    # Shared by every spec; set below whenever ANY spec could still resolve a hit
    # from this file (unfiltered, or a filter keeping ≥1 row) — never gated on the
    # CURRENT spec alone, since a restrictive spec's filter dropping a whole file
    # must not block a DIFFERENT spec's id resolution for that same file.
    corpus_ids: dict[int, object] = {}
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
            # Wrapped in try/except so a bad read/decode/filter (e.g. a filter
            # field missing from this file's schema, or a type pyarrow can't
            # compare) fails the run loudly instead of silently killing this
            # thread — an uncaught exception here would otherwise just print a
            # traceback to stderr and die, leaving the consumer's fixed-count
            # `fq.get()` loop blocked forever waiting for an item that will
            # never arrive. Putting the exception itself on the queue lets the
            # consumer re-raise it in the main thread with a clear message.
            try:
                t0 = time.perf_counter()
                table = cstore.read_columns(f.read_path, read_cols)
                # Decode each vector_type at most ONCE per file, regardless of how many
                # specs need it — wrapped in the batch abstraction (`DenseCorpusBatch`/
                # `SparseCorpusBatch`) in the consumer loop below, where every spec of
                # that vector_type shares it (see `run_compute`'s `has_baseline`).
                arrs: dict[str, object] = {}
                if "dense" in vts_needed:
                    arrs["dense"] = dense_to_2d(table[dense_col])
                if "sparse" in vts_needed:
                    sp_offsets, sp_idx, sp_val = sparse_to_coo_parts(table[sparse_col])
                    sp_norms = _sparse_file_norms(sp_offsets, sp_idx, sp_val) if need_sparse_norms else None
                    arrs["sparse"] = (sp_offsets, sp_idx, sp_val, sp_norms)
                # carry the id column (combined to one contiguous array) to decode;
                # None when id_column isn't configured. Same row order as `arrs`.
                ids = table[id_col].combine_chunks() if id_col else None
                t1 = time.perf_counter()
                # One mask per DISTINCT filter (`None` for the unfiltered entry),
                # evaluated against the same table — timed separately from the read
                # above (CPU-vectorized work, not IO wait). Keyed by the `Filter`
                # object itself (frozen, hashable — see `nova_bf.config.Filter`)
                # so two specs sharing an identical filter never evaluate it twice.
                # `query_filter_vals` feeds any per-query condition; `evaluate()`
                # returns `(rows,)` for a purely-uniform filter (unchanged cost),
                # or `(n_queries, rows)` the moment any condition is per-query.
                #
                # A GPU-eligible per-query filter (Front A — see _gpu_eligible)
                # skips evaluate() entirely — its FINE, per-query mask is built
                # lazily, per batch slice, straight from per-CONDITION corpus
                # arrays instead (`leaf_arrays`, keyed by the FilterCondition
                # object so two eligible filters sharing an identical leaf
                # compute it once) — shared per (field, leaf-kind) exactly like
                # `keeps` is shared per whole Filter. It still gets a `keeps`
                # entry when SOME vt actually needs it (`filters_needing_row_
                # union`): a cheap (rows,) safe-superset "does any query want
                # this row at all" reduction (Front B — see
                # _row_union_from_gpu_leaves), used only for union-compaction
                # (`_union_keep`) and the corpus_ids retention check below —
                # never for the per-query cell_mask itself.
                #
                # The CPU-fallback branch below (a filter with a `match_text`/
                # `match_text_from_query` leaf, so ineligible for Front A) is
                # the ONLY place `evaluate()` can still return a genuine
                # `(n_queries, rows)` array — held in `keeps[f]` for this
                # whole file's batch loop. Bit-packed along the query axis
                # (`np.packbits(mask, axis=0)`, 8 queries/byte) to cut that
                # long-lived footprint 8x; every consumer below either works
                # unchanged on the packed bytes (`.any(axis=0)` in
                # `_union_keep` — a byte is 0 iff every query bit in it is,
                # so byte-truthiness IS query-truthiness) or unpacks lazily,
                # only for the batch-row slice actually needed
                # (`_process_shared_batch`'s `select`).
                n_rows = len(table)
                keeps: dict[Filter | None, np.ndarray | None] = {}
                leaf_arrays: dict[FilterCondition, np.ndarray] = {}
                for f in distinct_filters:
                    if f is None:
                        keeps[f] = None
                    elif filter_is_gpu_eligible[f]:
                        for cond in f.all_conditions():
                            if cond not in leaf_arrays:
                                leaf_arrays[cond] = _corpus_leaf_array(cond, table, gpu_vocabs.get(cond))
                        if f in filters_needing_row_union:
                            union = _row_union_from_gpu_leaves(f, leaf_arrays, query_filter_vals, n_rows)
                            keeps[f] = union if union is not None else np.ones(n_rows, dtype=bool)
                    else:
                        mask = evaluate(f, table, query_filter_vals)
                        keeps[f] = _pack_query_axis(mask) if mask.ndim == 2 else mask
                t2 = time.perf_counter()
                fq.put((gidx, arrs, ids, keeps, leaf_arrays, t1 - t0, t2 - t1))
            except Exception as exc:
                fq.put(exc)
                return

    for _ in range(io_workers):
        Thread(target=reader, daemon=True).start()

    # Timing split (debug): `io_wait` is real time the consumer blocked on an
    # empty queue == the GPU starved waiting for reads — the number that matters
    # here. `gpu_secs` is just CPU-side enqueue time (CUDA is async and overlaps
    # the next read), so it being tiny is itself evidence we're not compute-bound.
    # `read_secs` is summed per-file read latency across the reader threads.
    # `filter_secs` is summed per-file mask-evaluation time (0 when unfiltered),
    # also across the reader threads — kept apart from `read_secs` so a slow
    # filter doesn't masquerade as slow IO. All four are run-wide (summed across
    # every spec's filter/scoring work on a file), not per spec.
    any_filter = any(s.filter is not None for s in specs)
    io_wait = gpu_secs = read_secs = filter_secs = 0.0
    rows_seen = 0
    bytes_seen = 0  # decoded float32 bytes consumed (~= wire bytes for snappy-float32)
    wall0 = time.perf_counter()

    def _fetch_or_raise() -> tuple:
        it = fq.get()
        if isinstance(it, Exception):
            raise RuntimeError(
                "a reader thread failed while reading/decoding/filtering a corpus file"
            ) from it
        return it

    # `mine` already lists this worker's files in a fixed, deterministic
    # order (ascending `gidx`); `_next_in_order` reorders reader threads'
    # arbitrary-arrival-order output back into that order (bounded by
    # `io_workers`-many buffered items, since `fq`'s own bound already caps
    # how far readers can race ahead) — see its docstring for why this
    # matters for reproducibility, not just correctness.
    pending: dict[int, tuple] = {}

    with tqdm(total=len(mine), unit="file", dynamic_ncols=True, desc="bf") as bar:
        for want_gidx, _f in mine:
            w0 = time.perf_counter()
            gidx, arrs, ids, keeps, leaf_arrays, rsec, fsec = _next_in_order(want_gidx, pending, _fetch_or_raise)
            io_wait += time.perf_counter() - w0
            read_secs += rsec
            filter_secs += fsec
            bar.update(1)

            # Wrap this file's decoded arrays in the vector_type-agnostic batch
            # abstraction ONCE — every spec of a vector_type shares the same
            # wrapper below, never rebuilding or mutating the underlying
            # decoded arrays.
            batches: dict[str, object] = {}
            if "dense" in vts_needed:
                batches["dense"] = DenseCorpusBatch(arrs["dense"])
            if "sparse" in vts_needed:
                sp_offsets, sp_idx, sp_val, sp_norms = arrs["sparse"]
                batches["sparse"] = SparseCorpusBatch(sp_offsets, sp_idx, sp_val, sp_norms, query_vocab, need_sparse_norms)

            # Front A: transfer this file's GPU-eligible per-query leaf arrays
            # to the GPU ONCE here (not once per batch slice) — mirrors
            # Q_gpu_by_vt's one-time transfer, vector_type-agnostic since a
            # filter condition reads a payload column, never a vector column.
            leaf_gpu: dict[FilterCondition, object] = {
                cond: torch.from_numpy(arr).to(device, non_blocking=True)
                for cond, arr in leaf_arrays.items()
            }

            # Whole-file bookkeeping, computed BEFORE any spec's per-vector-type
            # compaction below. rows_seen/bytes_seen are counted once per file per
            # distinct vector_type present (pre-filter), not per spec — with
            # multiple specs there's no longer a single "the" filtered row count
            # to report.
            #
            # corpus_ids is kept only for files where SOME spec could still
            # resolve a hit — i.e. it's unfiltered, or its filter keeps at least
            # one row in this file. Each entry of `keeps` is a mask over this
            # file's rows independent of vector_type (filters read payload
            # columns, not the vector columns), so checking it here — before
            # any vector-type-specific compaction below — is exact, not an
            # approximation: a restrictive spec's filter dropping the whole
            # file must never block a DIFFERENT spec's id resolution for that
            # same file, but a file every spec's filter drops needs no ids
            # kept at all. A GPU-eligible filter's `keeps` entry (when
            # present — see `filters_needing_row_union` above; it's skipped
            # entirely for a filter no vt's union ever needs) is a safe
            # OVER-approximation (Front B — see _row_union_from_gpu_leaves),
            # never a false negative, so `.any()` here is still exact for
            # "definitely nobody wants this file" and only ever conservative
            # (never wrongly dropping) in the "maybe somebody does" direction.
            # A gpu-eligible filter MISSING from `keeps` only happens when
            # its own vt has has_baseline=True, i.e. some OTHER spec of that
            # vt is unfiltered — `keeps[None]` (`is None`) already covers
            # retention for that file, so the missing entry costs nothing.
            if id_col and any(mask is None or mask.any() for mask in keeps.values()):
                corpus_ids[gidx] = ids
            for vt in vts_needed:
                rows_seen += batches[vt].n_rows
                bytes_seen += batches[vt].nbytes

            for vt in vts_needed:
                b = batches[vt]
                if has_baseline[vt]:
                    batch, orig_rows = b, None
                else:
                    # Compact to the union OUTSIDE _process_shared_batch's own
                    # timer, same as the old per-filter-group compaction did —
                    # this CPU-side cost (row filtering, array copies — scales
                    # with corpus size, not GPU work) never counts toward the
                    # gpu_secs signal (see the io_wait/gpu_secs split docs at
                    # the top of this module).
                    batch, orig_rows = b.compact(_union_keep(vt_union_filters[vt], keeps))
                gpu_secs += _process_shared_batch(
                    batch, vt_spec_idxs[vt], specs, spec_Q, spec_top_scores, spec_top_enc,
                    keeps, filter_is_per_query, filter_is_gpu_eligible, leaf_gpu, gpu_query_gpu,
                    filter_share_count, vt_batch_size[vt], gidx, device, orig_rows=orig_rows,
                )

            if bar.n % 200 == 0:
                postfix = f"io_wait={io_wait:.0f}s gpu={gpu_secs:.0f}s"
                if any_filter:
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
        f"(summed {filter_secs:.0f}s over {io_workers} threads)" if any_filter else "",
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

    # 4. decode each spec's final top-K into hit ids and write its own output.
    #    Either recompute make_point_id from (file_key, row) — only K*n_q ids,
    #    nothing corpus-wide — or, when an id column is configured, read it back
    #    from the in-RAM per-file arrays. Shared by every spec: depends only on
    #    id_col/corpus_ids/all_files, none of which is per-spec.
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

    out = Store(cfg.output.path)
    results: dict[str, str] = {}
    for i, s in enumerate(specs):
        enc = spec_top_enc[i].cpu().numpy()
        sc = spec_top_scores[i].cpu().numpy()
        valid = sc > float("-inf")
        hit_ids, hit_scores = [], []
        for q in range(n_q):
            qe, qs = enc[q][valid[q]], sc[q][valid[q]]
            hit_ids.append([resolve_id(int(e)) for e in qe])
            hit_scores.append(qs.tolist())

        table = build_result_table(query_ids, payload, hit_ids, hit_scores)
        if num_jobs is not None:
            width = max(3, len(str(num_jobs - 1)))
            name = f"{partial_dir(cfg, s)}/rank{job_rank:0{width}d}.parquet"
            # This worker's slice of the corpus naturally has fewer than k
            # candidates per query most of the time (stride partitioning spreads
            # the corpus thin) -- that's expected here, not a signal of anything.
            # Only `merge` (or this function's own single-node path below) sees
            # the true final count, so that's the only place worth warning.
        else:
            name = result_name(cfg, s)
            warn_if_short(sum(1 for h in hit_ids if len(h) < s.k), len(hit_ids), s.k, s.name, logger)
        path = out.write(name, table)
        logger.info("search=%r wrote %s (%d queries)", s.name, path, n_q)
        results[s.name] = path
    return results
