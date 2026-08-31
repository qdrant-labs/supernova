"""The `compute` phase: score corpus slices and produce an exact per-query top-K.

Each worker loads its queries onto the GPU, streams its assigned corpus files, and
incrementally merges each file's results into a running top-K. I/O is prefetched
in parallel with GPU computation, and large files can be processed in row batches
to bound GPU memory. The final top-K is written as one Parquet file per worker.

The running state stores `(score, encoded_row)` rather than materializing corpus
IDs on the GPU. `encoded_row` identifies the source file and row; final IDs are
resolved only for the winning K entries. If `corpus.id_column` is provided, those
IDs are read from the corpus instead.

One invocation may evaluate multiple independent `SearchSpec`s, including dense,
sparse, multi-vector, filtered, and unfiltered searches. Searches remain
independent ranked lists, but redundant work is shared: each required vector type
is decoded and transferred once per batch, and searches using the same vector
type reuse the same GPU-resident data and score computations.

For dense search, multiple metrics share one `Q @ C^T` product and derive dot,
cosine, and Euclidean scores from it. Sparse dot and cosine similarly share their
underlying sparse product.

Filters are applied before or during scoring. Uniform filters compact the corpus
to surviving rows before GPU transfer. When several filtered searches share a
vector type, their surviving rows are unioned into one shared batch and each
search masks that batch back to its own subset. If any search is unfiltered, the
whole batch is scored once and filtered searches reuse those scores.

Per-query filters cannot be represented by one shared row subset, so they retain
the necessary candidate columns and mask individual `(query, row)` scores to
`-inf` before top-K selection. Numeric and categorical per-query filters are
evaluated GPU-natively when possible; text predicates fall back to the CPU path.

The CPU-fallback mask is the one `(n_queries, file_rows)` array a run
materializes, held for a whole corpus file's batch loop. Its query axis is that
FILTER's own row union (`run_compute`'s `filter_rows`), not the queries file's
height, so it does not grow when unrelated query sets are unioned into one file
behind a `SearchSpec.rows` selector.
"""

from __future__ import annotations

import logging
import os
import re
import time

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from queue import Empty, Queue
from threading import Semaphore, Thread

import numpy as np

from tqdm import tqdm

from nova_bf import manifest as run_manifest
from nova_bf.config import BruteForceConfig, Filter, FilterCondition, SearchSpec
from nova_bf.filters import _condition_mask, _match_any_membership, _static_first, evaluate
from nova_bf.dates import convert_table_date_columns, normalize_date_fields
from nova_bf.ids import make_point_id
from nova_bf.io import ParquetFile, Store, dense_to_2d, multivector_to_ragged, sparse_to_coo_parts
from nova_bf.results import (
    build_result_table,
    partial_dir,
    provenance,
    result_name,
    vector_dtype,
    warn_if_short,
)
from nova_bf.tiebreak import (
    MAX_ROWS_PER_WORKER,
    build_ordinals,
    id_order_scalar,
    id_order_array,
    pack,
    pack_topk,
    sentinel_key,
    unpack_score,
)

logger = logging.getLogger(__name__)

PREFETCH_QUEUE_SIZE = 4
# Limit how many top-K entries are decoded/sorted at once to bound peak GPU
# memory during final result decoding (~384 MiB of temporary storage).
DECODE_CHUNK_SLOTS = 1 << 24
# Maximum rows per corpus file; used to keep encoded row IDs collision-free.
MAX_ROWS_PER_FILE = 100_000_000

# Fraction of FREE CUDA memory a multivector batch's whole token matrix may
# occupy for the batch to be transferred to the device ONCE and sliced as
# zero-copy views (see _process_batch_group), instead of one H2D per slice.
# Conservative: the per-slice score matrix P (bounded by the token budget),
# metric copies, and the running top-k state share the same pool.
_MV_RESIDENT_FREE_FRACTION = 0.5


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
    a MatchAny column can legitimately have `None` (that query matches
    nothing, same as a null scalar) for some query and a real list for
    another, and checking only `values[0]` would
    misclassify the whole column whenever THAT one row happens to be null —
    `np.array([None, [...], [...]])` raises `ValueError` (inhomogeneous
    shape) rather than producing the object array `filters.py` expects."""
    if any(isinstance(v, (list, tuple)) for v in values):
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        return arr
    return np.array(values)


def load_queries(
    store: Store, qcfg, filter_cols: list[str] = (), rows: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list], dict[str, np.ndarray]]:
    """`rows` (sorted file-row indices, see `SearchSpec.rows`) restricts the
    returned MATRIX to those queries. ids/payload/filter_vals stay FULL length
    either way, and per-query filter masks are built over the full query axis"""
    cols = [qcfg.dense_column]
    if qcfg.id_column:
        cols.append(qcfg.id_column)
    cols += [c for c in qcfg.payload_fields if c not in cols]
    cols += [c for c in filter_cols if c not in cols]

    embs: list[np.ndarray] = []
    ids: list[str] = []
    payload: dict[str, list] = {c: [] for c in qcfg.payload_fields}
    filter_vals: dict[str, list] = {c: [] for c in filter_cols}
    q_date_fmts = normalize_date_fields(qcfg.date_fields)
    for f in store.list_parquets():
        table = store.read_columns(f.read_path, cols)
        d = table.to_pydict()  # ORIGINAL values — payload/id keep their source form
        # Declared datetime query columns -> int64 epoch µs, but ONLY for the
        # per-query filter arrays, so a range_from_query bound compares as a
        # number against the (also-µs) corpus date column. A date field carried
        # in payload_fields keeps its original (string) form in `d` above.
        conv = convert_table_date_columns(table, q_date_fmts)
        embs.append(dense_to_2d(conv[qcfg.dense_column]))
        n = len(conv)
        if qcfg.id_column:
            ids += [str(x) for x in d[qcfg.id_column]]
        else:
            ids += [make_point_id(f.key, r) for r in range(n)]
        for c in qcfg.payload_fields:
            payload[c] += d[c]
        for c in filter_cols:
            filter_vals[c] += conv[c].to_pylist()
    Q = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), np.float32)
    if rows is not None:
        Q = Q[rows]
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
    store: Store, qcfg, filter_cols: list[str] = (), rows: np.ndarray | None = None,
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
    q_date_fmts = normalize_date_fields(qcfg.date_fields)
    for f in store.list_parquets():
        table = store.read_columns(f.read_path, cols)
        d = table.to_pydict()  # ORIGINAL values — payload/id keep their source form
        # Date columns -> epoch µs for filter arrays only (payload keeps strings).
        conv = convert_table_date_columns(table, q_date_fmts)
        row_offsets, idx, val = sparse_to_coo_parts(conv[qcfg.sparse_column])
        counts_parts.append(np.diff(row_offsets))
        indices_parts.append(idx)
        values_parts.append(val)
        n = len(row_offsets) - 1
        if qcfg.id_column:
            ids += [str(x) for x in d[qcfg.id_column]]
        else:
            ids += [make_point_id(f.key, r) for r in range(n)]
        for c in qcfg.payload_fields:
            payload[c] += d[c]
        for c in filter_cols:
            filter_vals[c] += conv[c].to_pylist()

    indices = np.concatenate(indices_parts) if indices_parts else np.zeros(0, np.int64)
    values = np.concatenate(values_parts) if values_parts else np.zeros(0, np.float32)
    counts = np.concatenate(counts_parts) if counts_parts else np.zeros(0, np.int64)
    n_q = len(counts)
    row_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    if rows is not None:
        # Compact to the subset BEFORE the vocab build and the densify.
        # Narrowing the vocab (only the subset's own token ids) is exact:
        # a token no surviving query uses can never contribute to any score.
        keep = np.zeros(n_q, dtype=bool)
        keep[rows] = True
        row_offsets, indices, values, _, _ = _compact_sparse_rows(
            row_offsets, indices, values, None, keep
        )
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


def load_queries_multivector(
    store: Store, qcfg, filter_cols: list[str] = (),
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list], dict[str, np.ndarray]]:
    """Multivector analog of `load_queries`/`load_queries_sparse`: reads the
    `list<list<float32>>` column from every query file and stacks all queries'
    token vectors into one flat `(total_query_tokens, D)` matrix plus a
    length-`n_q+1` token-offset array (`doc_offsets` semantics, query side).
    Queries are few, so holding every query token on the GPU at once is cheap;
    the query axis is tiled at SCORE time (`multivector_query_block`), not
    here. Returns `(flat_tokens, q_offsets, ids, payload, filter_vals)`."""
    cols = [qcfg.multivector_column]
    if qcfg.id_column:
        cols.append(qcfg.id_column)
    cols += [c for c in qcfg.payload_fields if c not in cols]
    cols += [c for c in filter_cols if c not in cols]

    tokens_parts: list[np.ndarray] = []
    counts_parts: list[np.ndarray] = []  # tokens per query, per file
    dim = 0
    ids: list[str] = []
    payload: dict[str, list] = {c: [] for c in qcfg.payload_fields}
    filter_vals: dict[str, list] = {c: [] for c in filter_cols}
    q_date_fmts = normalize_date_fields(qcfg.date_fields)
    for f in store.list_parquets():
        table = store.read_columns(f.read_path, cols)
        d = table.to_pydict()  # ORIGINAL values — payload/id keep their source form
        conv = convert_table_date_columns(table, q_date_fmts)
        # Decode the multivector column from the ORIGINAL table (date conversion
        # only ever touches declared date columns, never a vector column — but
        # reading it straight from `table` removes any doubt); `conv` feeds only
        # the filter fields, which may be date-normalized.
        doc_offsets, flat = multivector_to_ragged(table[qcfg.multivector_column])
        if flat.shape[1] > 0:
            if dim and flat.shape[1] != dim:
                raise ValueError(
                    f"queries.{qcfg.multivector_column} token dim mismatch: file "
                    f"{f.key!r} has D={flat.shape[1]} but an earlier file had D={dim} "
                    "— every query file must share one token dimension"
                )
            dim = flat.shape[1]
        tokens_parts.append(flat)
        counts_parts.append(np.diff(doc_offsets))
        n = len(doc_offsets) - 1
        if qcfg.id_column:
            col_ids = d[qcfg.id_column]
            if any(x is None for x in col_ids):
                raise ValueError(
                    f"queries.id_column {qcfg.id_column!r} has null value(s) in {f.key!r} "
                    "— query ids must be present (a null would become the string 'None')"
                )
            ids += [str(x) for x in col_ids]
        else:
            ids += [make_point_id(f.key, r) for r in range(n)]
        for c in qcfg.payload_fields:
            payload[c] += d[c]
        for c in filter_cols:
            filter_vals[c] += conv[c].to_pylist()

    # A file whose queries are all zero-token decodes to a (0, 0) `flat`; drop
    # those empty shards before concat so the width is the real token dim, and
    # fall back to a (0, dim) empty matrix if EVERY query is zero-token. Zero-
    # token queries are NOT lost — their zero counts stay in `counts_parts`, so
    # they keep their (zero-width) slot in `q_offsets` (and score -inf).
    nonempty = [t for t in tokens_parts if t.shape[0] > 0]
    flat_tokens = (
        np.concatenate(nonempty, axis=0) if nonempty else np.zeros((0, dim), np.float32)
    )
    counts = np.concatenate(counts_parts) if counts_parts else np.zeros(0, np.int64)
    q_offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    # Alignment invariants (cheap; a violation means a decode/loader bug upstream,
    # which would otherwise misattribute one query's vectors to another's id).
    n_q = len(q_offsets) - 1
    assert len(ids) == n_q, f"query id count {len(ids)} != query count {n_q}"
    assert int(q_offsets[-1]) == flat_tokens.shape[0], (
        f"query offsets end at {int(q_offsets[-1])} but token matrix has "
        f"{flat_tokens.shape[0]} rows"
    )
    return flat_tokens, q_offsets, ids, payload, {c: _to_query_array(v) for c, v in filter_vals.items()}


def _compact_multivector_rows(
    doc_offsets: np.ndarray, flat_tokens: np.ndarray, keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-level filter compaction for ragged multivector parts — the
    token-granular analog of `_compact_sparse_rows`. Whole per-doc token spans
    are gathered by offset; dropped docs' tokens are dropped too, but surviving
    docs' TRUE file-row numbers are preserved via `orig_rows` (the same
    invariant `make_point_id`/`id_column` resolution rely on). Returns
    `(new_doc_offsets, new_flat_tokens, orig_rows)`."""
    orig_rows = np.nonzero(keep)[0]
    tok_keep = np.repeat(keep, np.diff(doc_offsets))
    new_flat = flat_tokens[tok_keep]
    new_doc_offsets = np.concatenate(([0], np.cumsum(np.diff(doc_offsets)[orig_rows]))).astype(np.int64)
    return new_doc_offsets, new_flat, orig_rows


def _remap_sparse_file(
    row_offsets: np.ndarray, indices: np.ndarray, values: np.ndarray, vocab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remap one whole file's raw CSR parts into the query vocabulary:
    out-of-vocab entries dropped (see `_build_query_vocab`), duplicate
    (row, col) pairs summed and per-row column order sorted
    (`_coalesce_by_row_col` — both required for valid torch CSR, see
    `_sparse_batch_to_csr`).

    Runs ONCE per file, in the reader threads (parallel, overlapped with GPU
    work) — it used to run per batch SLICE on the single consumer thread,
    serializing an O(nnz log nnz) lexsort with every GPU call (~240 slices/
    file at fineweb scale). Must run AFTER `_sparse_file_norms`, which needs
    the raw pre-truncation values (see its docstring)."""
    n_rows = len(row_offsets) - 1
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(row_offsets))
    idx = _vocab_lookup(vocab, indices)
    keep_nnz = idx >= 0
    row_ids, idx, val = row_ids[keep_nnz], idx[keep_nnz], values[keep_nnz]
    row_ids, idx, val = _coalesce_by_row_col(row_ids, idx, val)
    counts = np.bincount(row_ids, minlength=n_rows)
    new_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return new_offsets, idx, val


def _sparse_batch_to_csr(
    row_offsets: np.ndarray, indices: np.ndarray, values: np.ndarray,
    r0: int, r1: int, vocab: np.ndarray, device: str,
):
    """One row-slice of an ALREADY-remapped file (see `_remap_sparse_file`:
    indices are query-vocab column ids, per-row sorted and deduped) as a
    torch sparse CSR on `device`. Pure slicing — no lookup, no sort.

    Deliberately takes no `norms`/metric argument and never scales values —
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

    `check_invariants=False` skips torch's validation that each row's column
    indices are sorted and distinct — a real, enforced CSR invariant.
    `_remap_sparse_file`'s coalesce established exactly those two properties
    for the whole file, and row-slicing preserves them, so the skip is safe."""
    import torch

    lo, hi = int(row_offsets[r0]), int(row_offsets[r1])
    crow = (row_offsets[r0 : r1 + 1] - lo).astype(np.int64, copy=False)
    Cb = torch.sparse_csr_tensor(
        torch.from_numpy(crow),
        torch.from_numpy(indices[lo:hi]),
        torch.from_numpy(values[lo:hi]),
        size=(r1 - r0, len(vocab)), check_invariants=False,
    )
    return Cb.to(device, non_blocking=True)


def _sparse_scores(Q, Cb, q_cache=None):
    """Raw `(n_q, n_rows)` sparse dot-product scores for one corpus slice.

    Uses `Q_csr @ dense(Cb).T` when the dense corpus slice fits in memory. This
    is much faster than `Cb @ Q.T`, whose dense query operand is a poorly
    strided transposed view. Falls back to the contiguous-transpose path when
    densifying `Cb` would exceed `_SPARSE_SWAP_MAX_DENSE_BYTES`.

    Returns raw scores; metric-specific transforms are applied by the caller.
    """
    import torch

    n_rows, vocab = Cb.shape
    cache = q_cache or _SparseQueryCache()
    if vocab * n_rows * Q.element_size() <= _SPARSE_SWAP_MAX_DENSE_BYTES:
        # Produces the `(n_q, n_rows)` layout downstream consumers expect.
        return torch.matmul(cache.values(Q), _dense_slice_t(Cb, Cb.values()))
    return torch.matmul(Cb, cache.transpose(Q)).T


def _scores(Q, C, metric: str, q_norms=None):
    import torch.nn.functional as F

    if metric == "cosine":
        # C is normalized per file; Q stays RAW (shared with dot searches —
        # see run_compute's `q_norms_by_vt`), so divide each query's row of
        # the score matrix by its norm here instead.
        return (Q @ F.normalize(C, dim=1).T).div_(q_norms[:, None])
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
    vector_type.

    `share_gram` is set by `_process_batch_group` (from `metric_share_count`)
    once it knows how many DISTINCT metrics will score this batch, and is
    handed to every slice `transfer` produces — carried on the batch rather
    than passed as a `transfer` argument so the `transfer(r0, r1, device)`
    surface stays identical across vector_types. See `DenseBatchSlice`."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.share_gram = False

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
        return DenseBatchSlice(Cb, self.share_gram)


@dataclass
class DenseBatchSlice:
    """One on-device slice of a dense corpus batch.

    When two or more DISTINCT metrics score this slice (`share_gram`, set from
    `metric_share_count` — see `_process_batch_group`), all of them are derived
    from ONE raw Gram matrix `Q @ Cbᵀ` instead of each running its own GEMM:

        dot        the Gram itself, returned WITHOUT a copy
        cosine     Gram / ‖c‖ / ‖q‖              (per-row, then per-query scalar)
        euclidean  −sqrt(‖q‖² + ‖c‖² − 2·Gram)   (clamped at 0)

    The two formulations are mathematically equivalent and differ in float32 only because
    normalization and accumulation happen in a different order — measured ~1 ulp
    apart, with NO consistent accuracy advantage either way.

    Cost: one extra `(n_q, rows)` matrix at peak — the Gram stays alive across
    the member loop while each derived metric is built, where the unshared path
    holds only the current metric's matrix. Bounded by `params.dense_batch_size`,
    and the same trade `SparseBatchSlice` already documents for its own raw
    cache. A batch scored by a SINGLE metric keeps the unshared `_scores` path
    untouched, so the common case pays neither the extra matrix nor any change
    in output.
    """

    Cb: object  # torch.Tensor, (n_rows, dim)
    share_gram: bool = False
    _raw: object = None       # lazy Q @ Cbᵀ — only ever built when share_gram
    _c_norms: object = None   # lazy per-row L2 norms (cosine)
    _c_sq: object = None      # lazy per-row squared L2 norms (euclidean)

    @property
    def n_rows(self) -> int:
        return self.Cb.shape[0]

    def score(self, Q, metric: str, q_norms=None):
        if not self.share_gram:
            return _scores(Q, self.Cb, metric, q_norms)
        if self._raw is None:
            self._raw = Q @ self.Cb.T
        raw = self._raw
        if metric == "dot":
            # The Gram IS the dot score matrix — no copy. Callers never mutate
            # what `score()` hands back in place (`_process_batch_group` uses
            # `masked_fill`/column indexing, both of which allocate), so this
            # stays safe to alias for the other metrics below.
            return raw
        if metric == "cosine":
            if self._c_norms is None:
                # clamp matches F.normalize's eps, so a zero corpus row scores
                # 0 rather than NaN — identical convention to `_scores`.
                self._c_norms = self.Cb.norm(dim=1).clamp_min(1e-12)
            # First div allocates (raw must survive for the other metrics);
            # the second is in place on that fresh copy.
            return raw.div(self._c_norms[None, :]).div_(q_norms[:, None])
        # euclidean: negate the distance so larger = nearer (topk picks
        # nearest), matching `_scores`. Accumulated into ONE new tensor
        # (`raw.mul(-2)` then two broadcast adds in place) rather than the
        # naive `q_sq + c_sq - 2*raw`, which would hold two temporaries of the
        # full (n_q, rows) size at once.
        if self._c_sq is None:
            self._c_sq = self.Cb.pow(2).sum(1)
        # `run_compute` supplies `q_norms` for euclidean specs precisely so this
        # is a reuse rather than a recompute; the fallback keeps `score()`
        # callable standalone (tests, any future direct caller).
        q_sq = q_norms.pow(2) if q_norms is not None else Q.pow(2).sum(1)
        d2 = raw.mul(-2.0)
        d2 += self._c_sq[None, :]
        d2 += q_sq[:, None]
        return d2.clamp_min_(0).sqrt_().neg_()


class SparseCorpusBatch:
    """Decoded sparse corpus CSR parts for one file — ALREADY remapped into
    the query vocabulary, per-row sorted and deduped (`_remap_sparse_file`,
    done once per file in the reader threads; `.transfer()` is pure slicing)
    — plus each row's true (untruncated, PRE-remap) L2 norm (`norms`, `None`
    unless some spec needs cosine — see `_sparse_file_norms`). `nbytes`
    consequently reflects the post-OOV-drop nnz, which only feeds the
    coalesce-group size heuristic. `vocab`/`need_row_norms` are fixed
    RUN-WIDE (see `run_compute`'s `need_sparse_norms`) and carried through
    `.compact()` unchanged, so `.transfer()` needs no extra args beyond the
    `(r0, r1, device)` every corpus batch takes. This means a batch/filter-group made
    up entirely of `dot` searches still moves `row_norms` to the GPU whenever
    ANY search in the run needs cosine — a negligible extra transfer (one
    float per row in the batch) that a `dot` search's own `.score()` never
    reads, so it costs a little bandwidth, never correctness."""

    def __init__(
        self, row_offsets, indices, values, norms, vocab: np.ndarray, need_row_norms: bool,
        zero_gate_ok: bool = False, q_cache=None,
    ):
        self.row_offsets = row_offsets
        self.indices = indices
        self.values = values
        self.norms = norms
        self.vocab = vocab
        self.need_row_norms = need_row_norms
        # Whether this file's slices may use the cheap `raw == 0` no-overlap
        # gate (see `_zero_gate_file_ok`), and the run-wide `_SparseQueryCache`
        # for the signed fallback — both fixed per file / per run, carried
        # through `.compact()`/`_concat_sparse_batches` like `vocab`.
        self.zero_gate_ok = zero_gate_ok
        self.q_cache = q_cache

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
            SparseCorpusBatch(
                row_offsets, indices, values, norms, self.vocab, self.need_row_norms,
                self.zero_gate_ok, self.q_cache,
            ),
            orig_rows,
        )

    def transfer(self, r0: int, r1: int, device: str) -> "SparseBatchSlice":
        import torch

        Cb = _sparse_batch_to_csr(self.row_offsets, self.indices, self.values, r0, r1, self.vocab, device)
        row_norms = (
            torch.from_numpy(self.norms[r0:r1]).to(device, non_blocking=True)
            if self.need_row_norms else None
        )
        return SparseBatchSlice(Cb, row_norms, self.zero_gate_ok, self.q_cache)


# The zero-score no-overlap gate (`raw == 0` in `SparseBatchSlice.score`) is
# only valid when no product of a query value and a corpus value can flush to
# 0.0 in f32 (GPU denormal flush-to-zero would otherwise fake a structural
# miss): require min_positive(Q) * min_positive(C_file) above this. Real
# embedder weights (~1e-4..10) clear it by >20 orders of magnitude.
_ZERO_GATE_MIN_PRODUCT = 1e-30

# Max size of the dense `(vocab, slice_rows)` corpus operand used by swapped 
# sparse scoring. Larger slices fall back to the contiguous-transpose path to 
# avoid excessive temporary memory.
_SPARSE_SWAP_MAX_DENSE_BYTES = 512 << 20


def _zero_gate_file_ok(values: np.ndarray, q_nonneg: bool, q_min_pos: float) -> bool:
    """Per corpus file: may `SparseBatchSlice.score` use the cheap
    `raw == 0` no-overlap gate for this file's slices?  True iff the query
    side is non-negative, this file's values are STRICTLY positive, and the
    smallest possible cross-product stays comfortably normal (see
    `_ZERO_GATE_MIN_PRODUCT`). Strictness matters twice over: a negative
    value allows cancellation zeros (real candidates), and a STORED-ZERO
    entry is a structural overlap (it occupies a posting in an inverted
    index — Qdrant would surface the doc at score 0.0) that contributes 0 to
    the dot, so `raw == 0` would wrongly gate it — both fall back to the
    structural indicator path. `values` are the file's raw pre-coalesce
    entries — coalescing only ever SUMS same-sign values, so raw strict
    positivity implies scored positivity, and the raw min is a lower bound
    on the scored min (conservative)."""
    if not q_nonneg:
        return False
    if len(values) == 0:
        return True  # no entries -> nothing can overlap; the gate is vacuous
    vmin = float(values.min())
    if vmin <= 0.0:
        return False
    return q_min_pos * vmin > _ZERO_GATE_MIN_PRODUCT


def _dense_slice_t(Cb, values):
    """Densify a CSR corpus slice as `(vocab, n_rows)`.

    `values` controls what is scattered at each stored entry: use `Cb.values()`
    for scoring or ones for structural overlap checks. Direct assignment is safe
    because `Cb` is coalesced, so each `(row, col)` pair is unique.
    """
    import torch

    crow, col = Cb.crow_indices(), Cb.col_indices()
    row_ids = torch.repeat_interleave(
        torch.arange(Cb.shape[0], device=col.device), crow.diff()
    )
    out = torch.zeros((Cb.shape[1], Cb.shape[0]), dtype=values.dtype, device=values.device)
    out[col, row_ids] = values
    return out


class _SparseQueryCache:
    """Lazily cached representations of the run-wide sparse query matrix.

    Assumes every sparse spec shares the same `Q`; `rows` subsets are applied
    after scoring. If specs can ever receive different query tensors, this cache
    must be keyed by `Q`.

    Caches:
      values:    CSR query values for swapped sparse scoring.
      transpose: contiguous `Q.T` for the fallback scoring path.
      indicator: CSR query structure with ones for overlap checks.
    """

    def __init__(self):
        self._values = None
        self._transpose = None
        self._indicator = None

    def _csr(self, Q, vals_from_pattern):
        import torch

        nz = Q != 0  # (n_q, vocab) bool — one-time transient
        crow = torch.zeros(Q.shape[0] + 1, dtype=torch.int64, device=Q.device)
        torch.cumsum(nz.sum(1), 0, out=crow[1:])
        r, c = nz.nonzero(as_tuple=True)  # row-major order == valid CSR order
        return torch.sparse_csr_tensor(
            crow, c, vals_from_pattern(Q, r, c), size=tuple(Q.shape),
            check_invariants=False,
        )

    def values(self, Q):
        if self._values is None:
            self._values = self._csr(Q, lambda Q, r, c: Q[r, c])
        return self._values

    def transpose(self, Q):
        """Contiguous `Q.T` for the fallback sparse-scoring path."""
        if self._transpose is None:
            self._transpose = Q.t().contiguous()
        return self._transpose

    def indicator(self, Q):
        if self._indicator is None:
            import torch

            self._indicator = self._csr(
                Q, lambda Q, r, c: torch.ones(c.numel(), dtype=Q.dtype, device=Q.device)
            )
        return self._indicator


@dataclass
class SparseBatchSlice:
    """One on-device slice of a sparse corpus batch.

    `score()` marks every `(query, row)` cell with NO shared nonzero dimension
    as `-inf` rather than letting it surface as a real `0.0` hit. Sparse
    retrieval engines (Qdrant's inverted index included) can only ever return
    documents that share at least one token with the query — a zero-overlap
    document isn't "score zero", it's not a candidate at all. Without this,
    any query whose (post-filter) overlap set is smaller than k gets its
    top-K padded with arbitrary tie-ordered 0.0-score rows, which then
    pollute recall numbers computed against this ground truth. `-inf` rides
    the exact machinery that already exists for filtered-out cells: sunk by
    `_merge_topk`, truncated by the `valid = sc > -inf` write path, tallied
    by `warn_if_short`.

    The gate is SEMANTICALLY structural (shared nonzero dims), never
    `score == 0.0` in general — a signed sparse embedding can produce a
    genuine 0.0 from overlapping dimensions that cancel, and that row is a
    real candidate. Two implementations, selected per file by
    `zero_gate_ok` (see `_zero_gate_file_ok`):
    - Non-negative data (ReLU'd embedders like mGTE): cancellation is
      impossible, so `raw == 0` IS the structural gate — free, no extra
      matmul, underflow fenced by `_ZERO_GATE_MIN_PRODUCT`.
    - Signed data: an indicator spmm — the run-wide query-pattern CSR
      (`_SparseQueryCache.indicator`, built once) against this slice's densified
      indicator, counting shared dims exactly as before.

    The scoring spmm itself runs ONCE per slice regardless of how many
    metrics score it: `_masked_raw` caches the dot-scale, `-inf`-masked
    product, and `cosine` is derived from it by two scalar divisions (per
    corpus row, per query) — `-inf / positive == -inf`, so the gate survives
    the derivation. At ~200 nnz/row × 100k queries the second spmm was ~a
    third of all GPU work in the fineweb sparse GT run; the divisions are
    bandwidth-trivial. Trade-off: a cosine-ONLY run now briefly holds two
    (n_q, n_rows) matrices (the cached raw + the derived copy) where it
    previously held one — bounded by `sparse_batch_size`, and dwarfed by
    the spmm saved in the mixed-metric configs this cache exists for."""

    Cb: object  # torch.Tensor, sparse CSR (n_rows, vocab)
    row_norms: object  # torch.Tensor | None
    zero_gate_ok: bool = False  # may `raw == 0` stand in for the structural gate?
    q_cache: object = None  # run-wide _SparseQueryCache; lazy local if None
    _masked_raw: object = None  # lazy dot-scale masked (n_q, n_rows) — see docstring

    @property
    def n_rows(self) -> int:
        return self.Cb.shape[0]

    def _structural_no_overlap(self, Q):
        """Signed-data gate: (query-pattern CSR) @ (densified slice
        indicator, built transposed so no (n_q, …) transpose copy is ever
        made) == 0. Same FLOPs as the historical per-slice indicator spmm,
        but the dense operand is the ~300 MB corpus side, not a per-slice
        re-materialization of the multi-GB query side."""
        import torch

        ones = torch.ones(self.Cb.values().numel(), dtype=Q.dtype, device=Q.device)
        c_ind_t = _dense_slice_t(self.Cb, ones)  # ones, not values: a stored 0.0
                                                 # is still a structural overlap
        q_ind = (self.q_cache or _SparseQueryCache()).indicator(Q)
        return torch.matmul(q_ind, c_ind_t) == 0

    def score(self, Q, metric: str, q_norms=None):
        if self._masked_raw is None:
            raw = _sparse_scores(Q, self.Cb, self.q_cache)
            no_overlap = (raw == 0) if self.zero_gate_ok else self._structural_no_overlap(Q)
            self._masked_raw = raw.masked_fill_(no_overlap, float("-inf"))
        if metric == "dot":
            return self._masked_raw
        # cosine: a NEW tensor (dot may still read `_masked_raw`), then the
        # second division in place on that fresh copy.
        return self._masked_raw.div(self.row_norms.clamp_min(1e-12)[None, :]).div_(q_norms[:, None])


def _concat_dense_batches(batches: list[DenseCorpusBatch]) -> DenseCorpusBatch:
    """Concatenate several files' (already union-compacted) `DenseCorpusBatch`
    rows into one combined batch — used to coalesce many small per-file
    post-filter batches into fewer, larger GPU calls (see `run_compute`'s
    `_flush_coalesce_group`)."""
    return DenseCorpusBatch(np.concatenate([b.arr for b in batches], axis=0))


def _concat_sparse_batches(batches: list[SparseCorpusBatch]) -> SparseCorpusBatch:
    """Concatenate several files' (already union-compacted) `SparseCorpusBatch`
    rows into one combined CSR — the sparse analog of `_concat_dense_batches`.
    `indices`/`values`/`norms` are simple per-nnz/per-row concatenations, but
    `row_offsets` (CSR's cumulative nnz-per-row counts) can't just be
    concatenated as-is — every batch after the first restarts counting from
    0, so naive concatenation would produce a non-monotonic, garbage offset
    array. Each subsequent batch's offsets get shifted by the running total
    nnz seen so far, and its own leading `0` entry is dropped (it would
    exactly duplicate the previous batch's final, already-accumulated
    offset). `vocab`/`need_row_norms` are fixed run-wide (see
    `SparseCorpusBatch`'s own docstring), so the first batch's copy is
    authoritative for all of them."""
    indices = np.concatenate([b.indices for b in batches])
    values = np.concatenate([b.values for b in batches])
    norms = np.concatenate([b.norms for b in batches]) if batches[0].norms is not None else None
    offsets_parts = [batches[0].row_offsets]
    running_nnz = int(batches[0].row_offsets[-1])
    for b in batches[1:]:
        offsets_parts.append(b.row_offsets[1:] + running_nnz)
        running_nnz += int(b.row_offsets[-1])
    row_offsets = np.concatenate(offsets_parts).astype(np.int64)
    return SparseCorpusBatch(
        row_offsets, indices, values, norms, batches[0].vocab, batches[0].need_row_norms,
        # A coalesced group may mix files: the cheap gate needs EVERY part
        # to qualify; the indicator holder is run-wide, any part's copy works.
        all(b.zero_gate_ok for b in batches), batches[0].q_cache,
    )


@dataclass
class MultiVectorQuery:
    """The query side of a multivector (MaxSim) search, held on-device: every
    query's token vectors stacked into one `(total_query_tokens, D)` matrix
    (`flat`) plus a length-`n_q+1` token-offset array (`offsets`), so query `q`'s
    tokens are `flat[offsets[q]:offsets[q+1]]`. This is what a spec's `Q` is for
    a multivector search — passed through `_process_batch_group` unchanged to
    `MultiVectorBatchSlice.score`. Both implementations (torch and the hybrid
    Triton reducer) tile the query axis by whole queries (`query_block`) to
    bound their per-slice intermediate. `n_q` is stored explicitly because a
    trailing run of zero-token queries is unrecoverable from `flat`.

    `offsets` (device) drives the on-device ragged boundaries; `offsets_cpu`
    (the same values, host numpy) drives the Python block-loop slicing so
    `int(offsets[qs])` never forces a per-block CUDA→CPU sync (see `score`)."""

    flat: object       # torch.Tensor (total_query_tokens, D)
    offsets: object    # torch.Tensor (n_q + 1,) int64, ON DEVICE — for reductions
    offsets_cpu: object  # np.ndarray (n_q + 1,) int64 — for host block-loop slicing (no sync)
    n_q: int
    query_block: int | None  # queries per query-axis tile (None = all at once)
    kernel: str = "torch"  # torch reference, fused Triton, or compatible auto-selection

    @cached_property
    def max_block_tokens(self) -> int:
        """Largest actual token count in any whole-query score block."""
        block = self.query_block or self.n_q
        if self.n_q == 0:
            return 0
        return max(
            int(self.offsets_cpu[min(start + block, self.n_q)])
            - int(self.offsets_cpu[start])
            for start in range(0, self.n_q, block)
        )

    @cached_property
    def flat_normalized(self):
        """Per-token L2-normalized copy of `flat`, built once per run.

        Cosine scoring previously re-ran `F.normalize` on the query block
        inside every (slice x block) iteration — identical output every time,
        since `F.normalize` is strictly row-wise. Caching trades one extra
        Q-sized tensor (never materialized on a dot-only run — this property
        is lazy) for eliminating that per-slice recompute; row-slicing this
        cached copy is bit-identical to normalizing the slice."""
        import torch.nn.functional as F

        return F.normalize(self.flat, dim=1)


def _segment_max_over_cols(P, col_group, n_groups: int):
    """`P` is `(rows, n_cols)`; `col_group` maps each column to a group id in
    `[0, n_groups)`. Returns `(rows, n_groups)` where entry `(r, g)` is the MAX
    of `P[r, cols-in-group-g]`, or `-inf` for a group with no columns (a
    zero-token doc — its `-inf` propagates through the later sum, marking it a
    non-candidate exactly like the sparse no-overlap gate). Ragged segment-max
    via `scatter_reduce_(amax)`, which runs on CPU and GPU alike."""
    import torch

    out = torch.full((P.shape[0], n_groups), float("-inf"), dtype=P.dtype, device=P.device)
    idx = col_group.unsqueeze(0).expand(P.shape[0], -1)
    out.scatter_reduce_(1, idx, P, reduce="amax", include_self=True)
    return out


class MultiVectorCorpusBatch:
    """Decoded multivector corpus tokens for one file, ragged: `flat_tokens`
    is `(total_tokens, D)` and `doc_offsets` (length `n_rows+1`) delimits each
    doc's token span. Exposes the same `.n_rows`/`.nbytes`/`.compact`/
    `.transfer` surface as `Dense`/`SparseCorpusBatch` so `run_compute`'s
    per-file loop never branches on vector_type. Query-axis tiling
    (`query_block`) lives on the query object (`MultiVectorQuery`), NOT here —
    a corpus batch has no business knowing how the query set is tiled."""

    def __init__(self, doc_offsets: np.ndarray, flat_tokens: np.ndarray):
        self.doc_offsets = doc_offsets
        self.flat_tokens = flat_tokens

    @property
    def n_rows(self) -> int:
        return len(self.doc_offsets) - 1

    @property
    def nbytes(self) -> int:
        return int(self.flat_tokens.nbytes)

    def compact(self, keep: np.ndarray) -> tuple["MultiVectorCorpusBatch", np.ndarray]:
        doc_offsets, flat_tokens, orig_rows = _compact_multivector_rows(
            self.doc_offsets, self.flat_tokens, keep
        )
        return MultiVectorCorpusBatch(doc_offsets, flat_tokens), orig_rows

    def transfer(
        self, r0: int, r1: int, device: str, *, pin_memory: bool = False
    ) -> "MultiVectorBatchSlice":
        import torch

        t0, t1 = int(self.doc_offsets[r0]), int(self.doc_offsets[r1])
        # doc_offsets for this slice, rebased so the slice's first token is 0.
        local_off_cpu = torch.from_numpy(
            (self.doc_offsets[r0 : r1 + 1] - t0).astype(np.int64, copy=False)
        )
        flat_cpu = torch.from_numpy(self.flat_tokens[t0:t1])
        if pin_memory:
            local_off_cpu = local_off_cpu.pin_memory()
            flat_cpu = flat_cpu.pin_memory()
        local_off = local_off_cpu.to(device, non_blocking=True)
        flat = flat_cpu.to(device, non_blocking=True)
        return MultiVectorBatchSlice(flat, local_off)


@dataclass
class MultiVectorBatchSlice:
    """One on-device slice of a multivector corpus batch. `score()` computes
    the MaxSim score matrix `(n_q, n_rows)` — the SAME shape every other
    slice type returns, so `_merge_topk`, the shared loop, and merge all work
    unchanged.

    MaxSim(q, d) = sum over q's tokens of (max over d's tokens of q_tok . d_tok).
    The established torch path computes per-query-axis tiles and materializes
    `(block_query_tokens × slice_doc_tokens)` product `P`, then performs two
    ragged reductions. The hybrid `triton_reduce` path retains that cuBLAS
    product but fuses both reductions into one kernel. A zero-token query or document 
    is `-inf`in every implementation.

    `dot` (Qdrant's default) uses the raw tokens; `cosine` L2-normalizes every
    token (query and corpus) FIRST — genuinely a separate matmul, not a
    scalar rescale of the dot result, because normalizing changes each token's
    per-token argmax. The scoring runs once per distinct metric via
    `_process_batch_group`'s `score_cache`, so a dot-only run never pays the
    cosine normalization."""

    flat: object       # torch.Tensor (slice_total_tokens, D)
    doc_offsets: object  # torch.Tensor (n_rows + 1,) int64, rebased to 0

    @property
    def n_rows(self) -> int:
        return self.doc_offsets.shape[0] - 1

    def record_stream(self, stream) -> None:
        """Keep transfer-stream allocations alive through their consumer."""
        self.flat.record_stream(stream)
        self.doc_offsets.record_stream(stream)

    def score(self, Q: "MultiVectorQuery", metric: str, q_norms=None):
        # `q_norms` is unused: it's the per-query cosine scalar the dense/sparse
        # slices take, but multivector cosine normalizes each TOKEN (not a
        # per-query rescale), so there's no scalar to apply. The parameter stays
        # for the uniform `slice.score(Q, metric, q_norms)` interface the shared
        # loop calls across every vector_type.
        import torch
        import torch.nn.functional as F

        if metric not in ("dot", "cosine"):
            raise ValueError(
                f"multivector metric must be 'dot' or 'cosine', got {metric!r}"
            )
        dev = self.flat.device
        n_rows = self.n_rows
        n_q = Q.n_q
        if n_rows == 0 or self.flat.shape[0] == 0:
            # No corpus tokens in this slice — every doc is a non-candidate.
            return torch.full(
                (n_q, n_rows), float("-inf"), dtype=self.flat.dtype, device=dev
            )
        if Q.flat.shape[0] == 0:
            # Every query is zero-token — all remain non-candidates.
            return torch.full(
                (n_q, n_rows), float("-inf"), dtype=self.flat.dtype, device=dev
            )
        if Q.flat.shape[0] and Q.flat.shape[1] != self.flat.shape[1]:
            raise ValueError(
                f"multivector token dim mismatch: query D={Q.flat.shape[1]} vs "
                f"corpus D={self.flat.shape[1]} — the query and corpus multivector "
                "columns must share one token dimension"
            )

        C = F.normalize(self.flat, dim=1) if metric == "cosine" else self.flat
        triton_compatible = (
            dev.type == "cuda"
            and Q.flat.device == dev
            and self.flat.dtype == torch.float32
            and Q.flat.dtype == torch.float32
            and self.flat.shape[1] > 0
        )
        selected_kernel = Q.kernel
        if selected_kernel != "torch":
            if triton_compatible:
                try:
                    from nova_bf.multivector_kernels import fused_ragged_maxsim_reduce
                except ImportError as exc:
                    if selected_kernel == "triton_reduce":
                        raise RuntimeError(
                            f"params.multivector_kernel={selected_kernel!r} requires Triton; "
                            "install a CUDA PyTorch distribution that includes it"
                        ) from exc
                    selected_kernel = "torch"
                else:
                    if selected_kernel == "auto":
                        selected_kernel = "triton_reduce"
            elif selected_kernel == "triton_reduce":
                raise RuntimeError(
                    f"params.multivector_kernel={selected_kernel!r} requires same-device CUDA "
                    "float32 query/document tokens with a positive token dimension"
                )
            else:
                selected_kernel = "torch"

        out = torch.full(
            (n_q, n_rows), float("-inf"), dtype=self.flat.dtype, device=dev
        )
        # Only the torch reference needs a token->doc map; the hybrid Triton
        # reducer consumes the contiguous offsets directly.
        col_doc = None
        if selected_kernel != "triton_reduce":
            col_doc = torch.repeat_interleave(
                torch.arange(n_rows, device=dev), self.doc_offsets.diff()
            )
        qoff = Q.offsets            # device — feeds the on-device reductions below
        qoff_cpu = Q.offsets_cpu    # host — feeds the Python block-loop bounds (no CUDA sync)
        # Row-slicing the run-wide normalized cache is bit-identical to
        # normalizing each block (F.normalize is strictly row-wise) and skips
        # a per-(slice x block) recompute.
        q_source = Q.flat_normalized if metric == "cosine" else Q.flat
        block = Q.query_block or n_q
        for qs in range(0, n_q, block):
            qe = min(qs + block, n_q)
            t0, t1 = int(qoff_cpu[qs]), int(qoff_cpu[qe])  # host read: no per-block device sync
            if t1 == t0:
                continue  # every query in this block is zero-token -> left -inf (see below)
            Qb = q_source[t0:t1]
            P = Qb @ C.T                                   # (block_q_tokens, slice_doc_tokens)
            if selected_kernel == "triton_reduce":
                # Written straight into this block's rows of `out` (the kernel
                # takes a row stride) — no separate device-to-device copy.
                fused_ragged_maxsim_reduce(
                    P,
                    qoff,
                    self.doc_offsets,
                    query_start=qs,
                    query_token_base=t0,
                    n_queries=qe - qs,
                    out=out[qs:qe],
                )
                continue
            M = _segment_max_over_cols(P, col_doc, n_rows)  # (block_q_tokens, n_rows)
            # Segment-SUM over each query's tokens, written STRAIGHT into this
            # block's output rows (zeroed first): `index_add_` into the `out`
            # view avoids a separate `(block_queries × n_rows)` temporary — at
            # scale (n_q=100k, a large corpus batch) that temporary equals the
            # output-slice size, a multi-GB transient on the GPU. A non-candidate
            # doc's `-inf` in `M` sums to `-inf`, surviving as a non-candidate.
            counts = qoff[qs + 1 : qe + 1] - qoff[qs:qe]   # tokens per query in this block
            row_q = torch.repeat_interleave(torch.arange(qe - qs, device=dev), counts)
            dest = out[qs:qe]
            dest.zero_()
            dest.index_add_(0, row_q, M)
            # A ZERO-TOKEN query has no tokens to score, so it retrieves nothing:
            # mark it a non-candidate (-inf) EVERYWHERE — identical to a whole
            # zero-token block (the `continue` above) and to the sparse
            # zero-support gate. Without this, a zero-token query sharing a block
            # with a non-empty one would keep the `dest.zero_()` value (0.0
            # against every doc) and its top-K would fill with arbitrary tied
            # 0.0-score rows — AND the result would depend on how `query_block`
            # happens to tile the query axis (a pure performance knob), breaking
            # tiling-invariance. Qdrant rejects zero-token multivector queries
            # outright; nova-bf instead returns them cleanly as "no hits".
            # Unconditional (no `.any()` guard): when no query is zero-token the
            # mask is all-False and this is a no-op scatter — cheaper than the
            # `bool(...)` guard, which would force a device->host sync every
            # query block on the GPU.
            dest[counts == 0] = float("-inf")
        return out


def _concat_multivector_batches(batches: list["MultiVectorCorpusBatch"]) -> "MultiVectorCorpusBatch":
    """Concatenate several files' (already union-compacted) multivector batches
    into one — the ragged analog of `_concat_sparse_batches`. `flat_tokens` is a
    plain per-token concatenation; `doc_offsets` are shifted by the running
    token total and their duplicate leading `0` dropped, exactly as CSR
    row_offsets are."""
    # Skip zero-ROW token arrays before concatenating: an all-zero-token corpus
    # shard decodes to a width-0 `(0, 0)` `flat_tokens` (the decoder can't infer
    # D with no tokens — see `multivector_to_ragged`), which would otherwise make
    # `np.concatenate(..., axis=1-mismatch)` throw against real `(m, D)` shards.
    # A zero-row array contributes no tokens anyway; its docs still exist as rows
    # and ride along via `doc_offsets` below (they score `-inf`, non-candidates).
    # This mirrors `load_queries_multivector`'s own `nonempty` guard on the query
    # side. If EVERY part is empty the group has no tokens at all, so the width is
    # irrelevant (`MultiVectorBatchSlice.score` short-circuits on 0 tokens).
    flat_parts = [b.flat_tokens for b in batches if b.flat_tokens.shape[0] > 0]
    flat = np.concatenate(flat_parts, axis=0) if flat_parts else batches[0].flat_tokens
    offsets_parts = [batches[0].doc_offsets]
    running = int(batches[0].doc_offsets[-1])
    for b in batches[1:]:
        offsets_parts.append(b.doc_offsets[1:] + running)
        running += int(b.doc_offsets[-1])
    doc_offsets = np.concatenate(offsets_parts).astype(np.int64)
    return MultiVectorCorpusBatch(doc_offsets, flat)


def _merge_topk(top_key, top_enc, parts: list[tuple], k: int):
    """Merge pending candidate columns into the running `(top_key, top_enc)` top-k state.

    The state stores packed keys rather than scores. `nova_bf.tiebreak.pack`
    combines each score with its row ordinal into a single int64 whose ordering
    defines both score order and deterministic tie-breaking. Because the
    transform is bijective, the original score can be recovered exactly with
    `unpack_score` during decode.

    `parts` contains `(keys, encoded)` pairs accumulated across slices in corpus
    order. Each `keys` tensor has shape `(n_q, cols)` with `cols <= k`; wider
    slices are reduced to their local top-k before being appended. `encoded` is
    either a matching `(n_q, cols)` tensor for an already-selected part or a
    1-D `(cols,)` tensor of fully encoded corpus row IDs, which is broadcast
    across queries here.

    Pending parts are concatenated with the running state and reduced with a
    single top-k operation. Since selection is performed on the packed key,
    the final result is independent of slice boundaries, candidate grouping,
    and flush timing.
    """
    import torch

    from nova_bf import merge_triton

    # Check whether the Triton fold is viable before preparing its inputs.
    #  This is especially important for large query counts, where the temporary
    # copies can be substantial.
    pending_width = sum(int(p.shape[1]) for p, _ in parts)
    if merge_triton.enabled(top_key) and k + pending_width <= merge_triton.MAX_BLOCK:
        # A single part is the common case. Multiple parts arise from narrow
        # slices, such as file tails or heavily filtered batches; combine them once
        # here rather than concatenating each with the full running state.
        if len(parts) > 1:
            pk = torch.cat([p for p, _ in parts], dim=1)
            pe = torch.cat(
                [e if e.ndim == 2 else e.unsqueeze(0).expand(p.shape[0], -1) for p, e in parts],
                dim=1,
            )
        else:
            pk, pe = parts[0]
        # The Triton kernel requires contiguous inputs. Sparse scoring can produce
        # transposed/non-contiguous tensors, so materialize them here rather than
        # forcing the more expensive portable merge.
        if not pk.is_contiguous():
            pk = pk.contiguous()
        if not pe.is_contiguous():
            pe = pe.contiguous()
        if merge_triton.available(top_key, top_enc, pk, pe, k):
            try:
                return merge_triton.fold(top_key, top_enc, pk, pe, k)
            except torch.cuda.OutOfMemoryError:
                # OOM does not indicate an unsupported kernel configuration, and
                # the portable path requires even more temporary memory. Preserve
                # the original error rather than permanently disabling Triton.
                raise
            except Exception as exc:
                merge_triton.disable(exc)
        # Preparation has already combined the pending inputs, so reuse that result
        # if execution falls through to the portable path.
        parts = [(pk, pe)]

    # Each intermediate is del'd as soon as the next line no longer needs it
    merged_k = torch.cat([top_key] + [p for p, _ in parts], dim=1)
    merged_e = torch.cat(
        [top_enc]
        + [
            e if e.ndim == 2 else e.unsqueeze(0).expand(p.shape[0], -1)
            for p, e in parts
        ],
        dim=1,
    )
    new_top_key, idx = torch.topk(merged_k, k=k, dim=1, sorted=False)
    del merged_k
    return new_top_key, merged_e.gather(1, idx)


def _sample_mean_doc_tokens(store: Store, mine: list, column: str) -> float:
    """Mean tokens-per-doc from this worker's FIRST corpus file (a metadata-
    cheap, single-file read) — used only to derive multivector tile sizes from
    `multivector_token_budget`. Returns 1.0 if the worker has no files or the
    sample is empty (a safe floor: it just makes the derived doc-tile larger)."""
    if not mine:
        return 1.0
    table = store.read_columns(mine[0][1].read_path, [column])
    doc_offsets, _ = multivector_to_ragged(table[column])
    n = len(doc_offsets) - 1
    total = int(doc_offsets[-1])
    return total / n if n > 0 and total > 0 else 1.0


def _resolve_multivector_tiles(
    budget: int, mean_q_tokens: float, mean_doc_tokens: float,
    configured_bs: int | None, configured_qb: int | None,
) -> tuple[int | None, int | None]:
    """Derive `(multivector_batch_size, multivector_query_block)` from a target
    peak element count for the per-slice score matrix `P = (block_query_tokens
    × slice_doc_tokens)`. Split geometrically — aim each axis near
    `sqrt(budget)` elements — then convert tokens back to items via the two
    means. An explicitly-configured knob is passed through untouched; only a
    `None` knob is filled in. Every derived value is floored at 1 (a
    zero/negative tile would empty the batch loop)."""
    import math

    axis = math.sqrt(max(1, budget))
    bs = configured_bs if configured_bs is not None else max(1, int(axis / max(1e-9, mean_doc_tokens)))
    qb = configured_qb if configured_qb is not None else max(1, int(axis / max(1e-9, mean_q_tokens)))
    return bs, qb


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
    `_unpack_query_axis`.

    `n_queries` here is that FILTER's query-row union, not the queries file's
    (see `run_compute`'s `filter_rows`)."""
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
    off the packed array rather than approximated.

    "any query" means any query the filter's own specs OWN, since that is what
    its mask now spans (`run_compute`'s `filter_rows`). Never LOOSER than when
    the mask covered the whole queries file, and still a superset of what the
    per-query `cell_mask` keeps, which is all the union has to be.

    Tighter only when the foreign rows held REAL values, though — don't sell
    it as a win it usually isn't. A null or empty phrase is token-less, and a
    token-less phrase in a `must` matches nothing while a null `should` slot
    contributes nothing, so a sentinel-filled foreign row's mask was already
    all-False and added nothing to this OR. Every filtered config in this repo
    is that case, so for them the narrowing changes this union by zero; its
    payoff is `keeps[f]`'s allocation, not the shared grid."""
    parts = [m.any(axis=0) if m.ndim == 2 else m for m in (keeps[f] for f in filters)]
    return np.logical_or.reduce(parts)


def _ragged_batch_ranges(
    offsets: np.ndarray, max_rows: int, max_tokens: int | None
) -> list[tuple[int, int]]:
    """Split ragged rows by both item count and actual packed token count.

    Rows are never split. If one row alone exceeds ``max_tokens``, it forms a
    one-row slice so iteration still progresses. Zero-token rows are included
    without consuming the token budget.
    """
    n_rows = len(offsets) - 1
    ranges: list[tuple[int, int]] = []
    r0 = 0
    while r0 < n_rows:
        row_cap = min(n_rows, r0 + max_rows)
        if max_tokens is None:
            r1 = row_cap
        else:
            token_cap = int(offsets[r0]) + max_tokens
            token_cap_row = int(np.searchsorted(offsets, token_cap, side="right") - 1)
            r1 = min(row_cap, max(r0 + 1, token_cap_row))
        ranges.append((r0, r1))
        r0 = r1
    return ranges


def _process_batch_group(
    batch, member_idxs: list[int], specs: list[SearchSpec], spec_Q, spec_q_norms,
    spec_top_key, spec_top_enc,
    batch_size: int | None, gidx: int, device: str, orig_rows: np.ndarray | None, select,
    spec_qsel: list, spec_qrows: list,
    encoded_row_ids: np.ndarray | None = None,
    ordinal_base: int = 0,
    ordinal_row_ids: np.ndarray | None = None,
    multivector_token_budget: int | None = None,
    multivector_double_buffer: bool = False,
) -> float:
    """The shared per-vector_type primitive behind `_process_shared_batch`:
    iterate `batch` in `batch_size`-row slices, transfer each slice once
    (`batch.transfer`), score it once per
    DISTINCT metric among `member_idxs` (`score_cache` — every member needing
    that metric reads the same tensor), and merge each member's own top-k
    from those shared columns via `_merge_topk`.

    `orig_rows` maps a slice position to its TRUE file-row number (used to
    index a filter's keep-mask, and — when `encoded_row_ids` is `None` — to
    build `_merge_topk`'s output-id encoding too): `None` means position IS
    the true row, either because `batch` is the raw, whole file (some search
    of this vector_type is unfiltered and needs every row), or because
    `batch` is several files' rows COALESCED into one (see `run_compute`'s
    `_flush_coalesce_group`) whose keep-masks were already rebuilt to align
    with the coalesced batch's own row order — either way, nothing to remap
    for keep-mask indexing. An array means `batch` is a SINGLE file's own
    compacted batch (the union of every active filter's surviving rows —
    see `run_compute`'s `has_baseline` and `_union_keep`) and this maps back
    to that one file's true rows.

    `encoded_row_ids`, independently, is `_merge_topk`'s output-id source:
    `None` means single-file — `gidx * MAX_ROWS_PER_FILE + rows` is computed
    here (cheaply, directly on device, no CPU round-trip); a caller-supplied
    array means it's already the fully-encoded id per row (needed for a
    coalesced batch, where different rows came from different files, so no
    single scalar `gidx` can encode all of them). This is deliberately a
    SEPARATE concept from `orig_rows`: a coalesced batch needs identity
    keep-mask indexing (rebuilt masks already match its own row order) but
    non-identity, per-source-file id encoding — the two purposes coincide
    for a single file but diverge once rows from several files share one
    batch.

    `ordinal_base`/`ordinal_row_ids` carry the row's tie-break ordinal, which
    rides in the low half of the packed selection key (see `nova_bf.tiebreak`)
    and decides which of two EXACTLY-tied candidates survives. They split for
    the same reason `encoded_row_ids` does: under `tiebreak='ordinal'` the
    value is just this worker's running row counter, so a scalar base plus the
    row index reconstructs it on device with no host array; under
    `tiebreak='id'`, or for any coalesced batch whose rows came from several
    files, no scalar covers it and the caller passes the array, already aligned
    to batch position.

    `select(m, rows, true_rows, cache) -> (sel_rows, sel_cols, cell_mask)` is
    the per-member filtering strategy (see `_process_shared_batch`):
    `sel_rows is None` skips the merge entirely for this member/slice (e.g.
    a filter keeping zero rows here); `sel_cols` is either `None` (member is
    unfiltered or per-query-filtered — use the slice's rows unchanged) or a
    column-index tensor used to mask the score matrix (and `encoded_rows`,
    identically) down to that member's own (uniform) filter's surviving
    columns; `cell_mask` is either `None` (no per-(query,row) masking
    needed) or a `(n_queries, len(rows))` boolean tensor applied via
    `masked_fill` — a per-query filter's own rows vary BY QUERY, so unlike a
    uniform filter it can't be expressed as one shared column selection;
    every column stays, and individual (query, row) cells get invalidated
    instead. `true_rows` indexes a filter's keep-mask (sized to match
    `batch`'s own row order, not necessarily one whole file — see above): a
    plain `slice(r0, r1)` when `orig_rows is None` (a cheap view, no copy),
    or `orig_rows`'s corresponding array slice otherwise. `cache` is a fresh
    dict per r0-slice for `select` to memoize per-filter lookups shared
    across members — it does not persist across slices, since the mask is
    slice-relative.

    `spec_qsel`/`spec_qrows` are the per-member `SearchSpec.rows` selectors
    (one entry per spec, `None` for a spec that owns every query — see
    `_row_selector`). They index the SAME rows in two different spaces
    and are NOT interchangeable: `spec_qsel[m]` addresses the score matrix,
    whose query axis spans this vector_type's row UNION, while `spec_qrows[m]`
    addresses a per-query filter mask, whose query axis spans that FILTER's
    row union (`run_compute`'s `filter_rows`; the whole file for a filter that
    kept full height — see `_pack_query_axis`). Both are required arguments —
    a member that owns every query passes `None` explicitly — because the two
    spaces coincide often enough (whenever both unions are the whole file)
    that a defaulted or swapped selector reads plausible rows on most fixtures
    and the wrong ones on the rest, with no error either way.

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
    max_doc_tokens = None
    if isinstance(batch, MultiVectorCorpusBatch) and multivector_token_budget is not None:
        # Every backend (torch and triton_reduce alike) materializes the
        # (block_query_tokens x slice_doc_tokens) matrix P, so the budget is
        # always enforced against the worst actual query block.
        max_query_tokens = max(
            1, max(spec_Q[m].max_block_tokens for m in member_idxs)
        )
        max_doc_tokens = max(1, multivector_token_budget // max_query_tokens)
    if isinstance(batch, MultiVectorCorpusBatch):
        ranges = _ragged_batch_ranges(batch.doc_offsets, step, max_doc_tokens)
    else:
        ranges = [(r0, min(r0 + step, n_rows)) for r0 in range(0, n_rows, step)]
    # How many members read each distinct metric's score matrix — used below
    # to skip caching one nobody else will reuse (the score-matrix analog of
    # `filter_share_count`): a single-reader (n_q, rows) matrix — 1.6 GiB at
    # n_q=100k / rows=4096 — gets collected right after its own merge instead
    # of sitting in `score_cache` through every later member's matmul+merge.
    metric_share_count = Counter(specs[m].metric for m in member_idxs)
    # Dense only: when 2+ DISTINCT metrics score this batch, derive all of them
    # from ONE raw Gram instead of one GEMM each (see `DenseBatchSlice`). A
    # single-metric batch keeps the unshared path, so it pays neither the extra
    # resident matrix nor any change in its float32 output.
    if isinstance(batch, DenseCorpusBatch):
        batch.share_gram = len(metric_share_count) > 1

    # Amortized running top-k (see _merge_topk): per-slice candidates are
    # buffered (each part pre-topk'd to <= k columns, so the buffer holds at
    # most ~2k columns per member) and folded with ONE topk once >= k pending
    # columns accumulate, instead of one topk per slice. Flushed for every
    # member before this function returns, so callers still observe a fully
    # merged running state per batch.
    pending: dict[int, list[tuple]] = {m: [] for m in member_idxs}
    pending_cols: dict[int, int] = {m: 0 for m in member_idxs}

    def _flush_pending(m: int) -> None:
        if not pending[m]:
            return
        parts = pending[m]
        pending[m] = []
        pending_cols[m] = 0
        spec_top_key[m], spec_top_enc[m] = _merge_topk(
            spec_top_key[m], spec_top_enc[m], parts, specs[m].k
        )

    def _flush_all_pending() -> None:
        for m in member_idxs:
            _flush_pending(m)

    def process_slice(r0: int, r1: int, sl) -> None:
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

        if encoded_row_ids is None:
            encoded_rows = gidx * MAX_ROWS_PER_FILE + rows
        else:
            encoded_rows = torch.from_numpy(encoded_row_ids[r0 : r0 + sl.n_rows]).to(device, non_blocking=True)

        # Tie-break ordinals for these rows (see `nova_bf.tiebreak`).
        if ordinal_row_ids is None:
            ordinals = ordinal_base + rows
        else:
            ordinals = torch.from_numpy(
                ordinal_row_ids[r0 : r0 + sl.n_rows].astype(np.int64, copy=False)
            ).to(device, non_blocking=True)

        score_cache: dict[str, object] = {}
        cache: dict[object, object] = {}  # keyed by whatever select() memoizes on (e.g. Filter)
        for m in member_idxs:
            s = specs[m]
            scores = score_cache.get(s.metric)
            if scores is None:
                scores = sl.score(spec_Q[m], s.metric, spec_q_norms[m])
                if metric_share_count[s.metric] > 1:
                    score_cache[s.metric] = scores

            sel_rows, sel_cols, cell_mask = select(m, rows, true_rows, cache)
            if sel_rows is None:
                continue
            sel_scores = scores if sel_cols is None else scores[:, sel_cols]
            sel_encoded = encoded_rows if sel_cols is None else encoded_rows[sel_cols]
            sel_ordinals = ordinals if sel_cols is None else ordinals[sel_cols]
            # Query-row subset (`SearchSpec.rows`). The score matrix spans this
            # vector_type's whole row union — every spec sharing the type reads
            # the same one — so a spec that owns only part of it slices here,
            # BEFORE the top-k.
            qsel = spec_qsel[m]
            if qsel is not None:
                sel_scores = sel_scores[qsel]
            if cell_mask is not None:
                if qsel is not None:
                    # `cell_mask` is built over the FULL query axis, so it is
                    # indexed by FILE row (`spec_qrows`), not by position
                    # within this spec's slice (`spec_qsel`). Rebinding, never
                    # mutating: the mask may be cached and shared with another
                    # spec that has a different subset.
                    cell_mask = cell_mask[spec_qrows[m]]
                sel_scores = sel_scores.masked_fill(~cell_mask, float("-inf"))

            # Append to the pending buffer (pre-topk wide slices down to k so
            # the buffer, and any slice score matrix it would otherwise pin
            # alive, stays bounded); merge only once >= k columns accumulate.
            #
            # This pre-top-K DISCARDS PERMANENTLY — a candidate dropped here is
            # never seen again, by the fold or by `merge` — so it has to select
            # on the packed key, not the score. It is the site where an
            # unstable `topk` used to silently decide ties.
            if sel_scores.shape[1] > s.k:
                part_key, part_local = pack_topk(sel_scores, sel_ordinals, s.k)
                part_enc = sel_encoded[part_local]
                del part_local
            else:
                part_key, part_enc = pack(sel_scores, sel_ordinals), sel_encoded
            pending[m].append((part_key, part_enc))
            pending_cols[m] += part_key.shape[1]
            if pending_cols[m] >= s.k:
                _flush_pending(m)

    use_double_buffer = (
        multivector_double_buffer
        and isinstance(batch, MultiVectorCorpusBatch)
        and str(device).startswith("cuda")
        and len(ranges) > 1
    )

    # Whole-batch device residency (synchronous path only): when the batch's
    # token matrix comfortably fits in free CUDA memory, transfer it ONCE and
    # score every slice as a device-side view — per-slice H2D and allocation
    # overhead disappear. Slicing `flat_gpu` by token range is a zero-copy
    # contiguous view; the rebased offsets are a tiny on-device subtraction
    # (no host sync). Deliberately NOT taken when double buffering was
    # requested: the resident upload is one big serial copy, while double
    # buffering exists precisely to hide per-slice H2D behind compute — a
    # compute-bound run with `multivector_double_buffer: true` must never get
    # slower by having its overlap silently replaced with an upfront copy.
    if (
        not use_double_buffer
        and isinstance(batch, MultiVectorCorpusBatch)
        and str(device).startswith("cuda")
        and batch.flat_tokens.shape[0] > 0
        and len(ranges) > 1
    ):
        free_mem, _ = torch.cuda.mem_get_info(torch.device(device))
        # mem_get_info reports torch's cached-but-unallocated pool as used, so
        # a warm process would under-report free memory and stop taking this
        # path after its first few large batches — count that reclaimable
        # cache back in (the allocator returns it under pressure).
        free_mem += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        need = int(batch.flat_tokens.nbytes) + int(batch.doc_offsets.nbytes)
        if need <= _MV_RESIDENT_FREE_FRACTION * free_mem:
            flat_gpu = torch.from_numpy(batch.flat_tokens).to(device)
            off_gpu = torch.from_numpy(
                np.ascontiguousarray(batch.doc_offsets, dtype=np.int64)
            ).to(device)
            for r0, r1 in ranges:
                tk0, tk1 = int(batch.doc_offsets[r0]), int(batch.doc_offsets[r1])
                sl = MultiVectorBatchSlice(
                    flat_gpu[tk0:tk1], off_gpu[r0 : r1 + 1] - off_gpu[r0]
                )
                process_slice(r0, r1, sl)
            _flush_all_pending()
            return time.perf_counter() - t0

    if not use_double_buffer:
        for r0, r1 in ranges:
            process_slice(r0, r1, batch.transfer(r0, r1, device))
        _flush_all_pending()
        return time.perf_counter() - t0

    transfer_stream = torch.cuda.Stream(device=torch.device(device))
    compute_stream = torch.cuda.current_stream(device=torch.device(device))

    # Pinned staging RING (two buffers, sized once for the largest slice) —
    # NOT a fresh pin_memory() per slice. Ragged slices make almost every
    # per-slice pinned allocation a unique size, and torch's pinned pool
    # caches freed blocks by size without returning them to the OS, so
    # per-slice pinning grew unswappable host memory by ~the size of every
    # slice ever staged (~one file's tokens per file scanned) — observed as
    # a host OOM on a 32 GB worker by its third 4.3 GB file. Two fixed
    # buffers bound pinned memory at 2x the largest slice for the whole run.
    stage_rows = max(r1 - r0 for r0, r1 in ranges)
    stage_tokens = max(
        int(batch.doc_offsets[r1] - batch.doc_offsets[r0]) for r0, r1 in ranges
    )
    stages = [
        (
            torch.empty(
                (stage_tokens, batch.flat_tokens.shape[1]),
                dtype=torch.float32,
                pin_memory=True,
            ),
            torch.empty(stage_rows + 1, dtype=torch.int64, pin_memory=True),
        )
        for _ in range(2)
    ]

    def prefetch(index: int, r0: int, r1: int):
        # Ring-reuse safety: stage[index % 2] was last used by slice
        # index - 2, and the backpressure below synchronized on slice
        # index - 2's COMPUTE event before this call — compute waits on its
        # H2D, so that H2D (the only reader of this buffer) has completed.
        flat_stage, off_stage = stages[index % 2]
        tk0, tk1 = int(batch.doc_offsets[r0]), int(batch.doc_offsets[r1])
        flat_host = flat_stage[: tk1 - tk0]
        off_host = off_stage[: r1 - r0 + 1]
        flat_host.copy_(torch.from_numpy(batch.flat_tokens[tk0:tk1]))
        off_host.copy_(
            torch.from_numpy(
                (batch.doc_offsets[r0 : r1 + 1] - tk0).astype(np.int64, copy=False)
            )
        )
        with torch.cuda.stream(transfer_stream):
            sl = MultiVectorBatchSlice(
                flat_host.to(device, non_blocking=True),
                off_host.to(device, non_blocking=True),
            )
            ready = torch.cuda.Event()
            ready.record(transfer_stream)
        return sl, ready

    # Backpressure: bound CPU run-ahead to the two slices double buffering
    # actually wants. Without this, the CPU races through the whole loop
    # enqueueing async work while each consumed slice's record_stream'd
    # buffers stay unreclaimable until the GPU reaches them — peak device
    # memory then grows with the CPU's lead (observed as a transient CUDA
    # OOM on a 24 GB A10G in a compute-bound run). Waiting on the
    # second-newest compute event is free when transfers are the bottleneck
    # (the event has already fired) and throttles exactly when compute is.
    inflight_compute: deque = deque()
    sl, ready = prefetch(0, *ranges[0])
    for index, (r0, r1) in enumerate(ranges):
        compute_stream.wait_event(ready)
        sl.record_stream(compute_stream)
        process_slice(r0, r1, sl)
        done = torch.cuda.Event()
        done.record(compute_stream)
        inflight_compute.append(done)
        if len(inflight_compute) > 1:
            inflight_compute.popleft().synchronize()
        if index + 1 < len(ranges):
            sl, ready = prefetch(index + 1, *ranges[index + 1])
    _flush_all_pending()
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
    leaf ANYWHERE in the filter — torch has no string tensor type, so text
    tokenization/matching stays CPU-only regardless of what else is in the
    filter (see `filters._token_row_masks`'s single-pass tokenization
    instead)."""
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
    """`(rows,)` SAFE OVER-APPROXIMATION of "does at least one query want
    this row" for a GPU-eligible per-query filter (Front B — see
    `run_compute`'s `has_baseline`/`vt_union_filters`), or `None` if `f.must`
    has no leaf at all to build one from (caller substitutes all-`True`: no
    tightening possible, but always correct).

    Built ONLY from `f.must`'s leaves — deliberately NEVER `should` or
    `must_not`: a `must` leaf is a NECESSARY condition for the row
    regardless of anything else in the filter, so whichever query actually
    wants a row necessarily satisfies THAT query's own version of every one
    of these leaves. That shared witness is what makes ANDing the per-leaf
    masks together valid: for any row some query `q` wants, `q` itself
    admits the row via EVERY per-query `must` leaf, so the row is in every
    leaf's "some query's own value admits this row" mask, hence in their
    intersection — provably a superset of the true "some query wants this
    row" set, however many `must` conditions there are (and strictly
    tighter than ORing the same masks, whenever there's more than one). A
    STATIC `must` leaf is a necessary condition too, and its exact
    `(rows,)` mask (already sitting in `leaf_arrays`, built by
    `_corpus_leaf_array` for `_gpu_cond_mask`) needs no over-approximating
    at all — it's ANDed in the same way, so a mixed filter like "my own
    tenant AND status=active" compacts down to only active rows instead of
    every row any tenant leaf admits. `should`/`must_not` don't have the
    necessary-condition property: a `should` branch's satisfaction doesn't
    require any ONE specific condition to hold (a row could be admitted
    via a totally different, possibly static, branch), and a `must_not`
    leaf's own predicate holding is what EXCLUDES a row, not what admits
    one — using either here could silently exclude a row some query
    genuinely wants, so a filter with no `must` leaves at all gets no
    tightening here (falls back to all-`True` via the `None` return),
    never a WRONG one.

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
    static = None
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
            # Static match/range leaf — leaf_arrays[cond] IS its exact
            # (rows,) boolean mask (see _corpus_leaf_array), a necessary
            # condition for every query, so it tightens the union exactly.
            static = leaf_arrays[cond] if static is None else (static & leaf_arrays[cond])
            continue
        mask = part if mask is None else (mask & part)
    if static is not None:
        mask = static if mask is None else (mask & static)
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
    for cond in _static_first(f.must):
        keep = keep & _gpu_cond_mask(cond, leaf_gpu, rows, query_gpu)
    if f.should:
        any_match = torch.zeros(n, dtype=torch.bool, device=device)
        for cond in _static_first(f.should):
            any_match = any_match | _gpu_cond_mask(cond, leaf_gpu, rows, query_gpu)
        keep = keep & any_match
    for cond in _static_first(f.must_not):
        keep = keep & ~_gpu_cond_mask(cond, leaf_gpu, rows, query_gpu)
    return keep


def _process_shared_batch(
    batch, member_idxs: list[int], specs: list[SearchSpec], spec_Q, spec_q_norms,
    spec_top_key, spec_top_enc,
    keeps: dict[Filter | None, np.ndarray | None], filter_is_per_query: dict[Filter | None, bool],
    filter_is_gpu_eligible: dict[Filter | None, bool], leaf_gpu: dict[FilterCondition, object],
    query_gpu: dict[FilterCondition, object], filter_share_count: dict[Filter | None, int],
    batch_size: int | None, gidx: int, device: str, orig_rows: np.ndarray | None,
    spec_qsel: list, spec_qrows: list, filter_n_q: dict[Filter | None, int],
    encoded_row_ids: np.ndarray | None = None,
    ordinal_base: int = 0,
    ordinal_row_ids: np.ndarray | None = None,
    multivector_token_budget: int | None = None,
    multivector_double_buffer: bool = False,
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

    `encoded_row_ids` is passed straight through to `_process_batch_group`
    (see there) — `None` for a single file (the common case), or a
    pre-encoded array when `batch` coalesces several files' rows into one
    (see `run_compute`'s `_flush_coalesce_group`).

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
                    # Height is THIS FILTER's query axis (`filter_n_q[s.filter]`
                    # — the union of its own specs' `rows` selectors)
                    cell_np = _unpack_query_axis(packed_np, filter_n_q[s.filter])
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
        batch, member_idxs, specs, spec_Q, spec_q_norms, spec_top_key, spec_top_enc,
        batch_size, gidx, device, orig_rows=orig_rows, select=select,
        encoded_row_ids=encoded_row_ids,
        ordinal_base=ordinal_base, ordinal_row_ids=ordinal_row_ids,
        multivector_token_budget=multivector_token_budget,
        multivector_double_buffer=multivector_double_buffer,
        spec_qsel=spec_qsel, spec_qrows=spec_qrows,
    )


def _load_query_columns(
    store: Store, qcfg, cols: list[str],
) -> dict[str, np.ndarray]:
    """Read a few small query columns, nothing else.

    Used for `SearchSpec.rows` selectors, which have to be resolved before the
    vector loaders run (they decide the query matrix height). Walks the same
    `store.list_parquets()` order the loaders do, so row i here is row i there.
    """
    out: dict[str, list] = {c: [] for c in cols}
    date_fmts = normalize_date_fields(qcfg.date_fields)
    for f in store.list_parquets():
        table = store.read_columns(f.read_path, list(cols))
        conv = convert_table_date_columns(table, date_fmts)
        for c in cols:
            out[c] += conv[c].to_pylist()
    return {c: _to_query_array(v) for c, v in out.items()}


def _resolve_spec_rows(
    specs: list, query_filter_vals: dict[str, np.ndarray], n_q: int,
) -> tuple[list, dict[str, np.ndarray | None]]:
    """Turn each spec's `rows` selector into sorted file-row indices.

    Returns `(spec_rows, vt_rows)`:
      * `spec_rows[m]` — that spec's own rows, or `None` for "every row".
      * `vt_rows[vt]`  — the UNION of that vector_type's specs' rows, or `None`
        if any of them takes every row. This is what the vector matrix gets
        built over: specs of one vector_type share a single query matrix (and
        a single score matrix per metric), so the matrix has to cover
        everything any of them asks for, and each spec indexes its own slice
        out of it afterwards.

    A selector matching nothing is an error, not an empty result: it means the
    column or the values are misspelled, and silently writing an empty output
    would look like a successful run.
    """
    spec_rows: list[np.ndarray | None] = []
    for s in specs:
        if s.rows is None:
            spec_rows.append(None)
            continue
        vals = query_filter_vals[s.rows.column]
        # Compare as strings so a selector works against an int/categorical
        # source column without the config having to mirror its dtype.
        mask = np.isin(vals.astype(str), np.asarray(s.rows.isin, dtype=str))
        idx = np.nonzero(mask)[0].astype(np.int64)
        if idx.size == 0:
            raise ValueError(
                f"search {s.name!r}: rows.column={s.rows.column!r} has no row matching "
                f"isin={s.rows.isin!r} — every query would be excluded"
            )
        spec_rows.append(idx)

    return spec_rows, _union_rows_by_key([s.vector_type for s in specs], spec_rows, n_q)


def _union_rows_by_key(keys: list, spec_rows: list, n_q: int) -> dict:
    """`key -> union of the `spec_rows` of every spec carrying that key`, or
    `None` when any of them takes every row (and likewise when the union turns
    out to cover the whole file, so the caller can skip indexing entirely).

    Two callers, same shape of question — "how tall does the thing these specs
    SHARE have to be?":

      * keyed by `vector_type`  -> the shared query/score matrix's height
        (see `_resolve_spec_rows`)
      * keyed by `SearchSpec.filter` -> the shared per-query filter mask's
        height (see `run_compute`'s `filter_rows`)

    `keys` must be hashable; `Filter` is frozen for exactly this reason, and
    `None` (the unfiltered entry) is a legitimate key.
    """
    out: dict = {}
    for key, rows in zip(keys, spec_rows):
        if key not in out:
            out[key] = rows
        elif out[key] is None or rows is None:
            out[key] = None
        else:
            out[key] = np.union1d(out[key], rows)
    for key, rows in out.items():
        if rows is not None and len(rows) == n_q:
            out[key] = None  # covers everything; skip the indexing entirely
    return out


def _local_positions(
    rows: np.ndarray | None, vt_rows: np.ndarray | None,
) -> np.ndarray | None:
    """Map a spec's file-row indices to positions in its vector_type's matrix.

    `None` means "the identity" — the spec reads every row of that matrix, so
    the scoring path can skip the gather entirely.
    """
    if rows is None:
        return None
    if vt_rows is None:
        return rows  # matrix is full height: file row == matrix row
    return np.searchsorted(vt_rows, rows).astype(np.int64)


def _row_selector(idx: np.ndarray | None, device: str):
    """One `SearchSpec.rows` index array -> the object the scoring path indexes
    the query axis with: `None` (all rows — skip the indexing entirely), a
    `slice` when `idx` is one CONTIGUOUS ascending run (indexing is then a
    view, no gather-copy), or an on-device int64 index tensor otherwise.

    Module-level, not a closure inside `run_compute`, so the contiguity guard
    is directly testable: `slice(idx[0], idx[-1] + 1)` is only equivalent to
    `idx` when `idx` has no gaps, and a run whose subsets are contiguous —
    which is what you get from a queries file sorted by query set, i.e. most
    fixtures — cannot tell the two apart.
    """
    if idx is None:
        return None
    if len(idx) and idx[-1] - idx[0] == len(idx) - 1:
        return slice(int(idx[0]), int(idx[-1]) + 1)
    import torch

    return torch.from_numpy(np.ascontiguousarray(idx)).to(device)


def _validate_query_filter_cols(
    qstore: Store, filter_cols: list[str], subset_cols: list[str] = (),
) -> None:
    """Fail fast with a clear message if a per-query filter condition
    (`match_from_query`/`range_from_query`/`match_text_from_query`) or a
    `SearchSpec.rows` selector (`rows.column`) names a queries column that
    doesn't exist — rather than letting a typo surface deep inside
    `load_queries`/`load_queries_sparse` as a generic pyarrow `ArrowInvalid`
    about a missing `FieldRef`. Peeks at the FIRST queries file's schema only
    (metadata read, no row data) since every queries file in a run is assumed
    to share one schema.

    `subset_cols` is the `rows.column` subset of `filter_cols` (they travel
    together from here on — see `run_compute`), passed separately ONLY so the
    error names the config key the reader actually has to go fix: pointing a
    misspelled `rows: {column: …}` at `match_from_query` sends them looking in
    the wrong place."""
    if not filter_cols:
        return
    files = qstore.list_parquets()
    if not files:
        return  # let the empty-queries-store case surface downstream as usual
    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(files[0].read_path, filesystem=qstore.fs).schema_arrow.names)
    missing = sorted(c for c in filter_cols if c not in schema_names)
    if missing:
        subset = set(subset_cols)
        missing_rows = sorted(c for c in missing if c in subset)
        missing_filter = sorted(c for c in missing if c not in subset)
        parts = []
        if missing_filter:
            parts.append(
                f"{missing_filter} referenced by a per-query filter "
                "(match_from_query/range_from_query/match_text_from_query)"
            )
        if missing_rows:
            parts.append(f"{missing_rows} referenced by a `rows` selector (rows.column)")
        raise ValueError(
            f"queries file is missing column(s): {' and '.join(parts)} "
            f"— available columns: {sorted(schema_names)}"
        )


def _select_device(torch) -> str:
    """Which torch device this run scores on: CUDA when present, else CPU.

    `NOVA_BF_DEVICE` overrides the choice (`cpu` or `cuda`). It exists for the
    parity harness (tests/parity), which needs to pin a GPU box to `cpu` to
    check that both devices produce the same ground truth from the same input.
    """
    want = os.environ.get("NOVA_BF_DEVICE", "").strip().lower()
    if not want:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want not in ("cpu", "cuda"):
        raise ValueError(
            f"NOVA_BF_DEVICE={want!r} is not a device nova-bf scores on "
            "— use 'cpu' or 'cuda' (or unset it to auto-select)"
        )
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NOVA_BF_DEVICE='cuda' but torch reports no CUDA device")
    return want


def run_compute(
    cfg: BruteForceConfig,
    num_jobs: int | None = None,
    job_rank: int | None = None,
    io_workers: int | None = None,
    io_thread_count: int | None = None,
    cpu_thread_count: int | None = None,
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

    # Whole-invocation timing for the manifest. The scan's own `wall0` (set much
    # further down) covers only the corpus pass, so it misses query loading and
    # result decoding.
    started_at = datetime.now(timezone.utc)
    run_t0 = time.perf_counter()
    job_rank = _resolve_rank(num_jobs, job_rank)
    specs = cfg.searches
    vts_needed = sorted({s.vector_type for s in specs})  # ["dense"] / ["sparse"] / both
    device = _select_device(torch)
    if device == "cpu":
        logger.warning("No GPU detected — brute force on CPU will be slow.")
    if "multivector" in vts_needed and cfg.params.multivector_kernel != "torch":
        if device != "cuda":
            if cfg.params.multivector_kernel == "triton_reduce":
                raise RuntimeError(
                    f"params.multivector_kernel={cfg.params.multivector_kernel!r} "
                    "requires a CUDA GPU"
                )
            logger.info(
                "params.multivector_kernel='auto': CUDA unavailable; using torch"
            )
        else:
            try:
                import nova_bf.multivector_kernels  # noqa: F401
            except ImportError as exc:
                if cfg.params.multivector_kernel == "triton_reduce":
                    raise RuntimeError(
                        f"params.multivector_kernel={cfg.params.multivector_kernel!r} "
                        "requires Triton; "
                        "install a CUDA PyTorch distribution that includes it"
                    ) from exc
                logger.warning(
                    "params.multivector_kernel='auto': Triton unavailable; using torch"
                )
            else:
                logger.info(
                    "multivector scoring uses cuBLAS FP32 matmul plus the "
                    "fused Triton ragged reducer"
                )
    # Opt-in TF32 tensor-core matmuls (see ParamsConfig.allow_tf32). CUDA-only;
    # torch's flag is a no-op on CPU, but gate on device anyway so the log is
    # honest. OFF by default keeps GT bit-exact f32 (matching Qdrant).
    if cfg.params.allow_tf32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.warning(
            "params.allow_tf32=True: TF32 matmuls enabled (CUDA) — ~1.75x faster "
            "multivector/dense matmul, but scores are NOT bit-exact f32 (~3e-4 "
            "relative error). Ensure this can't perturb your recall numbers."
        )

    # 1. queries — loaded once per DISTINCT vector_type needed across all specs
    #    (queries files are small, so no need to dedupe further than that).
    qstore = Store(cfg.queries.path, ranged_get=cfg.params.io_ranged_get)
    # Union of every spec's per-query filter columns — same "read exactly (and
    # only) what's referenced" guarantee corpus-side filter_cols already makes,
    # extended to the query side (see Filter.query_fields()).
    query_filter_cols = sorted({c for s in specs if s.filter for c in s.filter.query_fields()})
    # `SearchSpec.rows` columns ride along with the per-query filter columns:
    # both are read from the queries file into `query_filter_vals`, and both
    # get the same "referenced columns must exist" validation below.
    subset_cols = sorted({s.rows.column for s in specs if s.rows is not None})
    query_filter_cols = sorted(set(query_filter_cols) | set(subset_cols))
    _validate_query_filter_cols(qstore, query_filter_cols, subset_cols)
    # Row subsets must be known BEFORE the loaders run, because they decide how
    # tall each vector_type's query matrix is. The selector columns are a few
    # small columns, so reading them first is cheap next to the vector columns.
    spec_rows: list = [None] * len(specs)
    vt_rows: dict[str, np.ndarray | None] = {}
    n_q_pre: int | None = None  # None = no selectors, so nothing to cross-check
    if subset_cols:
        pre = _load_query_columns(qstore, cfg.queries, subset_cols)
        n_q_pre = len(next(iter(pre.values())))
        spec_rows, vt_rows = _resolve_spec_rows(specs, pre, n_q_pre)
    Q_np_by_vt: dict[str, np.ndarray] = {}
    query_vocab = None  # sparse only: sorted distinct query token ids (see _build_query_vocab)
    mv_q_offsets = None  # multivector only: (n_q+1,) query-token offsets (see load_queries_multivector)
    # Sparse gate + query-representation state (see _zero_gate_file_ok/_SparseQueryCache):
    # query-side flags fixed run-wide; the indicator holder is shared by every
    # file's batch and builds its CSR lazily, only if a signed file appears.
    sparse_q_nonneg, sparse_q_min_pos = False, float("inf")
    sparse_q_cache = _SparseQueryCache()
    query_ids: list[str] | None = None
    payload: dict[str, list] | None = None
    query_filter_vals: dict[str, np.ndarray] | None = None
    for vt in vts_needed:
        if vt == "sparse":
            Q_np, query_vocab, q_ids, q_payload, q_filter_vals = load_queries_sparse(
                qstore, cfg.queries, query_filter_cols, rows=vt_rows.get(vt)
            )
            if len(query_vocab) == 0 and len(q_ids) > 0:
                logger.warning(
                    "sparse query vocabulary is empty (every query has zero nonzero entries) — "
                    "every corpus row will score 0; check queries.sparse_column is correct."
                )
            # Query side of the zero-score no-overlap gate (see
            # `_zero_gate_file_ok`): computed on Q as SCORED (post-densify,
            # duplicates summed), once per run.
            sparse_q_nonneg = bool((Q_np >= 0).all())
            q_pos = Q_np[Q_np > 0]
            sparse_q_min_pos = float(q_pos.min()) if q_pos.size else float("inf")
            del q_pos
        elif vt == "multivector":
            # Q_np is the flat (total_query_tokens, D) token matrix — its
            # `.shape[1]` is the token dim, keeping the dim log / consistency
            # check below identical to dense. The per-query token offsets ride
            # alongside in `mv_q_offsets`.
            Q_np, mv_q_offsets, q_ids, q_payload, q_filter_vals = load_queries_multivector(
                qstore, cfg.queries, query_filter_cols
            )
            if Q_np.shape[0] == 0 and len(q_ids) > 0:
                logger.warning(
                    "multivector queries have zero tokens total (every query decoded empty) — "
                    "every corpus row will score 0; check queries.multivector_column is correct."
                )
        else:
            Q_np, q_ids, q_payload, q_filter_vals = load_queries(
                qstore, cfg.queries, query_filter_cols, rows=vt_rows.get(vt)
            )
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
                f"queries.{vt}_column produced "
                f"query ids that don't match a different vector_type's load for the same "
                f"query set (first mismatch at row "
                f"{next((i for i, (a, b) in enumerate(zip(q_ids, query_ids)) if a != b), min(len(q_ids), len(query_ids)))}"
                f"; {len(q_ids)} vs {len(query_ids)} total rows) — the two columns must "
                "agree on row count and order"
            )
    n_q = len(query_ids)
    if n_q_pre is not None and n_q_pre != n_q:
        # `spec_rows` indexes `Q`, `query_ids` and `payload` by FILE row, but it
        # was resolved from a SEPARATE pass over the queries store
        # (`_load_query_columns`) made before the vector loaders ran. The two
        # walk the same `store.list_parquets()` order, so they agree — but if
        # they ever stop agreeing, every subset silently addresses the wrong
        # queries (or IndexErrors deep inside a loader). Same reason the
        # cross-vector_type check above compares id IDENTITY, not just length.
        raise RuntimeError(
            f"queries store returned {n_q_pre} rows when reading the "
            f"`rows` selector column(s) but {n_q} rows when loading vectors — "
            "the query row set changed between the two reads, so a row subset "
            "cannot be trusted to name the right queries"
        )
    for i_spec, s in enumerate(specs):
        dim = Q_np_by_vt[s.vector_type].shape[1]
        # Per-spec query count, NOT the file's: with `rows` those differ, and
        # this is the line someone scans to confirm a run covers what they meant.
        s_n_q = n_q if spec_rows[i_spec] is None else len(spec_rows[i_spec])
        logger.info(
            "search=%r queries=%d%s %s=%d metric=%s k=%d device=%s%s",
            s.name, s_n_q, "" if s_n_q == n_q else f" of {n_q}",
            "vocab" if s.vector_type == "sparse" else "dim", dim,
            s.metric, s.k, device,
            f" rank={job_rank}/{num_jobs}" if num_jobs else "",
        )
        if s.filter is not None:
            logger.info("search=%r filter: %s", s.name, s.filter.model_dump(exclude_defaults=True))

    # 2. corpus files (global, deterministic order); this worker takes a stride
    #    slice so its global indices stay stable for id decoding. Shared across
    #    every spec — they must all see the identical file set/order/truncation.
    cstore = Store(cfg.corpus.path, ranged_get=cfg.params.io_ranged_get)
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

    # Multivector tile sizes: resolve the two knobs once. When
    # `multivector_token_budget` is set it fills in whichever of
    # `multivector_batch_size`/`multivector_query_block` was left None (an
    # explicit knob always wins), deriving from the mean tokens-per-query (from
    # the loaded queries) and mean tokens-per-doc (sampled cheaply from this
    # worker's first corpus file). See `_resolve_multivector_tiles`.
    mv_batch_size = cfg.params.multivector_batch_size
    mv_query_block = cfg.params.multivector_query_block
    if "multivector" in vts_needed and cfg.params.multivector_token_budget is not None:
        if mv_batch_size is None or mv_query_block is None:
            mean_q_tok = float(mv_q_offsets[-1]) / max(1, n_q) if n_q else 1.0
            mean_doc_tok = _sample_mean_doc_tokens(
                cstore, mine, cfg.corpus.multivector_column
            )
            mv_batch_size, mv_query_block = _resolve_multivector_tiles(
                cfg.params.multivector_token_budget,
                mean_q_tok,
                mean_doc_tok,
                mv_batch_size,
                mv_query_block,
            )
            logger.info(
                "multivector token_budget=%d derived batch_size=%s query_block=%s "
                "(mean tokens/query=%.1f, tokens/doc=%.1f); actual ragged offsets "
                "enforce the budget at scoring time",
                cfg.params.multivector_token_budget,
                mv_batch_size,
                mv_query_block,
                mean_q_tok,
                mean_doc_tok,
            )
        else:
            logger.info(
                "multivector token_budget=%d enforces an actual ragged token-product "
                "cap within batch_size=%d query_block=%d",
                cfg.params.multivector_token_budget,
                mv_batch_size,
                mv_query_block,
            )

    # 3. per-spec GPU state: each spec gets its own running (top_scores, top_enc)
    #    pair, but every spec of the same vector_type shares ONE raw query
    #    matrix regardless of metric. A cosine spec divides its score matrix
    #    by each query's norm inside `score()` (a per-query scalar, `q_norms`
    #    below) — holding a pre-normalized second copy of Q instead would
    #    double the biggest resident tensor on the GPU (n_q × vocab floats:
    #    7.2 GiB at 100k queries × 18k vocab, which is what OOM'd the
    #    dot+cosine fineweb run on a 24 GB A10G).
    Q_gpu_by_vt: dict[str, object] = {}
    for vt in vts_needed:
        flat = torch.tensor(Q_np_by_vt[vt], dtype=torch.float32, device=device)
        if vt == "multivector":
            # The token matrix + offsets (device for reductions, host for the
            # block-loop bounds) + tile size, wrapped so a spec's `Q` carries
            # everything `MultiVectorBatchSlice.score` needs.
            off_cpu = np.ascontiguousarray(mv_q_offsets, dtype=np.int64)
            Q_gpu_by_vt[vt] = MultiVectorQuery(
                flat, torch.tensor(off_cpu, dtype=torch.int64, device=device),
                off_cpu, n_q, mv_query_block, cfg.params.multivector_kernel,
            )
        else:
            Q_gpu_by_vt[vt] = flat
    # How many DISTINCT metrics each vector_type carries — the same quantity
    # `_process_batch_group` gates the shared-Gram path on (its `member_idxs`
    # is exactly this vt's spec list), computed here so a euclidean spec that
    # can never share is not charged for query norms it will not read.
    vt_distinct_metrics = {
        vt: len({s.metric for s in specs if s.vector_type == vt}) for vt in vts_needed
    }
    q_norms_by_vt: dict[str, object] = {}
    spec_Q, spec_q_norms, spec_top_key, spec_top_enc = [], [], [], []
    for i_spec, s in enumerate(specs):
        # Multivector cosine normalizes each TOKEN inside score() (not a
        # per-query scalar divide like dense/sparse), so it needs no `q_norms`
        # — and Q here is a MultiVectorQuery, not a tensor with .norm().
        #
        # `euclidean` (dense-only — rejected for sparse/multivector at config
        # load) takes `q_norms` ONLY when the shared-Gram derivation can
        # actually engage for its vector_type, i.e. some sibling spec uses a
        # different metric: `_scores`'s euclidean branch ignores the argument,
        # so building it for a euclidean-only run would be one O(n_q × dim)
        # reduction and an (n_q,) tensor that nothing ever reads.
        needs_q_norms = s.metric == "cosine" or (
            s.metric == "euclidean" and vt_distinct_metrics[s.vector_type] > 1
        )
        if needs_q_norms and s.vector_type != "multivector":
            if s.vector_type not in q_norms_by_vt:
                # clamp matches F.normalize's eps — a zero query scores 0
                # everywhere instead of NaN (and its no-overlap gate already
                # -infs every cell anyway).
                q_norms_by_vt[s.vector_type] = (
                    Q_gpu_by_vt[s.vector_type].norm(dim=1).clamp_min(1e-12)
                )
            spec_q_norms.append(q_norms_by_vt[s.vector_type])
        else:
            spec_q_norms.append(None)
        spec_Q.append(Q_gpu_by_vt[s.vector_type])
        # Height is this spec's OWN query count, not the file's: with several
        # specs over a unioned queries file that is the difference between
        # sum(len(subset)) and n_specs * n_q rows of running top-K state.
        h = n_q if spec_rows[i_spec] is None else len(spec_rows[i_spec])
        # The state holds packed keys, so the empty slot is the key for
        # `(-inf, worst ordinal)` — every real candidate outranks it, and the
        # decode gate recovers `-inf` from it exactly.
        spec_top_key.append(sentinel_key((h, s.k), device))
        spec_top_enc.append(torch.zeros((h, s.k), dtype=torch.int64, device=device))

    # Device-side row selectors for `SearchSpec.rows`, built once per run:
    #   spec_qsel[m]  — indexes this spec's rows in its vector_type's SCORE
    #                   matrix (which spans that type's row union)
    #   spec_qrows[m] — indexes the same rows in a FULL-query-axis per-query
    #                   filter mask (see `_pack_query_axis`)
    # A contiguous run becomes a `slice` so the score matrix is sliced as a
    # view instead of gathered; `None` means "all rows", the historical path,
    # which skips the indexing entirely. See `_row_selector`.
    spec_qsel = [
        _row_selector(_local_positions(spec_rows[m], vt_rows.get(specs[m].vector_type)), device)
        for m in range(len(specs))
    ]
    # spec_qrows is built further down, once `filter_rows` is known: its base is
    # the per-FILTER mask height, not the file's. See there.
    if any(r is not None for r in spec_rows):
        logger.info(
            "query-row subsets active: %s (file has %d queries; per-spec top-K "
            "state covers %d rows total instead of %d)",
            ", ".join(
                f"{s.name}={len(spec_rows[m])}"
                for m, s in enumerate(specs) if spec_rows[m] is not None
            ),
            n_q,
            sum(n_q if r is None else len(r) for r in spec_rows),
            n_q * len(specs),
        )

    # Per vector_type: does any spec have no filter (or an explicit-but-empty
    # one — see `_is_unfiltered`) OR a per-query filter (see `_is_per_query` —
    # it has no single row-subset to offer, since different queries need
    # different rows)? If so, every spec of that vector_type shares the whole
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

    # Narrow CPU per-query filter masks to the query rows that actually use them.
    # This keeps unrelated query sets in a unified file from increasing mask memory.
    filter_rows: dict[Filter | None, np.ndarray | None] = _union_rows_by_key(
        spec_filter, spec_rows, n_q
    )

    # GPU, uniform, and unfiltered paths do not use a narrowed per-file query mask.
    for f in list(filter_rows):
        if f is None or filter_is_gpu_eligible[f] or not filter_is_per_query[f]:
            filter_rows[f] = None

    # Query-axis height of each filter mask.
    filter_n_q: dict[Filter | None, int] = {
        f: (n_q if r is None else len(r)) for f, r in filter_rows.items()
    }

    # Slice per-query filter values to the same rows used by each mask.
    filter_query_vals: dict[Filter, dict[str, np.ndarray]] = {
        f: (
            query_filter_vals if r is None
            else {c: v[r] for c, v in query_filter_vals.items()}
        )
        for f, r in filter_rows.items() if f is not None
    }

    # Map each spec's query rows into its filter mask's local row numbering
    spec_qrows = [
        _row_selector(_local_positions(spec_rows[m], filter_rows[specs[m].filter]), device)
        for m in range(len(specs))
    ]
    if any(r is not None for r in filter_rows.values()):
        logger.info(
            "per-query filter mask height: %s (file has %d queries)",
            "; ".join(
                "+".join(s.name for s in specs if s.filter == f) + f"={filter_n_q[f]}"
                for f, r in filter_rows.items() if r is not None
            ),
            n_q,
        )

    vt_spec_idxs: dict[str, list[int]] = {vt: [] for vt in vts_needed}
    for i, s in enumerate(specs):
        vt_spec_idxs[s.vector_type].append(i)

    vt_configured_batch = {
        "dense": cfg.params.dense_batch_size,
        "sparse": cfg.params.sparse_batch_size,
        "multivector": mv_batch_size,  # already merged with the token-budget derivation
    }
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

    # Cross-file batch coalescing: when a vt's union filters are
    # ALL uniform (none per-query) and a real batch_size is configured,
    # small per-file post-compaction batches (selective filters -> few
    # surviving rows/file) get coalesced across several files into one
    # larger GPU call instead of paying per-file, per-member fixed overhead
    # on each tiny batch (see `_flush_coalesce_group` below). Excluded
    # whenever a per-query filter shares the vt: per-query masking
    # (`_gpu_evaluate`'s `leaf_gpu`, or the CPU-fallback `keeps[f]` packed
    # mask) is built PER FILE, sized to that one file's own rows —
    # coalescing would need those rebuilt/concatenated across files too,
    # which `_flush_coalesce_group` doesn't attempt (it only rebuilds
    # UNIFORM filters' plain `(rows,)` keeps entries). `vt_batch_size[vt]`
    # is `None` whenever the user hasn't configured a real batch size
    # (meaning no memory-bounding is wanted either), so coalescing is
    # skipped then too, rather than accumulating an unbounded number of
    # files before ever flushing.
    coalesce_eligible_vts: set[str] = {
        vt for vt in vt_union_filters
        if vt_batch_size[vt] is not None and not any(filter_is_per_query[f] for f in vt_union_filters[vt])
    }

    # Prefetch corpus files with a POOL of reader threads so many S3 GETs are in
    # flight at once — otherwise the GPU sits idle behind one file's latency at a
    # time. pyarrow releases the GIL during IO, so threads parallelize. Reader
    # threads may FINISH in any order, but the consumer below folds files into
    # the running top-K in a FIXED order (ascending `gidx`) regardless of
    # arrival order — see the comment above the consumer loop for why: the
    # merge is commutative in the SCORES it produces, but not in which of
    # several EXACTLY tied candidates a run picks.
    dense_col, sparse_col = cfg.corpus.dense_column, cfg.corpus.sparse_column
    multivector_col = cfg.corpus.multivector_column
    id_col = cfg.corpus.id_column  # None → derive make_point_id(file_key, row) at decode

    # --- tie-break ordinals (see `nova_bf.tiebreak`) --------------------------
    #
    # Every corpus row this worker reads gets a dense ordinal in
    # `[0, rows_this_worker)`, which rides in the low half of the packed
    # selection key and decides which of two EXACTLY-tied candidates wins.
    #
    #   ordinal  the row's position in CORPUS order — `rows_before` below is
    #            just a running counter, so this costs nothing.
    #   id       the row's position in SORTED-ID order, from one sort of this
    #            worker's whole id column here at startup.
    #
    # The `id` sort spans the worker's entire file list rather than each file,
    # because ordinals have to interleave across files to stay comparable when
    # the running top-K folds a later file against an earlier one.
    tiebreak = cfg.params.tiebreak
    rows_before = 0                      # ordinal of this worker's next row
    id_ordinals: dict[int, np.ndarray] | None = None
    # Whether `tiebreak='id'` orders NUMERICALLY. Read from the corpus SCHEMA,
    # not from any row, so every worker resolves the same rule — including one
    # that drew no files at all (`--num-jobs` above the file count).
    id_is_int = tie_unsigned = False
    if tiebreak == "id" and all_files:
        import pyarrow as _pa

        _t = cstore.read_schema(all_files[0].read_path).field(id_col).type
        id_is_int = _pa.types.is_integer(_t)
        tie_unsigned = _pa.types.is_unsigned_integer(_t)
        if not (
            id_is_int or _pa.types.is_string(_t) or _pa.types.is_large_string(_t)
        ):
            # Binary is deliberately NOT accepted. The ordinals would order it
            # by raw bytes, but `hit_ids` render it through Python's bytes
            # repr — whose order differs (`str(b'\\x41z') < str(b'\\x0az')`,
            # because 'A' < '\\') — and `merge` orders those reprs. The two
            # sides would disagree and the winner would move with --num-jobs.
            raise ValueError(
                f"params.tiebreak='id' needs an integer or string "
                f"corpus.id_column; {id_col!r} is {_t}. Use params.tiebreak='ordinal'."
            )
    if tiebreak == "id" and mine:
        t_ord = time.perf_counter()
        pool_n = max(1, min(io_workers or 8, 32))
        with ThreadPoolExecutor(max_workers=pool_n) as pool:
            # `map` preserves input order, which is what makes the ordinals
            # line up with `mine` — and `mine` is ascending `gidx`, so the
            # secondary "earliest corpus position wins" rule among duplicate
            # ids means what it says.
            id_arrays = list(pool.map(
                lambda f: cstore.read_columns(f.read_path, [id_col])[id_col],
                [f for _, f in mine],
            ))
        id_ordinals = dict(zip([g for g, _ in mine], build_ordinals(id_arrays)))
        n_ids = sum(len(a) for a in id_arrays)
        del id_arrays
        logger.info(
            "params.tiebreak='id': ranked %s ids from this worker's %d file(s) "
            "in %.1fs; ties go to the lowest %r, then to the earliest corpus row.",
            f"{n_ids:,}", len(mine), time.perf_counter() - t_ord, id_col,
        )
    # Union of every spec's filter fields — read_cols below stays exactly (and only)
    # the columns some spec actually references, same guarantee the single-search
    # path always made.
    filter_cols = sorted({c for s in specs if s.filter for c in s.filter.fields()})
    corpus_date_fmts = normalize_date_fields(cfg.corpus.date_fields)
    read_cols = list(dict.fromkeys(
        ([dense_col] if "dense" in vts_needed else [])
        + ([sparse_col] if "sparse" in vts_needed else [])
        + ([multivector_col] if "multivector" in vts_needed else [])
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
    # The OTHER pyarrow pool: parquet DECODE parallelism
    cpu_n = (cpu_thread_count if cpu_thread_count is not None
             else cfg.params.cpu_thread_count)
    if not cpu_n or cpu_n <= 0:
        cpu_n = os.cpu_count() or 1
    import pyarrow as pa
    if pa.cpu_count() != cpu_n:
        logger.info(
            "pyarrow CPU thread pool %d -> %d (parquet decode parallelism)",
            pa.cpu_count(), cpu_n,
        )
    pa.set_cpu_count(cpu_n)
    work: Queue = Queue()
    for item in mine:
        work.put(item)
    fq: Queue = Queue(maxsize=io_workers * 2)
    # `fq`'s bound alone is NOT the memory ceiling it looks like: the consumer
    # must fold files in ascending `gidx` order, so while it waits for a slow
    # file inside `_next_in_order` it keeps draining `fq` into the unbounded
    # `pending` dict below — every drain frees a queue slot, readers never
    # block, and one pathologically slow early file (a hung S3 read) would let
    # the ENTIRE remaining corpus accumulate decoded in host RAM. `window` is
    # the real end-to-end bound: a permit is held from the moment a reader
    # STARTS a file until the consumer CONSUMES it, so at most `io_workers*2`
    # files ever exist anywhere in the pipeline (being read + in `fq` + in
    # `pending`). Deadlock-free: readers start files in ascending `gidx`
    # order (`work` is FIFO), so the oldest unconsumed file — exactly the one
    # the consumer is waiting for — is always inside the window, and consuming
    # it releases the permit that slides the window forward. In a healthy run
    # the consumer keeps up and no reader ever blocks here; the permit only
    # binds in the stall scenario it exists for.
    window = Semaphore(io_workers * 2)

    def reader():
        while True:
            window.acquire()
            try:
                gidx, f = work.get_nowait()
            except Empty:
                window.release()
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
                # Declared datetime corpus columns -> int64 epoch µs before any
                # filter (static `range` or GPU-native `range_from_query`) reads
                # them, so every range path stays numeric and unchanged.
                table = convert_table_date_columns(table, corpus_date_fmts)
                # Decode each vector_type at most ONCE per file, regardless of how many
                # specs need it — wrapped in the batch abstraction (`DenseCorpusBatch`/
                # `SparseCorpusBatch`) below, where every spec of that vector_type shares
                # it (see `run_compute`'s `has_baseline`).
                arrs: dict[str, object] = {}
                if "dense" in vts_needed:
                    arrs["dense"] = dense_to_2d(table[dense_col])
                if "multivector" in vts_needed:
                    arrs["multivector"] = multivector_to_ragged(table[multivector_col])
                if "sparse" in vts_needed:
                    sp_offsets, sp_idx, sp_val = sparse_to_coo_parts(table[sparse_col])
                    # Norms and the zero-score gate BEFORE remap: both are
                    # defined over the raw, untruncated file values (see
                    # _sparse_file_norms / _zero_gate_file_ok docstrings).
                    sp_norms = _sparse_file_norms(sp_offsets, sp_idx, sp_val) if need_sparse_norms else None
                    sp_gate = _zero_gate_file_ok(sp_val, sparse_q_nonneg, sparse_q_min_pos)
                    sp_offsets, sp_idx, sp_val = _remap_sparse_file(
                        sp_offsets, sp_idx, sp_val, query_vocab
                    )
                    arrs["sparse"] = (sp_offsets, sp_idx, sp_val, sp_norms, sp_gate)
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

                # CPU-fallback filters (a match_text/match_text_from_query leaf
                # anywhere, so ineligible for Front A's GPU-native path — see
                # _gpu_eligible) each write only their OWN keeps[f] slot, with
                # no shared mutable state between them, so dispatch every
                # distinct one of THIS file's CPU-fallback filters concurrently
                # instead of one evaluate() call at a time: pyarrow's string
                # compute kernels release the GIL, so this is real
                # thread-level speedup, not GIL-serialized (measured ~2.5-3.5x
                # on real corpus text). Only worth the pool overhead when
                # there's more than one to dispatch. (`_token_row_masks` also
                # fans its row-batches out on its own inner pool — nested
                # thread pools are safe, just briefly oversubscribed.)
                cpu_fallback_filters = [
                    f for f in distinct_filters if f is not None and not filter_is_gpu_eligible[f]
                ]

                # `filter_query_vals[f]` is already narrowed to this filter's query rows,
                # so `evaluate` builds a mask with `filter_n_q[f]` rows.
                if len(cpu_fallback_filters) > 1:
                    with ThreadPoolExecutor(max_workers=len(cpu_fallback_filters)) as pool:
                        masks = list(pool.map(
                            lambda f: evaluate(f, table, filter_query_vals[f]), cpu_fallback_filters
                        ))
                else:
                    masks = [evaluate(f, table, filter_query_vals[f]) for f in cpu_fallback_filters]
                for f, mask in zip(cpu_fallback_filters, masks):
                    keeps[f] = _pack_query_axis(mask) if mask.ndim == 2 else mask

                # GPU-eligible filters (and the unfiltered `None` entry) stay
                # sequential: `leaf_arrays` is shared/deduped ACROSS filters
                # referencing the same FilterCondition (`if cond not in
                # leaf_arrays`), which isn't safe to parallelize without a
                # lock — and this branch has no text matching to speed up anyway.
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

                # Wrap into the vector_type-agnostic batch abstraction and,
                # per vt, compact to the union of every active filter's
                # surviving rows RIGHT HERE — moved off the single consumer
                # thread: this is a real CPU cost (a fancy-index array copy),
                # and io_workers reader threads can do it concurrently
                # instead of it all serializing behind the consumer's GPU
                # enqueue. `raw_stats` is the PRE-compaction (n_rows, nbytes)
                # per vt, since `run_compute`'s rows_seen/bytes_seen count the
                # whole file, not the compacted subset. `batch_orig_rows[vt]`
                # is `None` when `has_baseline[vt]` (no compaction — batch IS
                # the whole file), else the true-row array `.compact()`
                # returns, exactly as `_process_shared_batch` already expects.
                batches: dict[str, object] = {}
                raw_stats: dict[str, tuple[int, int]] = {}
                batch_orig_rows: dict[str, np.ndarray | None] = {}
                if "dense" in vts_needed:
                    b = DenseCorpusBatch(arrs["dense"])
                    raw_stats["dense"] = (b.n_rows, b.nbytes)
                    if has_baseline["dense"]:
                        batches["dense"], batch_orig_rows["dense"] = b, None
                    else:
                        batches["dense"], batch_orig_rows["dense"] = b.compact(
                            _union_keep(vt_union_filters["dense"], keeps)
                        )
                if "multivector" in vts_needed:
                    mv_offsets, mv_flat = arrs["multivector"]
                    b = MultiVectorCorpusBatch(mv_offsets, mv_flat)
                    raw_stats["multivector"] = (b.n_rows, b.nbytes)
                    if has_baseline["multivector"]:
                        batches["multivector"], batch_orig_rows["multivector"] = b, None
                    else:
                        batches["multivector"], batch_orig_rows["multivector"] = b.compact(
                            _union_keep(vt_union_filters["multivector"], keeps)
                        )
                if "sparse" in vts_needed:
                    sp_offsets, sp_idx, sp_val, sp_norms, sp_gate = arrs["sparse"]
                    b = SparseCorpusBatch(
                        sp_offsets, sp_idx, sp_val, sp_norms, query_vocab, need_sparse_norms,
                        sp_gate, sparse_q_cache,
                    )
                    raw_stats["sparse"] = (b.n_rows, b.nbytes)
                    if has_baseline["sparse"]:
                        batches["sparse"], batch_orig_rows["sparse"] = b, None
                    else:
                        batches["sparse"], batch_orig_rows["sparse"] = b.compact(
                            _union_keep(vt_union_filters["sparse"], keeps)
                        )
                # `t2 - t1` now covers filter evaluation AND compaction (moved
                # here together) — see the `filter_secs` logging below, whose
                # meaning widens accordingly.
                t2 = time.perf_counter()
                # `n_rows` is the file's own row count, carried explicitly:
                # the tie-break ordinal counter advances by it, and deriving it
                # from `raw_stats` instead would tie that counter to whichever
                # vector types this run happens to configure.
                fq.put((gidx, batches, batch_orig_rows, raw_stats, ids, keeps, leaf_arrays,
                        n_rows, t1 - t0, t2 - t1))
            except Exception as exc:
                # Permit deliberately NOT released: the consumer re-raises on
                # fetching this, killing the run — holding it just stops the
                # surviving readers from racing further ahead in the meantime.
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
    # arbitrary-arrival-order output back into that order — see its docstring
    # for why this matters for reproducibility, not just correctness.
    # `pending`'s size is bounded by the `window` semaphore above (at most
    # `io_workers * 2` files in flight end-to-end), NOT by `fq`'s maxsize:
    # the wait loop inside `_next_in_order` drains `fq` while blocked, so the
    # queue's own bound caps nothing on its own.
    pending: dict[int, tuple] = {}

    # Per-vt accumulation buffer for coalescing several files'
    # (already union-compacted) batches into one larger `_process_shared_
    # batch` call — see `coalesce_eligible_vts` above. Each buffered entry
    # is `(gidx, batch, orig_rows, keeps-restricted-to-this-vt's-own-
    # filters)` for one file; flushed once the accumulated row count
    # reaches `vt_batch_size[vt]`, or at the very end of the run for any
    # remainder. Memory cost: bounded by `vt_batch_size[vt]` rows' worth of
    # ALREADY-compacted (small) data plus each buffered file's own uniform
    # filters' keep-masks — not by file size or corpus size.
    coalesce_buf: dict[str, list[tuple]] = {vt: [] for vt in coalesce_eligible_vts}
    coalesce_rows: dict[str, int] = {vt: 0 for vt in coalesce_eligible_vts}

    def _flush_coalesce_group(vt: str) -> float:
        buf = coalesce_buf[vt]
        # A file contributing no rows — an empty shard, or every row dropped by
        # the union filter — must be dropped BEFORE concatenating.
        buf = [e for e in buf if e[1].n_rows]
        if not buf:
            coalesce_buf[vt] = []
            coalesce_rows[vt] = 0
            return 0.0
        concat = {
            "dense": _concat_dense_batches,
            "sparse": _concat_sparse_batches,
            "multivector": _concat_multivector_batches,
        }[vt]
        combined_batch = concat([entry[1] for entry in buf])
        encoded_ids = np.concatenate([
            file_gidx * MAX_ROWS_PER_FILE + orig_rows
            for file_gidx, _, orig_rows, _, _, _ in buf
        ])
        # Tie-break ordinals for the SAME rows in the SAME concatenated order.
        # A coalesced group mixes files, so no scalar base covers it even under
        # `tiebreak='ordinal'` — each file's own base is applied here.
        ordinal_ids = np.concatenate([
            (f_ord[orig_rows] if f_ord is not None else f_base + orig_rows)
            for _, _, orig_rows, _, f_base, f_ord in buf
        ]).astype(np.int64, copy=False)
        # Rebuild each of this vt's (uniform-only, by `coalesce_eligible_
        # vts`' own precondition) filters' keep-mask, restricted to
        # survivor rows and concatenated in the SAME order as
        # `combined_batch` — so `orig_rows=None` below (identity indexing)
        # correctly lines up `select()`'s `keeps[s.filter][true_rows]`
        # lookups with this GROUP's own row order, not any one file's
        # original per-file numbering.
        combined_keeps = {
            f: np.concatenate([
                file_keeps[f][orig_rows] for _, _, orig_rows, file_keeps, _, _ in buf
            ])
            for f in vt_union_filters[vt]
        }
        elapsed = _process_shared_batch(
            combined_batch, vt_spec_idxs[vt], specs, spec_Q, spec_q_norms,
            spec_top_key, spec_top_enc,
            combined_keeps, filter_is_per_query, filter_is_gpu_eligible, {}, gpu_query_gpu,
            filter_share_count, vt_batch_size[vt], 0, device, orig_rows=None,
            encoded_row_ids=encoded_ids, ordinal_row_ids=ordinal_ids,
            spec_qsel=spec_qsel, spec_qrows=spec_qrows, filter_n_q=filter_n_q,
            multivector_token_budget=(
                cfg.params.multivector_token_budget if vt == "multivector" else None
            ),
            multivector_double_buffer=(
                cfg.params.multivector_double_buffer if vt == "multivector" else False
            ),
        )
        coalesce_buf[vt] = []
        coalesce_rows[vt] = 0
        return elapsed

    with tqdm(total=len(mine), unit="file", dynamic_ncols=True, desc="bf") as bar:
        for want_gidx, _f in mine:
            w0 = time.perf_counter()
            gidx, batches, batch_orig_rows, raw_stats, ids, keeps, leaf_arrays, file_rows, rsec, fsec = _next_in_order(
                want_gidx, pending, _fetch_or_raise
            )
            window.release()  # file consumed — a reader may start another
            io_wait += time.perf_counter() - w0
            read_secs += rsec
            filter_secs += fsec
            bar.update(1)

            # `batches[vt]` is already wrapped AND, when `has_baseline[vt]` is
            # False, already compacted to the union of every active filter's
            # surviving rows — both done in the reader thread now (see
            # `reader()`), not here, so io_workers threads do that CPU work
            # concurrently instead of it serializing behind GPU enqueue on
            # this single consumer thread.

            # Front A: transfer this file's GPU-eligible per-query leaf arrays
            # to the GPU ONCE here (not once per batch slice) — mirrors
            # Q_gpu_by_vt's one-time transfer, vector_type-agnostic since a
            # filter condition reads a payload column, never a vector column.
            leaf_gpu: dict[FilterCondition, object] = {
                cond: torch.from_numpy(arr).to(device, non_blocking=True)
                for cond, arr in leaf_arrays.items()
            }

            # rows_seen/bytes_seen count the WHOLE file (pre-compaction) per
            # distinct vector_type present, not per spec — `raw_stats` carries
            # that pre-compaction (n_rows, nbytes) from the reader, since
            # `batches[vt]` itself may already be the compacted subset.
            #
            # corpus_ids is kept only for files where SOME spec could still
            # resolve a hit — i.e. it's unfiltered, or its filter keeps at least
            # one row in this file. Each entry of `keeps` is a mask over this
            # file's rows independent of vector_type (filters read payload
            # columns, not the vector columns), so checking it here is exact,
            # not an approximation: a restrictive spec's filter dropping the
            # whole file must never block a DIFFERENT spec's id resolution for
            # that same file, but a file every spec's filter drops needs no ids
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
            # This file's tie-break ordinals. The counter advances by the file's
            # PRE-compaction row count, so a row's ordinal is a property of the
            # corpus alone — advancing by survivors instead would make it depend
            # on which filters happened to run, and the specs sharing this file
            # do not share a filter.
            ordinal_base = rows_before
            rows_before += file_rows
            if rows_before > MAX_ROWS_PER_WORKER:
                raise RuntimeError(
                    f"this worker's corpus slice exceeds {MAX_ROWS_PER_WORKER:,} "
                    "rows, which overflows the 32-bit tie-break field and would "
                    "make ties non-deterministic again. Split the work further "
                    "with a larger `--num-jobs`."
                )
            # Popped, not read: the worker's ordinals are ~4 bytes/row and each
            # file is visited exactly once, so releasing them as they are
            # consumed keeps only the unread tail resident.
            file_ordinals = None if id_ordinals is None else id_ordinals.pop(gidx)

            for vt in vts_needed:
                raw_rows, raw_bytes = raw_stats[vt]
                rows_seen += raw_rows
                bytes_seen += raw_bytes

            for vt in vts_needed:
                if vt in coalesce_eligible_vts:
                    coalesce_buf[vt].append((
                        gidx, batches[vt], batch_orig_rows[vt],
                        {f: keeps[f] for f in vt_union_filters[vt]},
                        ordinal_base, file_ordinals,
                    ))
                    coalesce_rows[vt] += batches[vt].n_rows
                    if coalesce_rows[vt] >= vt_batch_size[vt]:
                        gpu_secs += _flush_coalesce_group(vt)
                else:
                    gpu_secs += _process_shared_batch(
                        batches[vt], vt_spec_idxs[vt], specs, spec_Q, spec_q_norms,
                        spec_top_key, spec_top_enc,
                        keeps, filter_is_per_query, filter_is_gpu_eligible, leaf_gpu, gpu_query_gpu,
                        filter_share_count, vt_batch_size[vt], gidx, device, orig_rows=batch_orig_rows[vt],
                        ordinal_base=ordinal_base,
                        ordinal_row_ids=(
                            None if file_ordinals is None
                            else (
                                file_ordinals if batch_orig_rows[vt] is None
                                else file_ordinals[batch_orig_rows[vt]]
                            )
                        ),
                        spec_qsel=spec_qsel, spec_qrows=spec_qrows, filter_n_q=filter_n_q,
                        multivector_token_budget=(
                            cfg.params.multivector_token_budget
                            if vt == "multivector"
                            else None
                        ),
                        multivector_double_buffer=(
                            cfg.params.multivector_double_buffer
                            if vt == "multivector"
                            else False
                        ),
                    )

            if bar.n % 200 == 0:
                postfix = f"io_wait={io_wait:.0f}s gpu={gpu_secs:.0f}s"
                if any_filter:
                    postfix += f" filter={filter_secs:.0f}s"
                bar.set_postfix_str(postfix, refresh=False)

    # Flush any remainder still buffered for coalescing — a coalesce-eligible
    # vt's LAST group may not have reached vt_batch_size[vt] on its own.
    for vt in coalesce_eligible_vts:
        gpu_secs += _flush_coalesce_group(vt)

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
    # `read_secs`/`filter_secs` are summed across the `io_workers` reader threads
    # that run concurrently, so divide by io_workers for a wall-clock-comparable
    # figure. Readers do read -> filter -> compact SERIALLY per file, so the
    # consumer's `io_wait` (starvation) is driven by BOTH — it is NOT "waiting on
    # IO" alone. Splitting the reader wall into its read vs filter halves is what
    # tells IO-bound from filter-bound.
    read_wall = read_secs / max(1, io_workers)
    filter_wall = filter_secs / max(1, io_workers)
    logger.info(
        "bf-bench io_workers=%d cpu_threads=%d files=%d rows=%d gb=%.3f wall_s=%.1f "
        "wall_mbps=%.1f stream_mbps=%.1f io_wait_s=%.1f gpu_s=%.1f filter_s=%.1f "
        "read_wall_s=%.1f filter_wall_s=%.1f",
        io_workers, cpu_n, len(mine), rows_seen, gb, wall,
        wall_mbps, stream_mbps, io_wait, gpu_secs, filter_secs,
        read_wall, filter_wall,
    )
    # Diagnose WHY the consumer starved (io_wait high), distinguishing the two
    # reader-side costs — raising io_workers only helps when reads, not filtering,
    # are the reader bottleneck. Comparing io_wait against gpu_secs alone (as this
    # once did) mislabels a filter-bound run "IO-bound" and wrongly advises more
    # readers.
    if io_wait > 3 * max(gpu_secs, 1e-6):
        if filter_wall > read_wall:
            logger.info(
                "FILTER-bound: readers spend more wall filtering (~%.0fs) than on IO "
                "(~%.0fs); io_wait=%.0fs is filter-driven, not slow IO (aggregate read "
                "%.0f MB/s). Raising params.io_workers (currently %d) won't help — reduce "
                "filter cost or add ranks.",
                filter_wall, read_wall, io_wait, wall_mbps, io_workers,
            )
        else:
            logger.info(
                "IO-bound: readers spend more wall on IO (~%.0fs) than filtering (~%.0fs) "
                "and the consumer idles %.0f%% waiting — raise params.io_workers "
                "(currently %d).",
                read_wall, filter_wall,
                100 * io_wait / max(io_wait + gpu_secs, 1e-9), io_workers,
            )

    # 4. decode each spec's final top-K into hit ids and write its own output.
    #    Either recompute make_point_id from (file_key, row) — only K*n_q ids,
    #    nothing corpus-wide — or, when an id column is configured, read it back
    #    from the in-RAM per-file arrays. Shared by every spec: depends only on
    #    id_col/corpus_ids/all_files, none of which is per-spec.
    
    # ID decoding is on the n_q*k hot path. Flatten per-file IDs once so encoded
    # (gidx, row) pairs resolve as one vectorized `take` instead of scalar lookups.
    # Built lazily and only when an ID column is configured.
    flat_ids: list = [None]  # boxed so the closure can memoize into it

    def _flat_ids():
        if flat_ids[0] is None:
            import pyarrow as pa

            gidxs = sorted(corpus_ids)
            arrays = [
                corpus_ids[g].combine_chunks()
                if isinstance(corpus_ids[g], pa.ChunkedArray) else corpus_ids[g]
                for g in gidxs
            ]
            lens = np.fromiter((len(a) for a in arrays), dtype=np.int64,
                               count=len(arrays))
            # `base` is indexed by gidx, so it must span the largest gidx seen,
            # not just len(arrays) — a filtered run can leave gaps.
            base = np.zeros((max(gidxs) + 1) if gidxs else 1, dtype=np.int64)
            base[np.asarray(gidxs, dtype=np.int64)] = np.concatenate(
                ([0], np.cumsum(lens)[:-1])
            ) if len(lens) else np.zeros(0, dtype=np.int64)
            flat_ids[0] = (pa.concat_arrays(arrays) if arrays else pa.array([]), base)
        return flat_ids[0]

    if id_col is not None:
        def resolve_id(e: int) -> str:
            gidx = e // MAX_ROWS_PER_FILE
            row = e % MAX_ROWS_PER_FILE
            return str(corpus_ids[gidx][row].as_py())

        def resolve_id_value(e: int):
            """The id's RAW value, before stringification — what `merge` needs
            to order a numeric id column, since `"10"` precedes `"9"`."""
            return corpus_ids[e // MAX_ROWS_PER_FILE][e % MAX_ROWS_PER_FILE].as_py()
    else:
        def resolve_id(e: int) -> str:
            return make_point_id(
                all_files[e // MAX_ROWS_PER_FILE].key, e % MAX_ROWS_PER_FILE
            )

    # Storage dtypes of the vectors that were actually scored, read from the
    # file footers (schema only, no column data). Best-effort: this is
    # provenance, so a store that cannot answer costs the metadata key, never
    # the run.
    def _dtypes_for(spec) -> dict[str, str]:
        column = (
            cfg.corpus.sparse_column
            if spec.vector_type == "sparse"
            else cfg.corpus.multivector_column
            if spec.vector_type == "multivector"
            else cfg.corpus.dense_column
        )
        qcolumn = (
            cfg.queries.sparse_column
            if spec.vector_type == "sparse"
            else cfg.queries.multivector_column
            if spec.vector_type == "multivector"
            else cfg.queries.dense_column
        )
        out: dict[str, str] = {}
        try:
            if all_files:
                out["corpus_dtype"] = vector_dtype(
                    cstore.read_schema(all_files[0].read_path), column or ""
                )
        except Exception as exc:  # noqa: BLE001 - provenance must never fail a run
            logger.debug("could not read corpus dtype for provenance: %s", exc)
        try:
            qfiles = qstore.list_parquets()
            if qfiles:
                out["queries_dtype"] = vector_dtype(
                    qstore.read_schema(qfiles[0].read_path), qcolumn or ""
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not read queries dtype for provenance: %s", exc)
        return out

    out = Store(cfg.output.path)
    results: dict[str, str] = {}
    # Per-search rows for the run manifest, filled as each output is written.
    manifest_searches: list[dict] = []
    # The corpus as an ordered file list, hashed once: it identifies which run
    # these partials belong to (see results.run_identity) AND goes in the
    # manifest. Computed over `all_files` — post include/exclude, PRE stride —
    # so every rank of one run derives the identical value without coordinating.
    corpus_fp = run_manifest.corpus_fingerprint(all_files)
    # A sharded run's partials must carry whatever `merge` needs to apply the
    # SAME rule across workers that each worker applied within itself. The
    # worker's ordinal cannot travel — it is a private relabelling, meaningless
    # against another worker's — so:
    #
    #   ordinal          the row's GLOBAL corpus position, which `enc` already
    #                    is (`gidx * MAX_ROWS_PER_FILE + row`, and `gidx`
    #                    indexes the same path-sorted list in every worker).
    #   id, string col   nothing: `merge` compares `hit_ids` directly.
    #   id, numeric col  the id's numeric order image, because `hit_ids` reach
    #                    `merge` already stringified.
    #
    # A string id column needs nothing at all, and a single-node run needs
    # nothing either, since there is no reduce.
    if tiebreak == "ordinal":
        # `enc` IS the ordinate — no per-hit work, just write the column.
        resolve_tie = None
        needs_ordinate = True
    elif id_is_int:
        def resolve_tie(e: int) -> int:
            return id_order_scalar(resolve_id_value(e), tie_unsigned)
        needs_ordinate = True
    else:
        resolve_tie = None
        needs_ordinate = False
    want_tie_column = needs_ordinate and num_jobs is not None

    for i, s in enumerate(specs):
       # Sort the final top-K by packed key so ties follow deterministic tie-break
        # order. Process query rows in chunks to bound peak GPU memory during sorting.
        h_i = spec_top_key[i].shape[0]
        chunk = max(1, min(h_i, DECODE_CHUNK_SLOTS // max(1, s.k)))
        enc_parts, sc_parts = [], []
        for r0 in range(0, h_i, chunk):
            kb, order = torch.sort(
                spec_top_key[i][r0 : r0 + chunk], dim=1, descending=True
            )
            enc_parts.append(spec_top_enc[i][r0 : r0 + chunk].gather(1, order).cpu().numpy())
            del order
            # Recover scores from the packed keys.
            sc_parts.append(unpack_score(kb).cpu().numpy())
            del kb
        enc = enc_parts[0] if len(enc_parts) == 1 else np.concatenate(enc_parts)
        sc = sc_parts[0] if len(sc_parts) == 1 else np.concatenate(sc_parts)
        del enc_parts, sc_parts
        # Release this search's GPU state before decoding results on the CPU.
        spec_top_key[i] = spec_top_enc[i] = None
        valid = sc > float("-inf")
        # A spec with a `rows` subset wrote state for its OWN queries only, so
        # its output covers those rows — ids and payload are sliced to match.
        # `query_ids`/`payload` stay full-length upstream (both loaders return
        # every row) precisely so this slice is the only place that has to know.
        rows_i = spec_rows[i]
        out_n = sc.shape[0]
        import pyarrow as pa

        counts = valid.sum(axis=1).astype(np.int64)
        offsets = pa.array(
            np.concatenate(([0], np.cumsum(counts))).astype(np.int32), type=pa.int32()
        )
        flat_enc = enc[valid]                       # row-major == list order
        hit_scores = pa.ListArray.from_arrays(
            offsets, pa.array(sc[valid], type=pa.float32())
        )

        raw_taken = None  # the id values for these hits, resolved ONCE
        if id_col is not None:
            values, base = _flat_ids()
            flat_idx = (base[flat_enc // MAX_ROWS_PER_FILE]
                        + (flat_enc % MAX_ROWS_PER_FILE))
            raw_taken = values.take(pa.array(flat_idx))
            hit_ids = pa.ListArray.from_arrays(
                offsets, raw_taken.cast(pa.string()).fill_null("None")
            )
        else:
            # make_point_id is an md5 per hit and cannot be vectorized into
            # Arrow
            keys = [f.key for f in all_files]
            hit_ids = pa.ListArray.from_arrays(offsets, pa.array(
                [make_point_id(keys[int(e) // MAX_ROWS_PER_FILE],
                               int(e) % MAX_ROWS_PER_FILE) for e in flat_enc],
                type=pa.string(),
            ))

        hit_tie = None
        if want_tie_column:
            if resolve_tie is None:
                # ordinal mode: the encoded value already IS the global corpus
                # position, so no resolution at all.
                tie_vals = pa.array(flat_enc.astype(np.int64), type=pa.int64())
            else:
                # `id` mode over a numeric column. Reuses `raw_taken` rather
                # than resolving every element a second time
                tie_vals = id_order_array(raw_taken, tie_unsigned)
            hit_tie = pa.ListArray.from_arrays(offsets, tie_vals)

        if rows_i is None:
            out_ids, out_payload = query_ids, payload
        else:
            out_ids = [query_ids[r] for r in rows_i]
            out_payload = {c: [v[r] for r in rows_i] for c, v in payload.items()}
        # Stamped on the PARTIALS too, not just the final file: a partial is a
        # parquet someone can pick up on its own, and a merge that mixed
        # partials from two different runs is exactly the mistake this makes
        # visible.
        dtypes = _dtypes_for(s)
        table = build_result_table(
            out_ids, out_payload, hit_ids, hit_scores,
            provenance(
                cfg, s, dtypes,
                corpus_sha=corpus_fp["sha256"],
                num_jobs=num_jobs,
                job_rank=job_rank,
                # A `--max-files` run read only part of its own slice, so its
                # output is not ground truth. Fingerprinting it separately is
                # what stops a benchmarking partial from ever merging with a
                # real one.
                partial_slice=max_files is not None,
            ),
            hit_tie=hit_tie,
        )
        short_i = int((counts < s.k).sum())
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
            warn_if_short(short_i, out_n, s.k, s.name, logger)
        path = out.write(name, table)
        logger.info("search=%r wrote %s (%d queries)", s.name, path, out_n)
        results[s.name] = path
        entry = run_manifest.search_entry(s)
        entry.update({
            "queries": out_n,
            "output_file": name,
            "output_path": path,
            "hit_tie_column": want_tie_column,
            # Storage dtypes of the vectors actually scored.
            "corpus_dtype": dtypes.get("corpus_dtype"),
            "queries_dtype": dtypes.get("queries_dtype"),
        })
        if num_jobs is None:
            # Only a whole-corpus run can say anything true about short top-Ks.
            # On a partial, "fewer than k hits" is the normal state of a stride
            # slice, so reporting it would read as a defect that isn't one.
            entry["queries_short_of_k"] = short_i
        manifest_searches.append(entry)

    # The run manifest — written LAST, so it only ever describes outputs that
    # actually landed, and best-effort, so it cannot fail a run that produced
    # them (see manifest.py).
    # See `counts.queries_searched`: any unsubsetted search covers every query,
    # otherwise it is the union of the subsets (searches may overlap, so this is
    # a set union, not a sum).
    queries_searched = (
        n_q if any(r is None for r in spec_rows)
        else len(set().union(*(set(r.tolist()) for r in spec_rows)))
    )
    doc = run_manifest.base_manifest(cfg, "compute", device=device)
    doc["compute"].update(run_manifest.gpu_peak(device))
    # Which corpus, as an ordered file list — the ids depend on that order.
    # Fingerprinted over `all_files` (post include/exclude, PRE stride), so
    # every rank of one run reports the identical hash and a mismatch means
    # the ranks disagreed about the corpus, not about their slice of it.
    doc["source"]["corpus"]["fingerprint"] = corpus_fp
    # `params` records what RAN, so the CLI overrides and the sizes resolved at
    # runtime replace the configured values (which are `null` whenever a knob
    # was left to be derived — see `_resolve_vt_batch_size` and the
    # multivector token-budget derivation).
    doc["params"].update({
        "io_workers": io_workers,
        "io_thread_count": itc,
        "batch_size_by_vector_type": vt_batch_size,
        "multivector_batch_size": mv_batch_size,
        "multivector_query_block": mv_query_block,
    })
    doc.update({
        "started_at": started_at.isoformat(),
        "sharding": {
            "num_jobs": num_jobs,
            "job_rank": job_rank,
            "corpus_files_total": len(all_files),
            "corpus_files_this_worker": len(mine),
            "max_files": max_files,
            # A `--max-files` run read only part of its own slice, so its output
            # is NOT valid ground truth. That has to survive into the artifact
            # record — it is exactly the file someone later mistakes for real.
            "partial_slice": max_files is not None,
        },
        "searches": manifest_searches,
        "counts": {
            # The queries FILE's row count. With `rows` subsets a search covers
            # fewer than this (each search's own count is in `searches[]`), so
            # the name says which number this is rather than letting a reader
            # take it for "queries searched".
            "queries_in_file": n_q,
            # The union across searches of the rows any search actually scored.
            # A search with no `rows` covers the whole file, so one of those
            # makes the union the whole file regardless of what the others subset.
            "queries_searched": queries_searched,
            "corpus_rows_scanned": rows_seen,
            "corpus_bytes_decoded": bytes_seen,
        },
        # `elapsed_seconds` is the whole invocation; `scan_seconds` and the four
        # splits below cover the corpus pass only (see the bf-bench log line).
        # read/filter seconds are SUMMED across io_workers reader threads, so
        # divide by io_workers to compare either with wall time.
        "timing": {
            "elapsed_seconds": round(time.perf_counter() - run_t0, 2),
            "scan_seconds": round(wall, 2),
            "io_wait_seconds": round(io_wait, 2),
            "gpu_seconds": round(gpu_secs, 2),
            "read_seconds_summed": round(read_secs, 2),
            "filter_seconds_summed": round(filter_secs, 2),
            "rows_per_second": round(rows_seen / wall, 1) if wall > 0 else 0,
            "wall_mbps": round(wall_mbps, 1),
            "stream_mbps": round(stream_mbps, 1),
        },
        "output_files": [e["output_file"] for e in manifest_searches],
    })
    run_manifest.write(out, run_manifest.manifest_name(cfg, "compute", job_rank, num_jobs), doc)
    return results
