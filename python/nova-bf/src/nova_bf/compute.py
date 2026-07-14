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

Per vector_type, searches share GPU work one of two ways (see `run_compute`'s
`has_baseline` / `_process_shared_batch` / `_process_filter_group`):

- If ANY search of that vector_type is unfiltered, EVERY search of that
  vector_type — filtered or not — shares one full-file batch grid: the GPU
  transfer/CSR build and each distinct metric's score matrix are computed
  ONCE per batch, and every search merges its own top-K from those same
  columns (an unfiltered search reads them directly; a filtered search masks
  them down to its own filter's surviving rows first). Masking a raw,
  per-row-independent score matrix is exact, not an approximation, so a
  filtered search's metric never needs to already be used by anyone else —
  computing one more metric on a batch that's already resident on the GPU is
  cheap, unlike a second transfer.
- Otherwise (no search of that vector_type is unfiltered), there's no
  full-file computation to derive from: searches are grouped by exact filter
  equality instead, and each such group compacts its own rows before
  transfer/scoring, once per group per batch.
"""

from __future__ import annotations

import logging
import os
import re
import time

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread

import numpy as np

from tqdm import tqdm

from nova_bf.config import BruteForceConfig, Filter, SearchSpec, filter_key
from nova_bf.filters import evaluate
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


def load_queries_sparse(store: Store, qcfg) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list]]:
    """Sparse analog of `load_queries`: reads the struct<indices,values> column
    from every query file, then densifies once over the query set's own
    vocabulary (see `_build_query_vocab`) — queries are few enough that a dense
    (n_q, vocab_size) matrix is cheap, same as loading Q fully upfront today."""
    cols = [qcfg.sparse_column]
    if qcfg.id_column:
        cols.append(qcfg.id_column)
    cols += [c for c in qcfg.payload_fields if c not in cols]

    counts_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    ids: list[str] = []
    payload: dict[str, list] = {c: [] for c in qcfg.payload_fields}
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

    indices = np.concatenate(indices_parts) if indices_parts else np.zeros(0, np.int64)
    values = np.concatenate(values_parts) if values_parts else np.zeros(0, np.float32)
    counts = np.concatenate(counts_parts) if counts_parts else np.zeros(0, np.int64)
    n_q = len(counts)
    row_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    vocab = _build_query_vocab(indices)
    Q = _sparse_rows_to_dense(row_offsets, indices, values, vocab)
    return Q, vocab, ids, payload


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
    the same rows, whether via `_process_shared_batch` (Path A) or a
    `FilterGroup` (Path B), including a mix of `cosine` and `dot` searches, so
    it must stay metric-agnostic BY CONSTRUCTION. Cosine
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


@dataclass(frozen=True)
class FilterGroup:
    """Specs sharing one vector_type and one exact filter (Path B only — see
    `run_compute`'s `has_baseline`): no spec of this vector_type is
    unfiltered, so there's no full-file score matrix to derive from, and this
    group compacts/transfers/scores its own rows once per batch instead —
    its members share the identical filter by construction, hence the
    identical surviving row set. Doesn't carry the `Filter` object itself —
    nothing downstream needs it (`run_compute`'s own `distinct_filters[key]`
    already holds it, if a caller ever does), and keeping only `key` here
    means there's exactly one place a group's filter identity lives, not two
    that could in principle drift apart."""

    key: str  # this group's filter_key — stored so the per-file loop never re-derives it
    member_idxs: list[int]
    batch_size: int | None  # resolved batch size; None = whole (compacted) file


def _path_b_group_batch_size(configured: int | None, k_floor: int, vt: str) -> int | None:
    """Path B's floor: `k_floor` is the largest `k` among one `FilterGroup`'s
    own (related — they share an identical filter by construction) members.
    Raise `configured` (`params.dense_batch_size`/`sparse_batch_size`) up to
    `k_floor` when it's below — below that, a search can't fill its own
    top-K from one batch and needs extra merge rounds to get there instead,
    which buys nothing here (this group's own memory footprint is the same
    either way), so there's no reason not to raise it."""
    if configured is None or configured >= k_floor:
        return configured
    logger.warning(
        "params.%s_batch_size=%d is below k=%d; raising to k (a smaller "
        "batch can't fill a search's top-K and gives no memory benefit).",
        vt, configured, k_floor,
    )
    return k_floor


def _path_a_batch_size(configured: int | None, k_floor: int, vt: str) -> int | None:
    """Path A's floor: `k_floor` is the largest `k` across EVERY search of
    the vector_type, related or not, since they all share one full-file
    grid. Unlike `_path_b_group_batch_size`, NEVER raise an explicit
    `configured` value here — doing so would let one search's large `k`
    silently blow past a DIFFERENT search's own memory bound, exactly the
    OOM footgun `dense_batch_size`/`sparse_batch_size` exists to prevent.
    Just warn instead: the larger-`k` search takes extra merge rounds to
    fill its own top-K (more of them the further `configured` sits below
    `k_floor` — each round is its own `torch.topk` pass), but at no extra
    GPU memory cost to anyone."""
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


def _build_filter_groups(
    idxs: list[int], specs: list[SearchSpec], spec_filter_key: list[str],
    batch_size_cfg: int | None, vt: str,
) -> list[FilterGroup]:
    """Group `idxs` (every spec of one vector_type, Path B only — none of
    them unfiltered) by exact filter equality via `filter_key`, so specs
    sharing an identical filter compact/transfer/score together once instead
    of each redoing it."""
    by_key: dict[str, list[int]] = {}
    for i in idxs:
        by_key.setdefault(spec_filter_key[i], []).append(i)
    groups = []
    for key, members in by_key.items():
        k_floor = max(specs[m].k for m in members)
        groups.append(FilterGroup(
            key=key,
            member_idxs=members,
            batch_size=_path_b_group_batch_size(batch_size_cfg, k_floor, vt),
        ))
    return groups


def _process_shared_batch(
    batch, member_idxs: list[int], specs: list[SearchSpec], spec_Q, spec_top_scores, spec_top_enc,
    keeps: dict[str, np.ndarray | None], spec_filter_key: list[str], batch_size: int | None,
    gidx: int, device: str,
) -> float:
    """Path A: every spec of this vector_type — filtered or not — shares one
    full-file batch grid. Built once per batch: the GPU transfer/CSR
    (`batch.transfer`) and each DISTINCT metric's score matrix (`score_cache`,
    keyed by metric — every spec needing that metric reads the same tensor,
    filtered or not). Every member then merges its own top-k from those
    shared columns: an unfiltered spec reads them directly; a filtered spec
    masks them down to its own filter's surviving rows first (`local_idx`,
    cached per filter so two specs sharing an identical filter don't
    recompute `nonzero` twice). Returns elapsed wall-clock seconds (folded
    into the caller's `gpu_secs`) — Path A never compacts, so the whole body
    is GPU transfer/score/merge work."""
    import torch

    t0 = time.perf_counter()
    n_rows = batch.n_rows
    if n_rows == 0:
        return time.perf_counter() - t0
    step = batch_size or n_rows
    for r0 in range(0, n_rows, step):
        r1 = min(r0 + step, n_rows)
        sl = batch.transfer(r0, r1, device)
        rows = torch.arange(r0, r0 + sl.n_rows, dtype=torch.int64, device=device)

        score_cache: dict[str, object] = {}
        local_idx_cache: dict[str, object] = {}
        for m in member_idxs:
            s = specs[m]
            if s.metric not in score_cache:
                score_cache[s.metric] = sl.score(spec_Q[m], s.metric)
            scores = score_cache[s.metric]

            if s.filter is None:
                sel_scores, sel_rows = scores, rows
            else:
                fk = spec_filter_key[m]
                if fk not in local_idx_cache:
                    local_np = np.nonzero(keeps[fk][r0:r1])[0]
                    local_idx_cache[fk] = torch.from_numpy(local_np).to(device, non_blocking=True)
                local_idx = local_idx_cache[fk]
                if local_idx.numel() == 0:
                    continue
                sel_scores, sel_rows = scores[:, local_idx], rows[local_idx]

            spec_top_scores[m], spec_top_enc[m] = _merge_topk(
                spec_top_scores[m], spec_top_enc[m], sel_scores, sel_rows, gidx, s.k
            )
    return time.perf_counter() - t0


def _process_filter_group(
    batch, group: FilterGroup, specs: list[SearchSpec], spec_Q, spec_top_scores, spec_top_enc,
    keep: np.ndarray, gidx: int, device: str,
) -> float:
    """Path B: no spec of this vector_type is unfiltered, so there's no
    full-file score matrix to derive from — this group's rows are compacted
    once, then transferred/scored/batched independently, same as a single
    spec running alone would. Returns elapsed wall-clock seconds spent on the
    GPU transfer/score/merge loop (folded into the caller's `gpu_secs`) —
    deliberately excludes `batch.compact()` above, a real CPU-side cost (row
    filtering, array copies) that scales with corpus size, not GPU work; it's
    real time, just not GPU time, so it shouldn't count toward the signal
    `gpu_secs` exists to give (see the `io_wait`/`gpu_secs` split docs at the
    top of this module)."""
    import torch

    compacted, orig_rows = batch.compact(keep)
    n_rows = compacted.n_rows
    if n_rows == 0:
        return 0.0
    t0 = time.perf_counter()
    step = group.batch_size or n_rows
    for r0 in range(0, n_rows, step):
        r1 = min(r0 + step, n_rows)
        sl = compacted.transfer(r0, r1, device)
        rows = torch.from_numpy(orig_rows[r0 : r0 + sl.n_rows]).to(device)

        score_cache: dict[str, object] = {}
        for m in group.member_idxs:
            s = specs[m]
            if s.metric not in score_cache:
                score_cache[s.metric] = sl.score(spec_Q[m], s.metric)
            spec_top_scores[m], spec_top_enc[m] = _merge_topk(
                spec_top_scores[m], spec_top_enc[m], score_cache[s.metric], rows, gidx, s.k
            )
    return time.perf_counter() - t0


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
    searches share it. Per vector_type, searches then share GPU work via
    `_process_shared_batch` (Path A, when some search is unfiltered) or
    `_process_filter_group` (Path B, otherwise) — see this module's
    docstring. Only each search's own scoring and top-K accumulation stay
    independent. Returns `{spec.name: output_path}`.
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
    Q_np_by_vt: dict[str, np.ndarray] = {}
    query_vocab = None  # sparse only: sorted distinct query token ids (see _build_query_vocab)
    query_ids: list[str] | None = None
    payload: dict[str, list] | None = None
    for vt in vts_needed:
        if vt == "sparse":
            Q_np, query_vocab, q_ids, q_payload = load_queries_sparse(qstore, cfg.queries)
            if len(query_vocab) == 0 and len(q_ids) > 0:
                logger.warning(
                    "sparse query vocabulary is empty (every query has zero nonzero entries) — "
                    "every corpus row will score 0; check queries.sparse_column is correct."
                )
        else:
            Q_np, q_ids, q_payload = load_queries(qstore, cfg.queries)
        Q_np_by_vt[vt] = Q_np
        if query_ids is None:
            query_ids, payload = q_ids, q_payload
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

    # Per vector_type: does any spec have no filter? If so every spec of that
    # vector_type shares one full-file batch grid (Path A, `_process_shared_
    # batch`); otherwise specs are grouped by exact filter equality instead
    # (Path B, `_process_filter_group`) since there's no full-file computation
    # to derive from. See this module's docstring for the full rationale.
    spec_filter_key = [filter_key(s.filter) for s in specs]
    distinct_filters: dict[str, Filter | None] = {}
    for s, fk in zip(specs, spec_filter_key):
        distinct_filters.setdefault(fk, s.filter)

    vt_spec_idxs: dict[str, list[int]] = {vt: [] for vt in vts_needed}
    for i, s in enumerate(specs):
        vt_spec_idxs[s.vector_type].append(i)

    vt_configured_batch = {"dense": cfg.params.dense_batch_size, "sparse": cfg.params.sparse_batch_size}
    has_baseline: dict[str, bool] = {}
    path_a_batch_size: dict[str, int | None] = {}
    path_b_groups: dict[str, list[FilterGroup]] = {}
    for vt, idxs in vt_spec_idxs.items():
        has_baseline[vt] = any(specs[i].filter is None for i in idxs)
        if has_baseline[vt]:
            k_floor = max(specs[i].k for i in idxs)
            path_a_batch_size[vt] = _path_a_batch_size(vt_configured_batch[vt], k_floor, vt)
            logger.info(
                "vector_type=%r: %d search(es) share one full-file batch pass (batch_size=%s)",
                vt, len(idxs), path_a_batch_size[vt],
            )
        else:
            path_b_groups[vt] = _build_filter_groups(idxs, specs, spec_filter_key, vt_configured_batch[vt], vt)
            logger.info(
                "vector_type=%r: no unfiltered search — %d distinct filter group(s), "
                "each compacting its own rows independently",
                vt, len(path_b_groups[vt]),
            )

    # Prefetch corpus files with a POOL of reader threads so many S3 GETs are in
    # flight at once — otherwise the GPU sits idle behind one file's latency at a
    # time. Order doesn't matter (the top-K merge is commutative). pyarrow releases
    # the GIL during IO, so threads parallelize.
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
            t0 = time.perf_counter()
            table = cstore.read_columns(f.read_path, read_cols)
            # Decode each vector_type at most ONCE per file, regardless of how many
            # specs need it — wrapped in the batch abstraction (`DenseCorpusBatch`/
            # `SparseCorpusBatch`) in the consumer loop below, where every spec (Path
            # A) or filter group (Path B) of that vector_type shares it.
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
            # One mask per DISTINCT filter (`None` for the unfiltered key),
            # evaluated against the same table — timed separately from the read
            # above (CPU-vectorized work, not IO wait). Keyed by `filter_key` so
            # two specs sharing an identical filter never evaluate it twice.
            keeps = {fk: evaluate(f, table) if f is not None else None for fk, f in distinct_filters.items()}
            t2 = time.perf_counter()
            fq.put((gidx, arrs, ids, keeps, t1 - t0, t2 - t1))

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

    with tqdm(total=len(mine), unit="file", dynamic_ncols=True, desc="bf") as bar:
        for _ in range(len(mine)):
            w0 = time.perf_counter()
            gidx, arrs, ids, keeps, rsec, fsec = fq.get()
            io_wait += time.perf_counter() - w0
            read_secs += rsec
            filter_secs += fsec
            bar.update(1)

            # Wrap this file's decoded arrays in the vector_type-agnostic batch
            # abstraction ONCE — every spec of a vector_type (Path A) or every
            # filter group of it (Path B) shares the same wrapper below, never
            # rebuilding or mutating the underlying decoded arrays.
            batches: dict[str, object] = {}
            if "dense" in vts_needed:
                batches["dense"] = DenseCorpusBatch(arrs["dense"])
            if "sparse" in vts_needed:
                sp_offsets, sp_idx, sp_val, sp_norms = arrs["sparse"]
                batches["sparse"] = SparseCorpusBatch(sp_offsets, sp_idx, sp_val, sp_norms, query_vocab, need_sparse_norms)

            # Whole-file bookkeeping, computed BEFORE any spec's per-vector-type
            # compaction below. rows_seen/bytes_seen are counted once per file per
            # distinct vector_type present (pre-filter), not per spec — with
            # multiple specs there's no longer a single "the" filtered row count
            # to report.
            #
            # corpus_ids is kept only for files where SOME spec could still
            # resolve a hit — i.e. it's unfiltered, or its filter keeps at least
            # one row in this file. `keeps[fk]` is a mask over this file's rows
            # independent of vector_type (filters read payload columns, not the
            # vector columns), so checking it here — before any vector-type-
            # specific compaction below — is exact, not an approximation: a
            # restrictive spec's filter dropping the whole file must never block
            # a DIFFERENT spec's id resolution for that same file, but a file
            # every spec's filter drops needs no ids kept at all.
            if id_col and any(mask is None or mask.any() for mask in keeps.values()):
                corpus_ids[gidx] = ids
            for vt in vts_needed:
                rows_seen += batches[vt].n_rows
                bytes_seen += batches[vt].nbytes

            for vt in vts_needed:
                b = batches[vt]
                if has_baseline[vt]:
                    gpu_secs += _process_shared_batch(
                        b, vt_spec_idxs[vt], specs, spec_Q, spec_top_scores, spec_top_enc,
                        keeps, spec_filter_key, path_a_batch_size[vt], gidx, device,
                    )
                else:
                    for group in path_b_groups[vt]:
                        keep = keeps[group.key]
                        gpu_secs += _process_filter_group(
                            b, group, specs, spec_Q, spec_top_scores, spec_top_enc, keep, gidx, device,
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
