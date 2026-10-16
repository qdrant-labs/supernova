"""Uniform parquet IO over local paths and ``s3://`` URIs, via pyarrow.fs.

One abstraction, no per-backend code in the compute/merge logic: a `Store` wraps
a pyarrow FileSystem + root, and lists/reads/writes parquet the same way whether
the root is a local directory or an S3 prefix. S3 uses the standard AWS
credential chain (env / profile / instance role); region from `AWS_REGION` or
pyarrow's per-bucket resolution.
"""

from __future__ import annotations

import logging
import os
import threading

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


@dataclass
class ParquetFile:
    read_path: str  # path handed to pyarrow (includes the bucket for s3)
    key: str        # loader-consistent filename used for id derivation
    size: int | None = None   # bytes, from the listing (content-ish identity)


# Parallel ranged-read tuning for LARGE parquet files (see Store.read_columns).
# A reader can only parallelize along a file's internal structure (row groups /
# column chunks). Files written with ~ONE row group cause the download to read
# one giant column chunk — so the read degenerates into a single sequential stream
# that no IO thread pool can speed up. Fetching the whole file as many
# concurrent byte ranges FIRST (the `aws s3 cp` strategy), then parsing from
# memory, sidesteps that regardless of parquet internals, on s3 and POSIX roots
# alike. OPT-IN via `Store(uri, ranged_get=True)` — wired to
# `params.io_ranged_get` in the compute config — and even then only for files
# of at least _RANGED_GET_MIN_BYTES (below that the normal reader is already
# fine). Costs one file's raw bytes of extra RAM while that file is parsed.
_RANGED_GET_BYTES = 64 * 1024 * 1024
_RANGED_GET_CONCURRENCY = 24
_RANGED_GET_MIN_BYTES = 256 * 1024 * 1024
# Process-wide pool for concurrent ranged GETs, capped at 512 threads.
# Per-file concurrency is limited separately. Threads are created on demand;
# NOVA_BF_FETCH_POOL_SIZE overrides the default at pool initialization.
_FETCH_POOL_SIZE_DEFAULT = 512
_FETCH_POOL: ThreadPoolExecutor | None = None
_FETCH_POOL_PID: int | None = None
_FETCH_POOL_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def _reset_fetch_pool_in_child() -> None:
    # A fork while another thread held the lock would otherwise deadlock the
    # child's first `_fetch_pool()`; the parent's pool threads don't exist there.
    global _FETCH_POOL_LOCK, _FETCH_POOL
    _FETCH_POOL_LOCK = threading.Lock()
    _FETCH_POOL = None


os.register_at_fork(after_in_child=_reset_fetch_pool_in_child)


def _fetch_pool_size() -> int:
    raw = os.environ.get("NOVA_BF_FETCH_POOL_SIZE", "").strip()
    try:
        size = int(raw) if raw else _FETCH_POOL_SIZE_DEFAULT
        if size < 1:
            raise ValueError(size)
    except ValueError:
        logger.warning("NOVA_BF_FETCH_POOL_SIZE=%r is not a positive integer; "
                       "using %d", raw, _FETCH_POOL_SIZE_DEFAULT)
        size = _FETCH_POOL_SIZE_DEFAULT
    return size


def _fetch_pool() -> ThreadPoolExecutor:
    """The process-wide ranged-GET pool, created on first use.
    
    Re-created after a fork, since a child inherits the executor
    object but not its threads."""
    global _FETCH_POOL, _FETCH_POOL_PID
    with _FETCH_POOL_LOCK:
        if _FETCH_POOL is None or _FETCH_POOL_PID != os.getpid():
            _FETCH_POOL = ThreadPoolExecutor(
                max_workers=_fetch_pool_size(), thread_name_prefix="nova-bf-fetch",
            )
            _FETCH_POOL_PID = os.getpid()
        return _FETCH_POOL


def _copy_range(dst: np.ndarray, buf) -> None:
    """Copy a fetched range without holding the GIL, avoiding GPU stalls.
    Validate length to prevent unintended NumPy broadcasting."""
    if len(buf) != dst.size:
        raise OSError(f"ranged read returned {len(buf)} bytes, expected {dst.size}")
    np.copyto(dst, np.frombuffer(buf, dtype=np.uint8))


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _fs_and_path(uri: str) -> tuple[pafs.FileSystem, str]:
    if "://" in uri:
        return pafs.FileSystem.from_uri(uri)
    # bare local path (relative or absolute)
    return pafs.LocalFileSystem(), os.path.abspath(uri)


class Store:
    """A Parquet root backed by local storage or S3."""

    def __init__(self, uri: str, ranged_get: bool = False,
                 ranged_get_concurrency: int | None = None):
        self.uri = uri
        self.is_s3 = _is_s3(uri)
        # Optional parallel ranged reads for large files
        self.ranged_get = ranged_get
        # Per-file concurrency; callers reading multiple files should divide
        # this to bound total concurrent GETs.
        self.ranged_get_concurrency = ranged_get_concurrency or _RANGED_GET_CONCURRENCY
        self.fs, self.root = _fs_and_path(uri)

    def _loader_key(self, read_path: str) -> str:
        # The loader's id `filename` is the object key (s3, no bucket) or the
        # absolute path (local). pyarrow's s3 path includes the bucket as the
        # first segment, so strip it to match.
        return read_path.split("/", 1)[1] if self.is_s3 else read_path

    def list_parquets(self, subpath: str | None = None) -> list[ParquetFile]:
        """
        Every `*.parquet` under the root (or root/subpath), sorted.
        """
        base = f"{self.root.rstrip('/')}/{subpath}" if subpath else self.root
        info = self.fs.get_file_info(base)
        if info.type == pafs.FileType.File:
            entries = [info]
        elif info.type == pafs.FileType.NotFound:
            return []
        else:
            entries = self.fs.get_file_info(pafs.FileSelector(base, recursive=True))
        out = [
            ParquetFile(read_path=e.path, key=self._loader_key(e.path), size=e.size)
            for e in entries
            if e.type == pafs.FileType.File and e.path.endswith(".parquet")
        ]
        out.sort(key=lambda f: f.read_path)
        return out

    def read_schema(self, read_path: str) -> pa.Schema:
        """The file's schema, footer only — no column data.

        Used to record what dtype the vectors were STORED as (see
        `results.provenance`), which the loaders can't report because they
        upcast to float32 on the way in.
        """
        return pq.read_schema(read_path, filesystem=self.fs)

    def read_columns(self, read_path: str, columns: list[str] | None,
                     ranged: bool | None = None) -> pa.Table:
        """`columns` of one parquet file.

        `ranged` overrides the store's `ranged_get` for this one call; `None`
        (the default) uses the store setting.

        WHY THE OVERRIDE EXISTS. Both paths below project `columns`, so they
        return the same table -- they differ only in how the bytes arrive.
        Ranged-GET buffers the WHOLE file (`_ranged_download` allocates the
        file's full size) and then parses the wanted columns out of it, which
        is the right trade when the caller wants most of the file's bytes, as
        the vector scan does. It is the wrong trade for a narrow column: the
        `tiebreak='id'` startup pass wants one id column, ~180 KB of a ~4.3 GB
        pubmed shard, and buying all 4.3 GB to keep 0.004% of it cost 1055 s
        per rank and (at the default 32-way fan-out) ~137 GB of resident
        buffers, which OOM-killed all 16 ranks of a pubmed GT run on
        2026-09-22 before a single corpus vector was scored.

        Note what the ranged branch gates on: FILE size. That answers "is this
        file big?" when the question that decides the trade is "are we reading
        most of it?" -- so a caller reading one narrow column has to opt out
        here rather than rely on the size gate to do the right thing.
        """
        use_ranged = self.ranged_get if ranged is None else ranged
        if use_ranged:
            size = self.fs.get_file_info(read_path).size
            if size is not None and size >= _RANGED_GET_MIN_BYTES:
                return pq.read_table(
                    pa.BufferReader(self._ranged_download(read_path, size)),
                    columns=columns,
                )
        # Arrow-native read with column projection, parallel I/O, and decoding.
        # Faster for files with many row groups; use io_ranged_get for few row groups.
        return pq.read_table(read_path, filesystem=self.fs, columns=columns,
                             pre_buffer=True, use_threads=True)

    def _ranged_download(self, read_path: str, size: int):
        """Download a file using concurrent byte-range reads on S3 or POSIX.

        Limits per-file concurrency using the shared fetch pool. Copies release
        the GIL; failed reads propagate after in-flight requests finish.
        Temporarily requires one file's size in host RAM.
        """
        data = np.empty(size, dtype=np.uint8)
        width = max(1, self.ranged_get_concurrency)
        pool = _fetch_pool()
        with self.fs.open_input_file(read_path) as f:
            def fetch(lo: int) -> None:
                hi = min(lo + _RANGED_GET_BYTES, size)
                _copy_range(data[lo:hi], f.read_at(hi - lo, lo))

            pending: set = set()
            try:
                for lo in range(0, size, _RANGED_GET_BYTES):
                    if len(pending) >= width:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            fut.result()
                    pending.add(pool.submit(fetch, lo))
                done, pending = wait(pending)
                for fut in done:
                    fut.result()
            finally:
                wait(pending)
        return pa.py_buffer(data)

    def write(self, filename: str, table: pa.Table) -> str:
        """
        Write a table to root/filename (creating local parent dirs).
        """
        path = self._prepare_write(filename)
        with self.fs.open_output_stream(path) as sink:
            pq.write_table(table, sink, compression="snappy")
        return path

    def write_bytes(self, filename: str, data: bytes) -> str:
        """Write raw bytes to root/filename — the run manifest (see manifest.py).

        Same path/parent-dir handling as `write`, so a manifest lands beside the
        parquets it describes whether the root is local or s3://.
        """
        path = self._prepare_write(filename)
        with self.fs.open_output_stream(path) as sink:
            sink.write(data)
        return path

    def _prepare_write(self, filename: str) -> str:
        path = f"{self.root.rstrip('/')}/{filename}"
        if not self.is_s3:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


def dense_to_2d(col) -> np.ndarray:
    """Convert an Arrow list column to a contiguous, NumPy-owned (n, dim) array.

    Validates uniform dimensions and non-null rows. Preserves float16/float32
    to avoid unnecessary CPU conversion and PCIe traffic; other types use
    float32. Copies release the GIL, avoiding GPU stalls from Arrow buffer
    cleanup. Float16 conversion to float32 occurs on the GPU.
    """
    chunks = list(col.chunks) if isinstance(col, pa.ChunkedArray) else [col]
    n = sum(len(c) for c in chunks)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    typ = chunks[0].type
    fixed = pa.types.is_fixed_size_list(typ)
    if not (fixed or pa.types.is_list(typ) or pa.types.is_large_list(typ)):
        raise TypeError(f"dense column must be a list of floats, got {typ}")
    parts: list[tuple[int, np.ndarray]] = []
    dim = typ.list_size if fixed else None
    for ch in chunks:
        if len(ch) == 0:
            continue
        if ch.null_count:
            raise ValueError(
                f"dense column has {ch.null_count} null row(s); every corpus "
                "and query row needs a vector"
            )
        if not fixed:
            # Uniform width, checked: O(rows), not O(values).
            lens = np.diff(ch.offsets.to_numpy(zero_copy_only=False))
            if dim is None:
                dim = int(lens[0])
            if not (lens == dim).all():
                raise ValueError(
                    f"dense column is ragged: list lengths {int(lens.min())}.."
                    f"{int(lens.max())}, a dense vector needs one width"
                )
        # `flatten` honours a sliced chunk's logical offset for both list kinds.
        parts.append((len(ch), ch.flatten().to_numpy(zero_copy_only=False)))
    src = parts[0][1].dtype
    dtype = src if src in (np.dtype(np.float16), np.dtype(np.float32)) else np.dtype(np.float32)
    out = np.empty((n, dim), dtype=dtype)
    r = 0
    for rows, flat in parts:
        if flat.size != rows * dim:
            raise ValueError(
                f"dense column chunk holds {flat.size} values for {rows} rows "
                f"of width {dim}"
            )
        np.copyto(out[r:r + rows], flat.reshape(rows, dim), casting="unsafe")
        r += rows
    return out


def _stitch_ragged_chunks(chunks: list) -> tuple[np.ndarray, np.ndarray]:
    """Decode each chunk on its own and concatenate — see `multivector_to_ragged`.

    Every chunk goes through the single-array path, so all of that path's
    validity/slice/shape handling applies per chunk. Only two things are joined
    here: the per-document token counts, re-prefix-summed into int64 offsets,
    and the float32 token matrix.

    A width disagreement between chunks is left to `np.concatenate` to reject.
    The decoder upstream guarantees a consistent token dimension, and this
    function does not know which FILE it is decoding — so it could only say
    "chunk 3 disagrees with chunk 0", which is no more useful than numpy
    naming the two shapes.
    """
    counts_parts: list[np.ndarray] = []
    mats: list[np.ndarray] = []
    dim = 0
    for chunk in chunks:
        off, flat = multivector_to_ragged(chunk)
        counts_parts.append(np.diff(off))
        if flat.shape[1] > 0:
            dim = flat.shape[1]
        mats.append(flat)
    counts = (np.concatenate(counts_parts) if counts_parts
              else np.zeros(0, dtype=np.int64))
    doc_offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    # A chunk of only null/empty docs decodes to a (0, 0) matrix whose zero
    # width would poison the concat, so drop the empties. `dim` is then used
    # only if EVERY chunk was empty, to keep the returned width meaningful.
    nonempty = [m for m in mats if m.shape[0] > 0]
    flat_tokens = (np.concatenate(nonempty, axis=0) if nonempty
                   else np.zeros((0, dim), dtype=np.float32))
    return doc_offsets, flat_tokens


def multivector_to_ragged(col: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """A `list<list<float32>>` column (one doc = outer entry, one D-dim token
    vector = inner entry) → `(doc_offsets, flat_tokens)`.

    Same flatten-the-Arrow-buffer-once approach as `dense_to_2d`/
    `sparse_to_coo_parts` — no per-row/per-token Python. `doc_offsets` is
    length n+1 (token-index prefix sums; doc `i`'s tokens are rows
    `doc_offsets[i]:doc_offsets[i+1]` of `flat_tokens`), `flat_tokens` is the
    `(total_tokens, D)` float32 matrix of every token across every doc,
    concatenated in doc order. This is the exact ragged analog of CSR's
    `(row_offsets, values)`.

    A null outer entry (nova-embed's `on_empty_input="null"`) or a non-null
    but empty inner list both decode to a zero-width span (`doc_offsets[i] ==
    doc_offsets[i+1]`) — i.e. a zero-token doc, which the compute path treats
    as a non-candidate (`-inf`), the same way the sparse path treats a
    zero-overlap doc.

    Robustness (all O(1) / O(n_docs), so the hot corpus path is unaffected):
    - The validity bitmap, not the offsets, decides a null doc's token count.
      Arrow does NOT guarantee a null list slot has equal adjacent offsets, so
      trusting the offsets alone could count a null doc's stray physical span
      as real tokens (nova-embed writes zero-span nulls, so this only guards
      against arrays from other producers / some slice+concat paths).
    - Buffer offsets (outer/inner) are honored so a sliced Arrow array (logical
      offset != 0) decodes correctly rather than misaligning tokens to docs.
    - Wrong-shape input fails loudly: a dense `list<float32>` column, or token
      floats containing nulls (which would silently become NaN and poison
      scoring), raise a clear error instead of an obscure one downstream.

    The token width D is taken from the first token; per-token width variance
    is left to `reshape` to catch (an O(total_tokens) uniformity scan would tax
    the corpus path, and nova-embed emits a uniform width by construction).

    Chunks are decoded INDIVIDUALLY and stitched in numpy."""
    chunks = list(col.chunks) if isinstance(col, pa.ChunkedArray) else [col]
    if len(chunks) > 1:
        return _stitch_ragged_chunks(chunks)
    col = chunks[0] if chunks else col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros(1, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    if not (pa.types.is_list(col.type) or pa.types.is_large_list(col.type)):
        raise TypeError(f"multivector column must be a list of token vectors, got {col.type}")
    inner = col.values  # child ListArray: one entry per token across all docs
    if not (pa.types.is_list(inner.type) or pa.types.is_large_list(inner.type)):
        raise TypeError(
            f"multivector column must nest list<float32> token vectors; inner type is "
            f"{inner.type} (a flat list<float32> is a DENSE vector — use vector_type=dense)"
        )
    outer_off = col.offsets.to_numpy(zero_copy_only=False).astype(np.int64)  # (n+1,), token indices
    lengths = np.diff(outer_off)  # physical span (token count) per doc
    interspersed = False
    if col.null_count:
        valid = np.asarray(col.is_valid())  # (n,) bool
        interspersed = not bool((lengths[~valid] == 0).all())  # a null slot carrying real tokens?
        lengths = np.where(valid, lengths, 0)  # a null doc is always zero-token
    doc_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    if int(doc_offsets[-1]) == 0:
        # every doc null/empty — no tokens, so D is unknowable from the data
        return doc_offsets, np.zeros((0, 0), dtype=np.float32)
    if inner.values.null_count:  # O(1) when there's no validity bitmap (the norm)
        raise ValueError(
            "multivector token values contain nulls — a token vector must be fully "
            "populated float32 (a null would decode to NaN and poison scoring)"
        )
    inner_off = inner.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    tok_lo, tok_hi = int(outer_off[0]), int(outer_off[-1])
    dim = int(inner_off[tok_lo + 1] - inner_off[tok_lo])  # token width, from the first token
    val_lo, val_hi = int(inner_off[tok_lo]), int(inner_off[tok_hi])
    phys = inner.values.to_numpy(zero_copy_only=False)[val_lo:val_hi].reshape(tok_hi - tok_lo, dim)
    if interspersed:  # rare: drop the stray tokens a null slot physically carried
        phys = phys[np.repeat(valid, np.diff(outer_off))]
    return doc_offsets, np.ascontiguousarray(phys, dtype=np.float32)


def sparse_to_coo_parts(col: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A `struct<indices: list<uint32>, values: list<float32>>` column → CSR parts.

    Same flatten-the-Arrow-buffer-once approach as `dense_to_2d`: no per-row
    Python conversion. Returns `(row_offsets, indices, values)` — `row_offsets`
    is length n+1 (CSR `crow_indices`), `indices`/`values` are the flat,
    concatenated-across-rows nonzero entries (CSR `col_indices`/`values`).

    `indices` keeps the STORED integer width — never widened, never copied.
    `values` is float32: `copy=False` passes a float32-stored column straight
    through, and a float64-stored one is NARROWED here (scoring is float32
    throughout, so carrying float64 further would only cost bandwidth).

    It used to widen `indices` to int64 and re-`astype` `values`
    unconditionally, which on a fineweb corpus file (~220 M nonzeros)
    allocated 1.76 GB for the widening and copied 0.88 GB of float32 onto
    itself, in the reader thread, once per file. The int64 width was never
    needed: `_vocab_lookup` indexes its table with whatever integer dtype it
    is handed, and torch's CSR builder is fed the REMAPPED column ids
    (`compute._remap_sparse_file`), not these.

    Consequently callers must not assume a dtype. Everything downstream is
    dtype-agnostic on purpose: `_build_query_vocab` (`np.unique`),
    `_vocab_lookup` (LUT gather or `searchsorted`), `_zero_gate_file_ok`
    (comparisons), `_sparse_file_norms` (accumulates in float64 explicitly).
    """
    col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32)
    idx_list = col.field("indices")
    val_list = col.field("values")
    row_offsets = idx_list.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    indices = idx_list.values.to_numpy(zero_copy_only=False)
    values = val_list.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return row_offsets, indices, values
