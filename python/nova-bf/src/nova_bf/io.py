"""Uniform parquet IO over local paths and ``s3://`` URIs, via pyarrow.fs.

One abstraction, no per-backend code in the compute/merge logic: a `Store` wraps
a pyarrow FileSystem + root, and lists/reads/writes parquet the same way whether
the root is a local directory or an S3 prefix. S3 uses the standard AWS
credential chain (env / profile / instance role); region from `AWS_REGION` or
pyarrow's per-bucket resolution.
"""

from __future__ import annotations

import os

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


@dataclass
class ParquetFile:
    read_path: str  # path handed to pyarrow (includes the bucket for s3)
    key: str        # loader-consistent filename used for id derivation


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


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _fs_and_path(uri: str) -> tuple[pafs.FileSystem, str]:
    if "://" in uri:
        return pafs.FileSystem.from_uri(uri)
    # bare local path (relative or absolute)
    return pafs.LocalFileSystem(), os.path.abspath(uri)


class Store:
    """A parquet root (local dir or s3:// prefix) you can list / read / write."""

    def __init__(self, uri: str, ranged_get: bool = False):
        self.uri = uri
        self.is_s3 = _is_s3(uri)
        # Opt-in parallel ranged reads for large files
        # (see the module comment above _RANGED_GET_BYTES).
        self.ranged_get = ranged_get
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
            ParquetFile(read_path=e.path, key=self._loader_key(e.path))
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

    def read_columns(self, read_path: str, columns: list[str] | None) -> pa.Table:
        if self.ranged_get:
            size = self.fs.get_file_info(read_path).size
            if size is not None and size >= _RANGED_GET_MIN_BYTES:
                return pq.read_table(
                    pa.BufferReader(self._ranged_download(read_path, size)),
                    columns=columns,
                )
        return pq.read_table(read_path, filesystem=self.fs, columns=columns)

    def _ranged_download(self, read_path: str, size: int):
        """The whole file via _RANGED_GET_CONCURRENCY concurrent byte-range
        reads into one buffer (see the module comment above the constants).
        Built from FileSystem API (`open_input_file`/`read_at`), so it runs
        unchanged on s3 and on POSIX roots.

        `read_at` is documented thread-safe on one Arrow file handle and a
        failed range re-raises out of the pool — the reader thread's existing
        try/except turns that into a loud run failure, same as any other read
        error. The raw buffer is dropped as soon as `read_columns` finishes
        parsing (parquet decompression copies out of it), so peak memory adds
        one file's raw size only while that file is being parsed."""
        data = np.empty(size, dtype=np.uint8)
        view = memoryview(data)
        with self.fs.open_input_file(read_path) as f:
            def fetch(lo: int) -> None:
                hi = min(lo + _RANGED_GET_BYTES, size)
                view[lo:hi] = f.read_at(hi - lo, lo)

            with ThreadPoolExecutor(max_workers=_RANGED_GET_CONCURRENCY) as pool:
                for fut in [
                    pool.submit(fetch, lo)
                    for lo in range(0, size, _RANGED_GET_BYTES)
                ]:
                    fut.result()
        return pa.py_buffer(data)

    def write(self, filename: str, table: pa.Table, row_group_size: int | None = None) -> str:
        """Write a table to `root/filename`.

        Smaller row groups bound memory for downstream streaming reads, since
        Parquet materializes data at row-group granularity.
        """
        path = self._prepare_write(filename)
        with self.fs.open_output_stream(path) as sink:
            pq.write_table(table, sink, compression="snappy",
                           row_group_size=row_group_size)
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


def dense_to_2d(col: pa.ChunkedArray) -> np.ndarray:
    """A list/fixed-size-list-of-float column → a contiguous (n, dim) float32 array.

    Avoids per-row Python conversion: flattens the Arrow values buffer once and
    reshapes. Assumes a uniform vector dimension and no null rows (always true
    for embedding output).
    """
    col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if pa.types.is_fixed_size_list(col.type):
        dim = col.type.list_size
        flat = col.flatten().to_numpy(zero_copy_only=False)
    else:  # variable-length list, uniform width in practice
        flat = col.values.to_numpy(zero_copy_only=False)
        dim = len(flat) // n
    return np.ascontiguousarray(flat.reshape(n, dim), dtype=np.float32)


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
    the corpus path, and nova-embed emits a uniform width by construction)."""
    col = col.combine_chunks()  # ChunkedArray -> a single ListArray
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
    """
    col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    idx_list = col.field("indices")
    val_list = col.field("values")
    row_offsets = idx_list.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    indices = idx_list.values.to_numpy(zero_copy_only=False).astype(np.int64)
    values = val_list.values.to_numpy(zero_copy_only=False).astype(np.float32)
    return row_offsets, indices, values
