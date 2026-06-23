"""Uniform parquet IO over local paths and ``s3://`` URIs, via pyarrow.fs.

One abstraction, no per-backend code in the compute/merge logic: a `Store` wraps
a pyarrow FileSystem + root, and lists/reads/writes parquet the same way whether
the root is a local directory or an S3 prefix. S3 uses the standard AWS
credential chain (env / profile / instance role); region from `AWS_REGION` or
pyarrow's per-bucket resolution.
"""

from __future__ import annotations

import os

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


@dataclass
class ParquetFile:
    read_path: str  # path handed to pyarrow (includes the bucket for s3)
    key: str        # loader-consistent filename used for id derivation


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _fs_and_path(uri: str) -> tuple[pafs.FileSystem, str]:
    if "://" in uri:
        return pafs.FileSystem.from_uri(uri)
    # bare local path (relative or absolute)
    return pafs.LocalFileSystem(), os.path.abspath(uri)


class Store:
    """A parquet root (local dir or s3:// prefix) you can list / read / write."""

    def __init__(self, uri: str):
        self.uri = uri
        self.is_s3 = _is_s3(uri)
        self.fs, self.root = _fs_and_path(uri)

    def _loader_key(self, read_path: str) -> str:
        # The loader's id `filename` is the object key (s3, no bucket) or the
        # absolute path (local). pyarrow's s3 path includes the bucket as the
        # first segment, so strip it to match.
        return read_path.split("/", 1)[1] if self.is_s3 else read_path

    def list_parquets(self, subpath: str | None = None) -> list[ParquetFile]:
        """Every `*.parquet` under the root (or root/subpath), sorted."""
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

    def read_columns(self, read_path: str, columns: list[str] | None) -> pa.Table:
        return pq.read_table(read_path, filesystem=self.fs, columns=columns)

    def write(self, filename: str, table: pa.Table) -> str:
        """Write a table to root/filename (creating local parent dirs)."""
        path = f"{self.root.rstrip('/')}/{filename}"
        if not self.is_s3:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with self.fs.open_output_stream(path) as sink:
            pq.write_table(table, sink, compression="snappy")
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
