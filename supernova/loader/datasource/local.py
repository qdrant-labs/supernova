"""Local filesystem parquet data reader."""

import logging
import os
from typing import Iterable

from .base import DataReader

logger = logging.getLogger(__name__)


def _strip_file_scheme(uri: str) -> str:
    """``file:///abs/x`` -> ``/abs/x``; non-file:// paths pass through unchanged."""
    return uri[len("file://") :] if uri.startswith("file://") else uri


class LocalDataReader(DataReader):
    """Reads pre-embedded parquet files from the local filesystem.

    ``path`` may be a directory (read recursively as ``<path>/**/*.parquet``), an
    explicit glob (e.g. ``.../*.parquet``), or a single ``.parquet`` file. Closes
    the loop for a fully-local embed -> load run: point this at the embedder's
    ``storage.output_dir``. DuckDB reads local parquet natively, so unlike the S3
    reader there are no credentials and no prefetch step.

    ``file_list`` is the filtered/sharded corpus the loader discovers via
    ``discover_corpus_parquets`` (``file://`` URIs) and always injects: it excludes
    eval/ and manifest artifacts the bare glob would pick up, and ``--num-jobs``
    splits it by rank. When present it overrides the glob.
    """

    def __init__(
        self,
        path: str,
        id_expression: str = "row_id",
        vectors: dict[str, dict] | None = None,
        payload_fields: dict[str, str] | None = None,
        file_list: list[str] | None = None,
        duckdb_memory_limit: str = "2GB",
        duckdb_threads: int = 2,
    ):
        super().__init__(
            id_expression=id_expression,
            vectors=vectors,
            payload_fields=payload_fields,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_threads=duckdb_threads,
        )
        self.path = os.path.abspath(os.path.expanduser(path))
        # discover_corpus_parquets hands back file:// URIs; DuckDB wants plain
        # paths, and abspath keeps filename-based ids aligned with root_dir.
        self.file_list = (
            [os.path.abspath(_strip_file_scheme(f)) for f in file_list]
            if file_list
            else None
        )
        # Base dir per-row filenames are made relative to, so filename-based ids
        # (vf_point_id) are stable regardless of where the dir sits on disk.
        if "*" in self.path:
            self.root_dir = self.path.split("*", 1)[0].rstrip("/")
        elif self.path.endswith(".parquet"):
            self.root_dir = os.path.dirname(self.path)
        else:
            self.root_dir = self.path.rstrip("/")

    @property
    def glob_path(self) -> str:
        if "*" in self.path or self.path.endswith(".parquet"):
            return self.path
        return f"{self.path.rstrip('/')}/**/*.parquet"

    @property
    def source_sql(self) -> str:
        # Prefer the explicit (filtered/sharded) file list; else the glob. Only
        # wrap in read_parquet() when the id_expression needs filename/row-number.
        if self.file_list:
            files_literal = ", ".join(f"'{f}'" for f in self.file_list)
            return f"read_parquet([{files_literal}]{self._parquet_kwargs})"
        if self._parquet_kwargs:
            return f"read_parquet('{self.glob_path}'{self._parquet_kwargs})"
        return f"'{self.glob_path}'"

    def _iter_sources(self) -> Iterable[str]:
        # One scan per file so DuckDB releases decode buffers between files
        # instead of buffering the whole corpus (mirrors S3DataReader).
        suffix = self._parquet_kwargs
        if self.file_list:
            for f in self.file_list:
                yield f"read_parquet('{f}'{suffix})"
        elif self._parquet_kwargs:
            yield f"read_parquet('{self.glob_path}'{suffix})"
        else:
            yield self.source_sql

    def _root_uri_prefix(self) -> str:
        # filename is the absolute local path; stripping root_dir leaves the path
        # relative to it as the bare key fed into make_point_id.
        return self.root_dir + "/"
