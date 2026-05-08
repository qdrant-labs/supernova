"""Abstract base class for reading pre-embedded parquet data."""

import json
import logging
import re

from abc import ABC, abstractmethod
from typing import Generator, Iterable

import duckdb

logger = logging.getLogger(__name__)

VECTOR_TYPES = {"dense", "sparse", "multivector"}

# Word-boundary match so column "filename" matches but "myfilename" does not.
_FILENAME_TOKEN = re.compile(r"\bfilename\b")
_FILE_ROW_NUMBER_TOKEN = re.compile(r"\bfile_row_number\b")


class DataReader(ABC):
    """
    Abstract base for reading pre-embedded parquet files via DuckDB.
    Subclasses provide the DuckDB-readable path (S3, HF, local, etc.).

    id_expression: DuckDB SQL expression that yields the point id for each row.
        A bare column name is the simplest form (e.g. "row_id"); any DuckDB
        expression that returns UBIGINT or a UUID string also works -- e.g.
        ``hash(text)`` for content-deduplicated ids, ``hash(filename, row_id)``
        for a globally-unique key across files, or ``uuid()`` for random
        per-row UUIDs. If the expression references ``filename`` or
        ``file_row_number``, the loader auto-enables the matching
        ``read_parquet(..., filename=true, file_row_number=true)`` flags so
        those virtual columns are available. Use ``file_row_number`` for the
        physical row index within the parquet (matches pyarrow's order); do
        NOT use ``ROW_NUMBER() OVER (PARTITION BY filename)`` -- that reflects
        DuckDB's parallel scan order and will not match brute-force IDs.
    vectors: dict of vector name -> spec, where each spec has:
        - type: "dense" | "sparse" | "multivector"
        - column: parquet column name
        - distance / comparator are read by the vector store, ignored here
    payload_fields: dict of payload key -> parquet column name. JSON-string
        values that parse to a dict are unpacked into the payload.
    """

    def __init__(
        self,
        id_expression: str = "row_id",
        vectors: dict[str, dict] | None = None,
        payload_fields: dict[str, str] | None = None,
        duckdb_memory_limit: str = "2GB",
        duckdb_threads: int = 2,
    ):
        self._conn = None
        self.id_expression = id_expression
        self.vectors = vectors or {
            "dense": {"type": "dense", "column": "dense_embedding"},
            "sparse": {"type": "sparse", "column": "sparse_embedding"},
            "multivector": {"type": "multivector", "column": "multivector_embedding"},
        }
        self.payload_fields = payload_fields or {}
        self.duckdb_memory_limit = duckdb_memory_limit
        self.duckdb_threads = duckdb_threads
        self._uses_filename = bool(_FILENAME_TOKEN.search(id_expression))
        self._uses_file_row_number = bool(_FILE_ROW_NUMBER_TOKEN.search(id_expression))

        for name, spec in self.vectors.items():
            vtype = spec.get("type")
            if vtype not in VECTOR_TYPES:
                raise ValueError(
                    f"vectors[{name!r}].type must be one of {sorted(VECTOR_TYPES)}, got {vtype!r}"
                )
            if "column" not in spec:
                raise ValueError(f"vectors[{name!r}] is missing required 'column'")

    @property
    @abstractmethod
    def glob_path(self) -> str:
        """
        DuckDB-readable glob path to the parquet files.
        Examples:
          - S3: "s3://bucket/prefix/**/*.parquet"
          - HuggingFace: "hf://datasets/repo/data/subdir/**/*.parquet"
          - Local: "/path/to/files/**/*.parquet"
        """

    @property
    def _parquet_kwargs(self) -> str:
        """
        Comma-prefixed kwargs to pass to ``read_parquet`` based on which
        virtual columns the id_expression references. Returns "" when none.
        """
        parts = []
        if self._uses_filename:
            parts.append("filename=true")
        if self._uses_file_row_number:
            parts.append("file_row_number=true")
        return (", " + ", ".join(parts)) if parts else ""

    @property
    def source_sql(self) -> str:
        """DuckDB FROM-clause expression spanning all parquet files. Used for
        metadata queries (count, dimensions) where the parallel scan is cheap.
        """
        return f"'{self.glob_path}'"

    def _iter_sources(self) -> Iterable[str]:
        """Yield FROM-clause expressions to iterate during a streaming read.

        Default: yield ``source_sql`` once (single combined scan). Subclasses
        with an explicit file list should override and yield one expression per
        file -- this serializes per-file scans so DuckDB releases httpfs /
        decode buffers between files instead of buffering all of them.
        """
        yield self.source_sql

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect()
            self._configure_connection()
            self._register_macros()
            # DuckDB defaults to ~80% of system RAM (and ignores it for httpfs
            # buffers anyway). Both threads and memory_limit are capped low
            # because the planner parallelizes parquet scans aggressively.
            self._conn.execute(f"SET memory_limit = '{self.duckdb_memory_limit}';")
            self._conn.execute(f"SET threads = {self.duckdb_threads};")
        return self._conn

    @abstractmethod
    def _root_uri_prefix(self) -> str:
        """
        URI prefix stripped from DuckDB's ``filename`` column to recover the
        bare key fed into ``make_point_id``. Both sides that compute IDs
        (this loader and the brute-force / generate-queries pipeline) must
        agree on this form.

          S3DataReader → "s3://{bucket}/"
          HuggingFaceDataReader → "hf://datasets/{repo_id}/"
        """

    def _register_macros(self) -> None:
        """
        Register DuckDB macros available in all id_expression / payload SQL.

        Subclasses can override to add more macros, but should call
        ``super()._register_macros()`` first so vf_point_id is registered.
        """
        conn = self._conn
        # UUID formatting: 32-char hex → xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        conn.execute("""
            CREATE OR REPLACE MACRO vf_uuid_from_hex(h) AS (
                substr(h, 1, 8) || '-' || substr(h, 9, 4) || '-' ||
                substr(h, 13, 4) || '-' || substr(h, 17, 4) || '-' || substr(h, 21, 12)
            )
        """)
        # Matches vectorforge.utils.make_point_id exactly: md5(file:row) as UUID.
        conn.execute("""
            CREATE OR REPLACE MACRO make_point_id(source_file, source_row) AS (
                vf_uuid_from_hex(md5(source_file || ':' || CAST(source_row AS VARCHAR)))
            )
        """)
        # Per-source vf_point_id: strips the backend-specific URI prefix from
        # filename to get the bare key that brute-force / query gen also use.
        prefix_len = len(self._root_uri_prefix())
        conn.execute(f"""
            CREATE OR REPLACE MACRO vf_point_id(fname, rnum) AS (
                make_point_id(substr(fname, {prefix_len + 1}), rnum)
            )
        """)

    def _configure_connection(self) -> None:
        """Override to configure the DuckDB connection (e.g. S3 credentials)."""

    def get_dimensions(self) -> dict[str, int]:
        """
        Return per-vector dimensions for dense and multivector vectors.
        Sparse vectors have no fixed dimension and are omitted.
        """
        conn = self._get_connection()
        dims: dict[str, int] = {}
        for name, spec in self.vectors.items():
            col = spec["column"]
            vtype = spec["type"]
            if vtype == "dense":
                # length(list<float>) -> dim
                sql = f"SELECT length({col}) FROM {self.source_sql} WHERE {col} IS NOT NULL LIMIT 1"
            elif vtype == "multivector":
                # column is list<list<float>>; inner-list length is the dim
                sql = (
                    f"SELECT length({col}[1]) FROM {self.source_sql} "
                    f"WHERE {col} IS NOT NULL AND length({col}) > 0 LIMIT 1"
                )
            else:
                # sparse: no fixed dim
                continue
            row = conn.execute(sql).fetchone()
            if row is None or row[0] is None:
                raise RuntimeError(
                    f"No data found for vector {name!r} at {self.source_sql}"
                )
            dims[name] = row[0]
        return dims

    def get_total_count(self) -> int:
        """Get the total number of records."""
        conn = self._get_connection()
        result = conn.execute(f"SELECT count(*) FROM {self.source_sql}").fetchone()
        return result[0]

    def _build_select(self) -> tuple[str, list[tuple[str, str]], list[str]]:
        """
        Returns (select_sql, vector_order, payload_keys).
        vector_order is [(name, type), ...] aligned with SELECT positions after id_expression.
        """
        cols = [self.id_expression]
        vector_order: list[tuple[str, str]] = []
        for name, spec in self.vectors.items():
            cols.append(spec["column"])
            vector_order.append((name, spec["type"]))

        payload_keys = list(self.payload_fields.keys())
        cols.extend(self.payload_fields.values())
        return ", ".join(cols), vector_order, payload_keys

    def _after_source(self, source: str) -> None:
        """Called after each source expression is fully exhausted. Override for cleanup (e.g. delete temp files)."""

    def read_batches(self, batch_size: int = 1000) -> Generator[list[dict], None, None]:
        """
        Stream records in batches via DuckDB fetchmany.
        Each record: {"id", "vectors": {name: value}, "payload": {...}}.
        """
        conn = self._get_connection()
        select_sql, vector_order, payload_keys = self._build_select()
        n_vectors = len(vector_order)

        for source in self._iter_sources():
            logger.info(f"Reading batches from {source}")
            try:
                conn.execute(f"SELECT {select_sql} FROM {source}")

                while True:
                    rows = conn.fetchmany(batch_size)
                    if not rows:
                        break

                    records = []
                    for row in rows:
                        row_id = row[0]
                        vector_values = row[1 : 1 + n_vectors]
                        payload_values = row[1 + n_vectors :]

                        vectors: dict[str, object] = {}
                        for (name, vtype), val in zip(vector_order, vector_values):
                            if vtype == "sparse":
                                # DuckDB returns parquet structs as dicts
                                vectors[name] = {
                                    "indices": list(val["indices"]),
                                    "values": list(val["values"]),
                                }
                            else:
                                vectors[name] = val

                        payload: dict = {}
                        for key, val in zip(payload_keys, payload_values):
                            # Unpack JSON-string columns (e.g. legacy "payload" blob)
                            if isinstance(val, str):
                                try:
                                    parsed = json.loads(val)
                                    if isinstance(parsed, dict):
                                        payload.update(parsed)
                                        continue
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            payload[key] = val

                        records.append(
                            {
                                "id": row_id,
                                "vectors": vectors,
                                "payload": payload,
                            }
                        )
                    yield records
            finally:
                self._after_source(source)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
