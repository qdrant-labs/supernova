"""Abstract base class for reading pre-embedded parquet data."""

import json
import logging

from abc import ABC, abstractmethod
from typing import Generator

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = {
    "id": "row_id",
    "embedding": "embedding",
}

DEFAULT_PAYLOAD_FIELDS = {
    "text": "text",
}


class DataReader(ABC):
    """
    Abstract base for reading pre-embedded parquet files via DuckDB.
    Subclasses provide the DuckDB-readable path (S3, HF, local, etc.).

    columns: maps logical names (id, embedding) to parquet column names
    payload_fields: maps payload key -> parquet column name
        e.g. {"abstract": "text", "source": "source"} produces
        payload = {"abstract": <text col value>, "source": <source col value>}
    """

    def __init__(
        self,
        columns: dict[str, str] | None = None,
        payload_fields: dict[str, str] | None = None,
    ):
        self._conn = None
        self.columns = {**DEFAULT_COLUMNS, **(columns or {})}
        self.payload_fields = payload_fields if payload_fields is not None else {**DEFAULT_PAYLOAD_FIELDS}

    @property
    @abstractmethod
    def glob_path(self) -> str:
        """DuckDB-readable glob path to the parquet files."""

    @property
    def source_sql(self) -> str:
        """DuckDB FROM-clause expression. Override for non-glob sources."""
        return f"'{self.glob_path}'"

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect()
            self._configure_connection()
        return self._conn

    def _configure_connection(self) -> None:
        """Override to configure the DuckDB connection (e.g. S3 credentials)."""

    def get_dimensions(self) -> int:
        """Get vector dimensions by reading a single embedding from the data."""
        conn = self._get_connection()
        embedding_col = self.columns["embedding"]
        result = conn.execute(
            f"SELECT length({embedding_col}) as dim FROM {self.source_sql} LIMIT 1"
        ).fetchone()
        if result is None:
            raise RuntimeError(f"No data found at {self.glob_path}")
        return result[0]

    def get_total_count(self) -> int:
        """Get the total number of records."""
        conn = self._get_connection()
        result = conn.execute(
            f"SELECT count(*) FROM {self.source_sql}"
        ).fetchone()
        return result[0]

    def _build_select(self) -> tuple[str, list[str]]:
        """Build the SELECT clause and return (sql, payload_keys) for row parsing."""
        id_col = self.columns["id"]
        embedding_col = self.columns["embedding"]

        # Collect unique parquet columns needed for payload
        payload_parquet_cols = list(self.payload_fields.values())
        payload_keys = list(self.payload_fields.keys())

        select_parts = [id_col, embedding_col] + payload_parquet_cols
        return ", ".join(select_parts), payload_keys

    def read_batches(self, batch_size: int = 1000) -> Generator[list[dict], None, None]:
        """
        Stream records in batches via DuckDB fetchmany.
        """
        conn = self._get_connection()
        select_sql, payload_keys = self._build_select()

        logger.info(f"Reading batches from {self.source_sql}")
        result = conn.execute(
            f"SELECT {select_sql} FROM {self.source_sql}"
        )

        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                break

            records = []
            for row in batch:
                row_id = row[0]
                embedding = row[1]
                payload_values = row[2:]

                payload = {}
                for key, val in zip(payload_keys, payload_values):
                    # Unpack JSON strings (e.g. the original "payload" column)
                    if isinstance(val, str):
                        try:
                            parsed = json.loads(val)
                            if isinstance(parsed, dict):
                                payload.update(parsed)
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass
                    payload[key] = val

                records.append({
                    "id": row_id,
                    "embedding": embedding,
                    "payload": payload,
                })
            yield records

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
