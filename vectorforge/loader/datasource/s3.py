"""S3 parquet data reader."""

import os

from typing import Iterable

from .base import DataReader


class S3DataReader(DataReader):
    """Reads pre-embedded parquet files from S3."""

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        id_column: str = "row_id",
        vectors: dict[str, dict] | None = None,
        payload_fields: dict[str, str] | None = None,
        file_list: list[str] | None = None,
        duckdb_memory_limit: str = "2GB",
        duckdb_threads: int = 2,
    ):
        super().__init__(
            id_column=id_column,
            vectors=vectors,
            payload_fields=payload_fields,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_threads=duckdb_threads,
        )
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/")
        self.file_list = file_list

    @property
    def glob_path(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/**/*.parquet"

    @property
    def source_sql(self) -> str:
        if self.file_list:
            files_literal = ", ".join(f"'{f}'" for f in self.file_list)
            return f"read_parquet([{files_literal}])"
        return f"'{self.glob_path}'"

    def _iter_sources(self) -> Iterable[str]:
        """When a file_list is provided, scan one file per query so DuckDB
        releases httpfs / decode buffers between files. The combined
        read_parquet([...]) form holds buffers for all files at once.
        """
        if self.file_list:
            for f in self.file_list:
                yield f"read_parquet('{f}')"
        else:
            yield self.source_sql

    def _configure_connection(self) -> None:
        conn = self._conn
        conn.execute("INSTALL httpfs; LOAD httpfs;")

        key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

        conn.execute(f"SET s3_region = '{region}';")
        if key and secret:
            conn.execute(f"SET s3_access_key_id = '{key}';")
            conn.execute(f"SET s3_secret_access_key = '{secret}';")

        session_token = os.environ.get("AWS_SESSION_TOKEN", "")
        if session_token:
            conn.execute(f"SET s3_session_token = '{session_token}';")
