"""HuggingFace Hub parquet data reader — streams directly, no local download."""

import os

from .base import DataReader


class HuggingFaceDataReader(DataReader):
    """
    Reads pre-embedded parquet files from a HuggingFace dataset repo via DuckDB's hf:// protocol.
    """

    def __init__(
        self,
        repo_id: str,
        subdir: str | None = None,
        id_column: str = "row_id",
        vectors: dict[str, dict] | None = None,
        payload_fields: dict[str, str] | None = None,
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
        self.repo_id = repo_id
        self.subdir = subdir

    @property
    def glob_path(self) -> str:
        if self.subdir:
            return f"hf://datasets/{self.repo_id}/{self.subdir}/**/*.parquet"
        return f"hf://datasets/{self.repo_id}/**/*.parquet"

    def _configure_connection(self) -> None:
        conn = self._conn
        conn.execute("INSTALL httpfs; LOAD httpfs;")

        token = os.environ.get("HF_TOKEN", "")
        if token:
            conn.execute(f"""
                CREATE SECRET hf_token (
                    TYPE HUGGINGFACE,
                    TOKEN '{token}'
                );
                """)
