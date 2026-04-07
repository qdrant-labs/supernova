"""HuggingFace Hub parquet data reader — streams directly, no local download."""

import os

from .base import DataReader


class HuggingFaceDataReader(DataReader):
    """
    Reads pre-embedded parquet files from a HuggingFace dataset repo via DuckDB's hf:// protocol.
    """

    def __init__(self, repo_id: str, subdir: str | None = None, columns: dict[str, str] | None = None, payload_fields: dict[str, str] | None = None):
        super().__init__(columns=columns, payload_fields=payload_fields)
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
