"""HuggingFace Hub parquet data reader — streams directly via DuckDB hf://."""

import os

from typing import Iterable

from .base import DataReader


class HuggingFaceDataReader(DataReader):
    """
    Reads pre-embedded parquet files from a HuggingFace dataset repo via
    DuckDB's hf:// protocol.

    Two modes:
      - whole-repo glob: pass ``repo_id`` (and optional ``subdir`` under
        ``data/``); all parquets are scanned via a single read_parquet glob.
      - explicit file_list: pass full hf://datasets/... URIs (used by
        distributed sharding so each rank only reads its assigned files).
    """

    def __init__(
        self,
        repo_id: str | None = None,
        subdir: str | None = None,
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
        if not repo_id and not file_list:
            raise ValueError("HuggingFaceDataReader requires repo_id or file_list")
        self.repo_id = repo_id
        self.subdir = subdir
        self.file_list = file_list

    @property
    def glob_path(self) -> str:
        # Only meaningful when file_list is not provided. We anchor at data/
        # so eval/ artifacts at the repo root are never picked up by the glob.
        if self.subdir:
            return f"hf://datasets/{self.repo_id}/data/{self.subdir}/**/*.parquet"
        return f"hf://datasets/{self.repo_id}/data/**/*.parquet"

    @property
    def source_sql(self) -> str:
        if self.file_list:
            files_literal = ", ".join(f"'{f}'" for f in self.file_list)
            return f"read_parquet([{files_literal}]{self._parquet_kwargs})"
        if self._parquet_kwargs:
            return f"read_parquet('{self.glob_path}'{self._parquet_kwargs})"
        return f"'{self.glob_path}'"

    def _iter_sources(self) -> Iterable[str]:
        suffix = self._parquet_kwargs
        if self.file_list:
            for f in self.file_list:
                yield f"read_parquet('{f}'{suffix})"
        elif self._parquet_kwargs:
            yield f"read_parquet('{self.glob_path}'{suffix})"
        else:
            yield self.source_sql

    def _root_uri_prefix(self) -> str:
        # vf_point_id strips this from filename so the bare key matches what
        # bare_key_for_uri produces on the brute-force / generate-queries
        # side: "data/..." (everything after hf://datasets/{repo_id}/).
        return f"hf://datasets/{self._effective_repo_id()}/"

    def _effective_repo_id(self) -> str:
        if self.repo_id:
            return self.repo_id
        # Derive from the first file_list entry: hf://datasets/ns/name/...
        first = self.file_list[0]
        rest = first[len("hf://datasets/"):]
        ns, _, tail = rest.partition("/")
        name, _, _ = tail.partition("/")
        return f"{ns}/{name}"

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