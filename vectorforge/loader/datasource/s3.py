"""S3 parquet data reader."""

import logging
import os

from typing import Iterable

from .base import DataReader

logger = logging.getLogger(__name__)


class S3DataReader(DataReader):
    """Reads pre-embedded parquet files from S3."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        id_expression: str = "row_id",
        vectors: dict[str, dict] | None = None,
        payload_fields: dict[str, str] | None = None,
        file_list: list[str] | None = None,
        duckdb_memory_limit: str = "2GB",
        duckdb_threads: int = 2,
        prefetch: bool = False,
        prefetch_dir: str = "/tmp/vectorforge_loader",
    ):
        super().__init__(
            id_expression=id_expression,
            vectors=vectors,
            payload_fields=payload_fields,
            duckdb_memory_limit=duckdb_memory_limit,
            duckdb_threads=duckdb_threads,
        )
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.file_list = file_list
        self.prefetch = prefetch
        self.prefetch_dir = prefetch_dir
        # maps source_expr -> local_path for cleanup after each file is exhausted
        self._prefetch_map: dict[str, str] = {}

    @property
    def glob_path(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/**/*.parquet"

    @property
    def source_sql(self) -> str:
        if self.file_list:
            files_literal = ", ".join(f"'{f}'" for f in self.file_list)
            return f"read_parquet([{files_literal}]{self._parquet_kwargs})"
        if self._parquet_kwargs:
            return f"read_parquet('{self.glob_path}'{self._parquet_kwargs})"
        return f"'{self.glob_path}'"

    def _download_file(self, s3_uri: str) -> str:
        """
        Download an S3 file to local disk and return the local path.
        """
        import boto3

        os.makedirs(self.prefetch_dir, exist_ok=True)
        safe_name = s3_uri.replace("s3://", "").replace("/", "__")
        local_path = os.path.join(self.prefetch_dir, safe_name)

        if os.path.exists(local_path):
            logger.info("Already cached: %s", s3_uri)
            return local_path

        # s3_uri = "s3://bucket/key/..."
        without_scheme = s3_uri[5:]  # strip "s3://"
        bucket, _, key = without_scheme.partition("/")
        s3 = boto3.client("s3")
        logger.info("Downloading %s -> %s", s3_uri, local_path)
        s3.download_file(bucket, key, local_path)
        size_gb = os.path.getsize(local_path) / 1e9
        logger.info("Downloaded %.2f GB: %s", size_gb, os.path.basename(local_path))
        return local_path

    def _root_uri_prefix(self) -> str:
        # vf_point_id strips this from filename, so the bare key passed into
        # make_point_id is the full S3 key (prefix + path), independent of
        # what prefix was used to scope this loader.
        return f"s3://{self.bucket}/"

    def _iter_sources(self) -> Iterable[str]:
        """
        Yield one FROM-clause expression per file.

        With prefetch=True: downloads each S3 file to local disk before
        yielding the local path expression. _after_source() deletes it once
        all batches from that file have been consumed.

        When id_expression uses `filename`, prefetch mode injects the original
        S3 URI as a literal column so vf_point_id gets the real path, not the
        local temp path. `file_row_number` is enabled directly on the inner
        read_parquet so it reflects the physical row index.
        """
        suffix = self._parquet_kwargs
        if self.file_list:
            for f in self.file_list:
                if self.prefetch:
                    local_path = self._download_file(f)
                    inner_args = (
                        ", file_row_number=true" if self._uses_file_row_number else ""
                    )
                    inner = f"read_parquet('{local_path}'{inner_args})"
                    if self._uses_filename:
                        # Inject original S3 URI so vf_point_id sees the real path
                        expr = f"(SELECT *, '{f}' AS filename FROM {inner})"
                    else:
                        expr = inner
                    self._prefetch_map[expr] = local_path
                    yield expr
                else:
                    yield f"read_parquet('{f}'{suffix})"
        elif self._parquet_kwargs:
            yield f"read_parquet('{self.glob_path}'{suffix})"
        else:
            yield self.source_sql

    def _after_source(self, source: str) -> None:
        local_path = self._prefetch_map.pop(source, None)
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
            logger.info("Deleted local file: %s", local_path)

    def _configure_connection(self) -> None:
        conn = self._conn
        conn.execute("INSTALL httpfs; LOAD httpfs;")

        key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )

        conn.execute(f"SET s3_region = '{region}';")
        if key and secret:
            conn.execute(f"SET s3_access_key_id = '{key}';")
            conn.execute(f"SET s3_secret_access_key = '{secret}';")

        session_token = os.environ.get("AWS_SESSION_TOKEN", "")
        if session_token:
            conn.execute(f"SET s3_session_token = '{session_token}';")
