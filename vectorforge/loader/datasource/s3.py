"""S3 parquet data reader."""

import os

from .base import DataReader


class S3DataReader(DataReader):
    """Reads pre-embedded parquet files from S3."""

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        columns: dict[str, str] | None = None,
        payload_fields: dict[str, str] | None = None,
        file_list: list[str] | None = None,
    ):
        super().__init__(columns=columns, payload_fields=payload_fields)
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
