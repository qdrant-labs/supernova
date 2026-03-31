from vectorforge.sources.base import DatasetSource
from vectorforge.models import Record


class S3Source(DatasetSource):
    """
    Stub — reads parquet/jsonl files already stored in S3.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        text_field: str = "text",
        payload_fields: list[str] | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.text_field = text_field
        self.payload_fields = payload_fields or []

    @property
    def source_name(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def stream(self):
        raise NotImplementedError("S3Source.stream() not yet implemented")

    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        payload = {k: row[k] for k in self.payload_fields if k in row}
        return Record(
            row_id=row_id,
            source_row_id=0,
            chunk_id=chunk_id,
            chunk_index=0,
            text=row[self.text_field],
            source=self.source_name,
            payload=payload,
        )
