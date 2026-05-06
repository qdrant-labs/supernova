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
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.text_field = text_field

    @property
    def source_name(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def get_total_rows(self) -> int:
        raise NotImplementedError("S3Source.get_total_rows() not yet implemented")

    def stream(self):
        raise NotImplementedError("S3Source.stream() not yet implemented")

    def format_record(self, row: dict) -> Record:
        return Record(text=row[self.text_field], columns=dict(row))
