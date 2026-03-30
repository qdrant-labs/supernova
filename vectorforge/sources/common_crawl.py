from vectorforge.sources.base import DatasetSource
from vectorforge.models import Record


class CommonCrawlSource(DatasetSource):
    """Stub — Common Crawl source. Implement stream() with WARC/WET parsing."""

    def __init__(self, crawl_id: str, text_field: str = "text"):
        self.crawl_id = crawl_id
        self.text_field = text_field

    @property
    def source_name(self) -> str:
        return f"common-crawl/{self.crawl_id}"

    def stream(self):
        raise NotImplementedError("CommonCrawlSource.stream() not yet implemented")

    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        return Record(
            row_id=row_id,
            source_row_id=0,
            chunk_id=chunk_id,
            chunk_index=0,
            text=row[self.text_field],
            source=self.source_name,
            payload={k: v for k, v in row.items() if k != self.text_field},
        )
