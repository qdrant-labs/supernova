from datasets import load_dataset

from vectorforge.sources.base import DatasetSource
from vectorforge.models import Record

_payload_extract_all = lambda row, fields: {k: row[k] for k in fields if k in row}

class HuggingFaceSource(DatasetSource):
    def __init__(
        self,
        dataset_name: str,
        config: str | None = None,
        split: str = "train",
        text_field: str = "text",
        payload_fields: list[str] | None = None,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.text_field = text_field
        self.payload_fields = payload_fields or []
        self._dataset = load_dataset(
            dataset_name, config, streaming=True, split=split
        )

    @property
    def source_name(self) -> str:
        return self.dataset_name

    def stream(self):
        yield from self._dataset

    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        payload = self.extract_payload(row)
        if payload is None:
            payload = _payload_extract_all(row, self.payload_fields) if self.payload_fields else {}
        return Record(
            row_id=row_id,
            source_row_id=0,  # set by get_chunks
            chunk_id=chunk_id,
            chunk_index=0,    # set by get_chunks
            text=row[self.text_field],
            source=self.source_name,
            payload=payload,
        )
