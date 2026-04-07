from datasets import load_dataset

from vectorforge.sources.base import DatasetSource
from vectorforge.models import Record

_payload_extract_all = lambda row, fields: {k: row[k] for k in fields if k in row}


def _build_text_extractor(text_field: str | None, text_template: str | None):
    """
    Returns a function that extracts text from a row.
    - text_template: format string like "{title}: {abstract}"
    - text_field: single field name (fallback)
    """
    if text_template:
        def extract(row: dict) -> str:
            return text_template.format(**row)
        return extract

    if text_field:
        def extract(row: dict) -> str:
            val = row.get(text_field)
            if val is None:
                raise ValueError(f"Row is missing text field '{text_field}'")
            return val
        return extract

    raise ValueError("Must specify either text_field or text_template")


class HuggingFaceSource(DatasetSource):
    def __init__(
        self,
        dataset_name: str,
        config: str | None = None,
        split: str = "train",
        text_field: str | None = "text",
        text_template: str | None = None,
        payload_fields: list[str] | None = None,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.text_field = text_field
        self.text_template = text_template
        self.payload_fields = payload_fields or []
        self._extract_text = _build_text_extractor(text_field, text_template)
        self._dataset = load_dataset(
            dataset_name, config, streaming=True, split=split
        )

    @property
    def source_name(self) -> str:
        return self.dataset_name

    def stream(self):
        yield from self._dataset

    def extract_text(self, row: dict) -> str:
        """Extract text from a row using text_template or text_field."""
        return self._extract_text(row)

    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        payload = self.extract_payload(row)
        if payload is None:
            payload = _payload_extract_all(row, self.payload_fields) if self.payload_fields else {}
        return Record(
            row_id=row_id,
            source_row_id=0,  # set by get_chunks
            chunk_id=chunk_id,
            chunk_index=0,    # set by get_chunks
            text=self.extract_text(row),
            source=self.source_name,
            payload=payload,
        )