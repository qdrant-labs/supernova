from datasets import load_dataset

from vectorforge.sources.base import DatasetSource
from vectorforge.models import Record


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
        exclude_columns: list[str] | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.text_field = text_field
        self.text_template = text_template
        self.exclude_columns = set(exclude_columns or [])
        self._offset = offset
        self._limit = limit
        self._extract_text = _build_text_extractor(text_field, text_template)
        self._dataset = load_dataset(
            dataset_name, config, streaming=True, split=split
        )

    @property
    def source_name(self) -> str:
        return self.dataset_name

    def get_total_rows(self) -> int:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(self.dataset_name, self.config)
        return builder.info.splits[self.split].num_examples

    def stream(self):
        ds = self._dataset
        if self._offset:
            ds = ds.skip(self._offset)
        if self._limit:
            ds = ds.take(self._limit)
        yield from ds

    def extract_text(self, row: dict) -> str:
        """Extract text from a row using text_template or text_field."""
        return self._extract_text(row)

    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        columns = {k: v for k, v in row.items() if k not in self.exclude_columns}
        return Record(
            row_id=row_id,
            source_row_id=0,
            chunk_id=chunk_id,
            chunk_index=0,
            text=self.extract_text(row),
            columns=columns,
        )