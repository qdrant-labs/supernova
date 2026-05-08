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
        total_rows_override: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.text_field = text_field
        self.text_template = text_template
        self.exclude_columns = set(exclude_columns or [])
        self._offset = offset
        self._limit = limit
        self._total_rows_override = total_rows_override
        self._extract_text = _build_text_extractor(text_field, text_template)
        self._dataset = load_dataset(dataset_name, config, streaming=True, split=split)

    @property
    def source_name(self) -> str:
        return self.dataset_name

    def get_total_rows(self) -> int:
        # explicit override wins. Use this for huge datasets where HF only
        # converts a sample to parquet (e.g. dclm-edu shows num_rows=1.1M but is
        # really ~1B); the datasets-server / builder.info will both lie.
        if self._total_rows_override is not None:
            return self._total_rows_override

        from datasets import load_dataset_builder

        builder = load_dataset_builder(self.dataset_name, self.config)
        splits = builder.info.splits or {}
        split_info = splits.get(self.split)
        if split_info is not None and split_info.num_examples:
            return split_info.num_examples

        return self._fetch_row_count_from_datasets_server()

    def _fetch_row_count_from_datasets_server(self) -> int:
        """
        Ask the HF datasets-server for the row count.

        Works for any dataset the Hub has indexed (the common case for public datasets
        and private datasets the caller is authenticated to). Returns num_rows without
        downloading any row data.
        """
        import requests
        from huggingface_hub.utils import build_hf_headers

        params: dict[str, str] = {"dataset": self.dataset_name}
        if self.config:
            params["config"] = self.config

        resp = requests.get(
            "https://datasets-server.huggingface.co/size",
            params=params,
            headers=build_hf_headers(),
            timeout=30,
        )
        if resp.status_code == 202:
            raise RuntimeError(
                f"HF datasets-server is still computing size for {self.dataset_name!r}; "
                "retry in a minute"
            )
        resp.raise_for_status()
        payload = resp.json()

        splits = payload.get("size", {}).get("splits", []) or []
        for entry in splits:
            if entry.get("split") != self.split:
                continue
            if self.config is not None and entry.get("config") != self.config:
                continue
            return int(entry["num_rows"])

        raise ValueError(
            f"datasets-server has no row count for dataset={self.dataset_name!r} "
            f"config={self.config!r} split={self.split!r}"
        )

    def stream(self):
        ds = self._dataset
        if self._offset:
            ds = ds.skip(self._offset)
        if self._limit:
            ds = ds.take(self._limit)
        yield from ds

    def extract_text(self, row: dict) -> str:
        """
        Extract text from a row using text_template or text_field.
        """
        return self._extract_text(row)

    def format_record(self, row: dict) -> Record:
        columns = {k: v for k, v in row.items() if k not in self.exclude_columns}
        return Record(text=self.extract_text(row), columns=columns)
