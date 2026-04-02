from abc import ABC, abstractmethod
from typing import Any, Iterator, TYPE_CHECKING

from vectorforge.models import Record

if TYPE_CHECKING:
    from vectorforge.embedders.base import Embedder


class DatasetSource(ABC):
    """
    Abstract base for all dataset sources.
    Implementors must yield chunks of raw records and define how to extract
    the text field from each raw row.
    """

    @abstractmethod
    def stream(self) -> Iterator[dict]:
        """Yield raw rows one at a time from the underlying source."""
        pass

    @abstractmethod
    def format_record(self, row: dict, row_id: int, chunk_id: int) -> Record:
        """
        Convert a raw row into a Record.
        The implementor decides which field(s) become `text`
        and what goes into `payload`.
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    def extract_payload(self, row: dict) -> dict[str, Any] | None:
        """
        Override to define custom payload extraction from a raw row.
        Return a dict to use as the record payload, or None to fall back
        to the default behavior (e.g. payload_fields).
        """
        return None

    def get_chunks(
        self,
        embedder: "Embedder",
        chunk_size: int = 10_000,
    ) -> Iterator[tuple[int, list[Record]]]:
        """
        Default chunking logic. Yields (chunk_id, records[]).
        Long texts are split using the embedder's tokenizer.
        """
        chunk: list[Record] = []
        chunk_id = 0
        row_id = 0
        source_row_id = 0

        for raw_row in self.stream():
            base_record = self.format_record(raw_row, row_id, chunk_id)

            if not base_record.text or not base_record.text.strip():
                source_row_id += 1
                continue

            text_pieces = embedder.split_text(base_record.text)

            for chunk_index, piece in enumerate(text_pieces):
                record = Record(
                    row_id=row_id,
                    source_row_id=source_row_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=piece,
                    source=base_record.source,
                    payload=base_record.payload,
                )
                chunk.append(record)
                row_id += 1

                if len(chunk) == chunk_size:
                    yield chunk_id, chunk
                    chunk = []
                    chunk_id += 1

            source_row_id += 1

        if chunk:
            yield chunk_id, chunk