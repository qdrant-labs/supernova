from abc import ABC, abstractmethod
from typing import Iterator, TYPE_CHECKING

from vectorforge.models import Record

if TYPE_CHECKING:
    from vectorforge.embedders.engine import EmbeddingEngine


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
    def format_record(self, row: dict) -> Record:
        """
        Convert a raw row into a Record.
        The implementor decides which field(s) become `text`.
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @abstractmethod
    def get_total_rows(self) -> int:
        """Return the total number of rows in the source (before any offset/limit)."""
        pass

    def get_chunks(
        self,
        engine: "EmbeddingEngine",
        chunk_size: int = 10_000,
        max_text_length: int | None = None,
    ) -> Iterator[tuple[int, list[Record]]]:
        """
        Default chunking logic. Yields (chunk_id, records[]).
        Long texts are split using the engine's tokenizer.
        If max_text_length is set, texts are truncated before splitting.
        """
        chunk: list[Record] = []
        chunk_id = 0

        for raw_row in self.stream():
            base_record = self.format_record(raw_row)

            if not base_record.text or not base_record.text.strip():
                continue

            text = base_record.text
            if max_text_length and len(text) > max_text_length:
                text = text[:max_text_length]

            for piece in engine.split_text(text):
                chunk.append(Record(text=piece, columns=base_record.columns))

                if len(chunk) == chunk_size:
                    yield chunk_id, chunk
                    chunk = []
                    chunk_id += 1

        if chunk:
            yield chunk_id, chunk
