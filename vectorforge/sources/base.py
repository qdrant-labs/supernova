from abc import ABC, abstractmethod
from typing import Iterator

from vectorforge.models import Record


class DatasetSource(ABC):
    """
    Abstract base for all dataset sources.
    Implementors must yield chunks of raw records and define how to extract
    the text field from each raw row.
    """

    @abstractmethod
    def stream(self) -> Iterator[dict]:
        """
        Yield raw rows one at a time from the underlying source.
        """
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

    def get_chunks(self, chunk_size: int = 10_000) -> Iterator[tuple[int, list[Record]]]:
        """
        Default chunking logic. Yields (chunk_id, records[]).
        Override if the source has native chunking support.
        """
        chunk: list[Record] = []
        chunk_id = 0
        row_id = 0

        for raw_row in self.stream():
            chunk.append(self.format_record(raw_row, row_id, chunk_id))
            row_id += 1
            if len(chunk) == chunk_size:
                yield chunk_id, chunk
                chunk = []
                chunk_id += 1

        if chunk:
            yield chunk_id, chunk
