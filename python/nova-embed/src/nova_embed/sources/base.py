from abc import ABC, abstractmethod
from typing import Iterator, TYPE_CHECKING

from nova_embed.models import Record

if TYPE_CHECKING:
    from nova_embed.chunkers import Chunker


def files_in_window(
    files_with_counts: list[tuple[str, int]], offset: int, limit: int | None
) -> list[tuple[str, int]]:
    """
    Of an ordered ``[(path, row_count)]`` list, return the files whose row range
    overlaps the window ``[offset, offset + limit)`` (``limit=None`` = open-ended).

    Row-window → file mapping: rank slices are row offsets/limits, but a worker
    only needs the parquet files those rows actually fall in. Shared by the HF
    source's prefetch and the distributed partition estimate so they never drift.
    """
    out: list[tuple[str, int]] = []
    cumulative = 0
    for path, num_rows in files_with_counts:
        file_end = cumulative + num_rows
        if file_end > offset and (limit is None or cumulative < offset + limit):
            out.append((path, num_rows))
        cumulative = file_end
    return out


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
        chunker: "Chunker",
        chunk_size: int = 10_000,
        max_text_length: int | None = None,
    ) -> Iterator[tuple[int, list[Record]]]:
        """
        Default batching logic. Yields (chunk_id, records[]).

        Each row's text is split into pieces by ``chunker`` (model-agnostic),
        and the pieces are packed into batches of ``chunk_size`` records. Note
        ``chunk_size`` is the embedding *batch* size, distinct from the chunker's
        text splitting. If max_text_length is set, texts are truncated before
        splitting.
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

            for piece in chunker.chunk(text):
                chunk.append(Record(text=piece, columns=base_record.columns))

                if len(chunk) == chunk_size:
                    yield chunk_id, chunk
                    chunk = []
                    chunk_id += 1

        if chunk:
            yield chunk_id, chunk
