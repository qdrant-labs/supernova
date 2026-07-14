from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

from nova_embed import media
from nova_embed.media import Modality
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

    A source is purely a row producer: it yields raw rows and normalizes them
    into Records. WHAT gets embedded out of each row is not its concern — every
    embedder entry declares its own input_column.
    """

    @abstractmethod
    def stream(self) -> Iterator[dict]:
        """Yield raw rows one at a time from the underlying source."""
        pass

    @abstractmethod
    def format_record(self, row: dict) -> Record:
        """Normalize a raw row into a Record (column filtering, derived columns)."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @abstractmethod
    def get_total_rows(self) -> int:
        """Return the total number of rows in the source (before any offset/limit)."""
        pass


@dataclass
class EmptyInputStats:
    """Rows dropped by the empty-input policy. Reported in the manifest — a
    skipped row is quiet, never silent."""

    rows_skipped: int = 0


def iter_chunks(
    source: DatasetSource,
    input_specs: dict[str, Modality],
    chunk_size: int,
    on_empty_input: str = "skip",
    chunker: "Chunker | None" = None,
    split_column: str | None = None,
    stats: EmptyInputStats | None = None,
) -> Iterator[tuple[int, list[Record]]]:
    """Assemble embedding batches from a source's rows.

    Sits between the source (pure row producer) and the workers: applies the
    empty-input policy across every input column, optionally splits ONE column
    via the chunker (config validation guarantees a splitting chunker implies a
    single input column), and packs Records into batches of ``chunk_size``.

    Note ``chunk_size`` is the embedding *batch* size, distinct from the
    chunker's text splitting.
    """
    if (chunker is None) != (split_column is None):
        raise ValueError("chunker and split_column must be passed together")

    chunk: list[Record] = []
    chunk_id = 0
    checked_columns = False

    for raw_row in source.stream():
        row = source.format_record(raw_row).row

        # Harass at launch: a wrong input_column (or one eaten by
        # exclude_columns) dies on the FIRST row, not after N hours of nulls.
        if not checked_columns:
            missing = [c for c in input_specs if c not in row]
            if missing:
                raise ValueError(
                    f"input_column(s) {missing} not found in source rows. "
                    f"Available columns: {sorted(row)}. Check the column name and "
                    f"the source's exclude_columns."
                )
            checked_columns = True

        empties = {
            col: media.is_empty(row.get(col), modality)
            for col, modality in input_specs.items()
        }
        if all(empties.values()):
            # nothing to embed for ANY entry — skipping is the only sane move
            if stats is not None:
                stats.rows_skipped += 1
            if on_empty_input == "error":
                raise ValueError(
                    f"empty input for column(s) {sorted(empties)} "
                    f"(on_empty_input=error). Row: { _row_summary(row) }"
                )
            continue
        if any(empties.values()):
            if on_empty_input == "error":
                empty_cols = sorted(c for c, e in empties.items() if e)
                raise ValueError(
                    f"empty input for column(s) {empty_cols} "
                    f"(on_empty_input=error). Row: { _row_summary(row) }"
                )
            if on_empty_input == "skip":
                if stats is not None:
                    stats.rows_skipped += 1
                continue
            # "null": keep the row; the engine masks the empty input and the
            # writer stores a null embedding.

        if chunker is not None and not empties.get(split_column, False):
            pieces = chunker.chunk(row[split_column])
            records = [Record(row={**row, split_column: piece}) for piece in pieces]
        else:
            records = [Record(row=row)]

        for record in records:
            chunk.append(record)
            if len(chunk) == chunk_size:
                yield chunk_id, chunk
                chunk = []
                chunk_id += 1

    if chunk:
        yield chunk_id, chunk


def _row_summary(row: dict, max_len: int = 200) -> str:
    text = repr(row)
    return text if len(text) <= max_len else text[:max_len] + "…"
