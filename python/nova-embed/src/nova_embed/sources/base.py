import fnmatch
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

from nova_embed import media
from nova_embed.media import Modality
from nova_embed.models import Record

if TYPE_CHECKING:
    from nova_embed.chunkers import Chunker


# Provenance columns stamped onto each row when include_provenance=True. Shared
# across sources so the output schema is identical whichever source produced it.
SOURCE_FILE_COLUMN = "source_file_name"
SOURCE_ROW_COLUMN = "source_row_number"


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


def files_for_shard(paths: list[str], job_rank: int, num_jobs: int) -> list[str]:
    """Assign whole files to a rank via a balanced *contiguous* split.

    File-granular sharding for sources with no cheap row index (e.g. jsonl,
    which has no footer): rather than mapping a row window to files, each rank
    owns a contiguous block of ~len(paths)/num_jobs files. Contiguous (not
    round-robin) so a rank's prefetch downloads adjacent files in order.

    The first ``len(paths) % num_jobs`` ranks get one extra file, so counts
    differ by at most one and every file is covered exactly once. When
    ``num_jobs > len(paths)`` the trailing ranks get an empty list (idle
    workers — the caller warns).
    """
    if num_jobs <= 0:
        raise ValueError(f"num_jobs must be positive, got {num_jobs}")
    if not 0 <= job_rank < num_jobs:
        raise ValueError(f"job_rank {job_rank} out of range for num_jobs {num_jobs}")
    n = len(paths)
    base, extra = divmod(n, num_jobs)
    # ranks [0, extra) get base+1 files; the rest get base.
    if job_rank < extra:
        start = job_rank * (base + 1)
        count = base + 1
    else:
        start = extra * (base + 1) + (job_rank - extra) * base
        count = base
    return paths[start : start + count]


def filter_paths(paths: list[str], pattern: str | list[str] | None) -> list[str]:
    """Filter a list of repo file paths.

    Patterns:
      - None: pass-through.
      - "regex:<expr>": treat <expr> as a Python regex (re.search).
      - any other string: glob (fnmatch).
      - list of patterns: union (a path matches if it matches any pattern),
        de-duplicated, preserving first-seen order.

    Shared by the HF (parquet) and jsonl sources so path selection behaves
    identically across formats.
    """
    if pattern is None:
        return list(paths)
    if isinstance(pattern, list):
        out: list[str] = []
        seen: set[str] = set()
        for sub in pattern:
            for p in filter_paths(paths, sub):
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out
    if pattern.startswith("regex:"):
        rx = re.compile(pattern[len("regex:") :])
        return [p for p in paths if rx.search(p)]
    return fnmatch.filter(paths, pattern)


def apply_record_projection(
    row: dict,
    render_columns: dict[str, str],
    exclude_columns: set[str],
) -> Record:
    """Normalize a raw row into a Record: render derived columns, then exclude.

    Derived columns are rendered from the FULL row first (a template may
    reference a column the user then excludes), then exclude_columns is applied.
    Shared by every source's ``format_record`` so the projection semantics can't
    drift between formats.
    """
    rendered = {name: tpl.format(**row) for name, tpl in render_columns.items()}
    columns = {k: v for k, v in row.items() if k not in exclude_columns}
    columns.update(rendered)
    return Record(row=columns)


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
    """Source-row accounting for the manifest — a skipped row is quiet, never
    silent, and a rank that consumed only half its window must be able to say so.

    `rows_seen` counts rows read FROM THE SOURCE, which is not the number of
    records written: a splitting chunker turns one row into N records. Only
    this counter is comparable with a rank's assigned row window, and only it
    gives an honest skip RATE."""

    rows_skipped: int = 0
    rows_seen: int = 0


def iter_chunks(
    source: DatasetSource,
    input_groups: list[dict[str, Modality]],
    chunk_size: int,
    on_empty_input: str = "skip",
    chunker: "Chunker | None" = None,
    split_column: str | None = None,
    stats: EmptyInputStats | None = None,
) -> Iterator[tuple[int, list[Record]]]:
    """Assemble embedding batches from a source's rows.

    Sits between the source (pure row producer) and the workers: applies the
    empty-input policy per input GROUP (one group = one embedder entry's
    column->modality inputs), optionally splits ONE column via the chunker
    (config validation guarantees a splitting chunker implies a single input
    column), and packs Records into batches of ``chunk_size``.

    A group is empty only when ALL of its columns are empty — a multimodal
    entry's row with just a text or just an image is a valid input, not a
    policy event. Single-column entries degenerate to the per-column check.

    Note ``chunk_size`` is the embedding *batch* size, distinct from the
    chunker's text splitting.
    """
    if (chunker is None) != (split_column is None):
        raise ValueError("chunker and split_column must be passed together")

    # decode/emptiness view: every distinct input column with its modality
    input_specs: dict[str, Modality] = {}
    for group in input_groups:
        input_specs.update(group)

    chunk: list[Record] = []
    chunk_id = 0
    checked_columns = False

    for raw_row in source.stream():
        if stats is not None:
            stats.rows_seen += 1
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

        col_empty = {
            col: media.is_empty(row.get(col), modality)
            for col, modality in input_specs.items()
        }
        empty_groups = [g for g in input_groups if all(col_empty[c] for c in g)]
        if len(empty_groups) == len(input_groups):
            # nothing to embed for ANY entry — skipping is the only sane move
            if stats is not None:
                stats.rows_skipped += 1
            if on_empty_input == "error":
                raise ValueError(
                    f"empty input for column(s) {sorted(input_specs)} "
                    f"(on_empty_input=error). Row: { _row_summary(row) }"
                )
            continue
        if empty_groups:
            if on_empty_input == "error":
                empty_cols = sorted({c for g in empty_groups for c in g})
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

        if chunker is not None and not col_empty.get(split_column, False):
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
