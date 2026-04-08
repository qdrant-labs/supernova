from dataclasses import dataclass, field


@dataclass
class Record:
    """
    A single record from a dataset source, before embedding.
    One source row may produce multiple Records if the text is split.

    `columns` carries all original source columns (after exclude_columns filtering).
    """
    row_id: int
    source_row_id: int
    chunk_id: int
    chunk_index: int
    text: str
    columns: dict = field(default_factory=dict)


@dataclass
class EmbeddedRecord:
    """
    A Record after embedding.
    """
    row_id: int
    source_row_id: int
    chunk_id: int
    chunk_index: int
    text: str
    embedding: list[float]
    columns: dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    """
    What comes back from a worker after embedding a full chunk.
    """
    chunk_id: int
    records: list[EmbeddedRecord]