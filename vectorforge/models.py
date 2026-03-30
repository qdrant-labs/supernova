from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    """
    A single record from a dataset source, before embedding.
    One source row may produce multiple Records if the text is split.
    """
    row_id: int                          # Auto-incrementing ID for this record
    source_row_id: int                   # Original row position in the dataset
    chunk_id: int                        # Pipeline batch this record belongs to
    chunk_index: int                     # Position within a split (0 if not split)
    text: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)


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
    source: str
    embedding: list[float]
    model: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkResult:
    """
    What comes back from a worker after embedding a full chunk.
    """
    chunk_id: int
    records: list[EmbeddedRecord]