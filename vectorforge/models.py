from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    """
    A single record from a dataset source, before embedding.
    """
    row_id: int
    chunk_id: int
    text: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedRecord:
    """
    A Record after embedding.
    """
    row_id: int
    chunk_id: int
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
