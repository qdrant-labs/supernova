from dataclasses import dataclass, field


@dataclass
class SparseEmbedding:
    """
    Sparse vector representation: parallel lists of token indices and weights.
    Used by BM25, SPLADE, and other sparse retrieval models.
    """
    indices: list[int]
    values: list[float]


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
    A Record after embedding. One or both of dense/sparse will be set
    depending on which embedders are configured.
    """
    row_id: int
    source_row_id: int
    chunk_id: int
    chunk_index: int
    text: str
    dense_embedding: list[float] | None = None
    sparse_embedding: SparseEmbedding | None = None
    columns: dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    """
    What comes back from a worker after embedding a full chunk.
    """
    chunk_id: int
    records: list[EmbeddedRecord]
