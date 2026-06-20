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
class MultiVectorEmbedding:
    """
    Multi-vector representation: N vectors of D floats per text.
    N varies per input (typically one vector per token). Used by ColBERT,
    BAAI/bge-m3 multi-vector mode, etc.

    `vectors` may be a nested Python list OR a numpy ndarray of shape (N, D).
    Embedders that produce raw numpy arrays (e.g. bge_m3) pass them through
    unconverted; pooling and pyarrow both accept either form. The type hint is
    kept as list[list[float]] for interop / serialization clarity.
    """

    vectors: list[list[float]]


@dataclass
class Record:
    """
    A single record from a dataset source, before embedding.
    One source row may produce multiple Records if the text is split.

    `columns` carries all original source columns (after exclude_columns filtering).
    """

    text: str
    columns: dict = field(default_factory=dict)


@dataclass
class EmbeddedRecord:
    """
    A Record after embedding. Any combination of dense/sparse/multivector
    may be set depending on which embedders are configured.
    """

    text: str
    dense_embedding: list[float] | None = None
    sparse_embedding: SparseEmbedding | None = None
    multivector_embedding: MultiVectorEmbedding | None = None
    columns: dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    """
    What comes back from a worker after embedding a full chunk.
    """

    chunk_id: int
    records: list[EmbeddedRecord]
