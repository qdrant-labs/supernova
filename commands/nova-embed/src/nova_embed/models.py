from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class OutputKind(str, Enum):
    """What one input becomes: the declared output contract of an embedder.

    Lives here (not in embedders.base) so the light config layer can import it
    without pulling the heavy ML stack.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    MULTIVECTOR = "multivector"


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
    Multi-vector representation: N vectors of D floats per input.
    N varies per input (typically one vector per token). Used by ColBERT,
    BAAI/bge-m3 multi-vector mode, etc.

    `vectors` may be a nested Python list OR a numpy ndarray of shape (N, D).
    Embedders that produce raw numpy arrays (e.g. bge_m3) pass them through
    unconverted; pooling and pyarrow both accept either form. The type hint is
    kept as list[list[float]] for interop / serialization clarity.
    """

    vectors: list[list[float]]


# What a single input turns into. Which variant an embedder produces is declared
# statically via `Embedder.output_kind` — consumers never sniff the shape.
Embedding = Union[list[float], SparseEmbedding, MultiVectorEmbedding]


@dataclass
class Record:
    """
    A single row from a dataset source, before embedding.

    `row` carries every column (after exclude_columns filtering). Embedders pick
    their input out of it by `input_column`. When a chunker is active, one source
    row may produce multiple Records, each with the chunked column rewritten to
    one piece (all other columns replicated).
    """

    row: dict = field(default_factory=dict)


@dataclass
class EmbeddedRecord:
    """
    A Record after embedding.

    `embeddings` maps embedder entry name -> its output for this row. A value is
    None when the entry's input was empty and on_empty_input="null" kept the row.
    """

    row: dict = field(default_factory=dict)
    embeddings: dict[str, Embedding | None] = field(default_factory=dict)


@dataclass
class ChunkResult:
    """
    What comes back from a worker after embedding a full chunk.
    """

    chunk_id: int
    records: list[EmbeddedRecord]
