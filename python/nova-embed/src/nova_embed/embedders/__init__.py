# Import every embedder subpackage so the @*_EMBEDDERS.register decorators run
# (importing `nova_embed.embedders` populates all three embedder registries).
from nova_embed.embedders.dense import (
    DenseEmbedder,
    OpenAIEmbedder,
    SentenceTransformerDenseEmbedder,
)
from nova_embed.embedders.sparse import (
    FastEmbedSparseEmbedder,
    SparseEmbedder,
    SentenceTransformerSparseEmbedder,
)
from nova_embed.embedders.multivector import (
    BGEM3MultiVectorEmbedder,
    MultiVectorEmbedder,
)
from nova_embed.embedders.engine import EmbeddingEngine, EmbedResult
from nova_embed.embedders.hybrid import SentenceTransformerHybridEmbedder
from nova_embed.embedders.runner import run_embedder

__all__ = [
    "DenseEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerDenseEmbedder",
    "SparseEmbedder",
    "FastEmbedSparseEmbedder",
    "SentenceTransformerSparseEmbedder",
    "MultiVectorEmbedder",
    "BGEM3MultiVectorEmbedder",
    "SentenceTransformerHybridEmbedder",
    "EmbeddingEngine",
    "EmbedResult",
    "run_embedder",
]
