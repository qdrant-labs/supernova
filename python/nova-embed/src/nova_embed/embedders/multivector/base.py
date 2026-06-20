from abc import ABC, abstractmethod

from nova_embed.models import MultiVectorEmbedding


class MultiVectorEmbedder(ABC):
    """
    Abstract base for multi-vector embedding backends (ColBERT-style).
    Each text produces N vectors of D floats, where N varies per input.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        """
        Takes a batch of strings. Returns a list of multi-vector embeddings
        in the same order.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier, written into parquet."""
        pass

    @property
    def dimensions(self) -> int | None:
        """
        Per-vector dimension (the D in N×D). None if not known statically.
        """
        return None

    @property
    def max_tokens(self) -> int:
        """Max token length this embedder supports. Must be overridden.

        Text splitting is owned by the chunkers module, not embedders (issue #12)."""
        raise NotImplementedError("MultiVectorEmbedder subclass must define max_tokens")
