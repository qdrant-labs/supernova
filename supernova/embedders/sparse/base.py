from abc import ABC, abstractmethod

from supernova.models import SparseEmbedding


class SparseEmbedder(ABC):
    """
    Abstract base for sparse embedding backends.
    Produces token-weight sparse vectors (e.g. BM25, SPLADE).
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[SparseEmbedding]:
        """
        Takes a batch of strings. Returns a list of SparseEmbedding objects
        in the same order.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        pass

    @property
    def max_tokens(self) -> int:
        """Max token length this embedder supports. Must be overridden.

        Text splitting is owned by the chunkers module, not embedders (issue #12)."""
        raise NotImplementedError("SparseEmbedder subclass must define max_tokens")
