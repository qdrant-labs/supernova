from abc import ABC, abstractmethod


class DenseEmbedder(ABC):
    """
    Abstract base for dense embedding backends.
    All implementations must be async -- parallelism comes from
    running many embed() calls concurrently via asyncio.gather.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Takes a batch of strings. Returns a list of dense embedding vectors
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
        Override if known statically. Used for buffer size estimation.
        If None, dimensions are inferred from the first batch result.
        """
        return None

    @property
    def max_tokens(self) -> int:
        """Max token length this embedder supports. Must be overridden.

        Used for the manifest and to size the model's own encode-time
        truncation. Text splitting is NOT an embedder concern — it's owned by
        the chunkers module (see issue #12)."""
        raise NotImplementedError("DenseEmbedder subclass must define max_tokens")
