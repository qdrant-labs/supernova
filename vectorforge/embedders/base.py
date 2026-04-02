from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Abstract base for all embedding backends.
    All implementations must be async — parallelism comes from
    running many embed() calls concurrently via asyncio.gather.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Takes a batch of strings. Needs to be async to allow for parallelism across batches.
        Returns a list of embedding vectors in the same order.
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
        """Max token length this embedder supports. Must be overridden."""
        raise NotImplementedError("Embedder subclass must define max_tokens")

    def split_text(self, text: str) -> list[str]:
        """
        Split text into pieces that fit within this embedder's token limit.
        Must be overridden — each embedder should use its own tokenizer.
        """
        raise NotImplementedError("Embedder subclass must implement split_text")