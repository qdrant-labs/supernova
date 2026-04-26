"""Abstract base class for vector store backends."""

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract base for all vector store backends used in loading."""

    @abstractmethod
    async def ensure_collection(self, dimensions: dict[str, int]) -> None:
        """Create or verify the target collection/index exists.

        dimensions: per-vector size (by vector name) for dense and multivector
        vectors. Sparse vectors are absent.
        """

    async def configure_index(self, params: dict) -> None:
        """Optionally configure the index after collection creation.

        Override this to apply backend-specific index tuning (e.g. HNSW params,
        quantization, on-disk settings) from the config's `params` block.
        Called by the runner after ensure_collection().
        """
        pass # no op by default

    async def defer_indexing(self) -> None:
        """Disable indexing for fast bulk loading. Called before upserts begin."""
        pass  # no-op by default

    async def enable_indexing(self) -> None:
        """Re-enable indexing after bulk load. Called after all upserts complete."""
        pass  # no-op by default

    async def wait_for_indexing(self) -> None:
        """Block until indexing is complete. Called after enable_indexing()."""
        pass  # no-op by default

    @abstractmethod
    async def upsert_batch(self, points: list[dict]) -> None:
        """Upsert a batch of points. Each point: {id, vectors: {name: value}, payload}."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
