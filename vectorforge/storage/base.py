from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract base for all storage backends."""

    @abstractmethod
    async def upload_file(self, local_path: str) -> None:
        """Upload a local file (parquet batch)."""
        pass

    @abstractmethod
    async def upload_bytes(self, data: bytes, filename: str) -> None:
        """Upload raw bytes (e.g. manifest JSON)."""
        pass

    @abstractmethod
    async def ensure_ready(self) -> None:
        """Create bucket/repo if it doesn't exist yet."""
        pass

    @property
    @abstractmethod
    def destination(self) -> str:
        """Human-readable destination string for logging/manifest."""
        pass