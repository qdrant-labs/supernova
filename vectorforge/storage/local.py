import logging
import os
import shutil

from vectorforge.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class LocalBackend(StorageBackend):
    """Stores parquet files locally. No upload, no cloud."""

    def __init__(self, output_dir: str = "/tmp/vectorforge"):
        self.output_dir = output_dir

    @property
    def destination(self) -> str:
        return self.output_dir

    async def ensure_ready(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    async def upload_file(
        self, local_path: str, remote_subpath: str | None = None
    ) -> None:
        remote = remote_subpath or os.path.basename(local_path)
        dest = os.path.join(self.output_dir, remote)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.abspath(local_path) != os.path.abspath(dest):
            shutil.move(local_path, dest)
        logger.info("Stored %s", dest)

    async def upload_bytes(self, data: bytes, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        logger.info("Stored %s", path)
