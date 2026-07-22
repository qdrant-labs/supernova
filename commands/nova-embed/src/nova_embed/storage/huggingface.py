import asyncio
import logging
import os

from nova_embed.registry import STORAGE
from nova_embed.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@STORAGE.register("hf")
class HuggingFaceBackend(StorageBackend):
    """
    Uploads parquet batches to a HuggingFace Storage Bucket.

    Buckets are HF's S3-like object storage (powered by Xet); they are
    non-versioned and mutable, addressed by ``hf://buckets/owner/name/...``.
    Files land at ``{prefix}/{remote_subpath}`` inside the bucket; ``prefix``
    is optional and defaults to "" (write straight to the bucket root).
    """

    def __init__(
        self,
        bucket_id: str,
        prefix: str = "",
        token: str | None = None,
        private: bool = True,
    ):
        self.bucket_id = bucket_id
        self.prefix = prefix.strip("/")
        self.token = token
        self.private = private
        self._ready = False

    @classmethod
    def from_config(cls, cfg: dict) -> "HuggingFaceBackend":
        # `bucket_id` is the canonical key; `repo_id` is a legacy alias.
        bucket_id = cfg.get("bucket_id") or cfg.get("repo_id")
        if not bucket_id:
            raise ValueError(
                "storage.type='hf' requires 'bucket_id' (HF bucket like 'owner/name')"
            )
        return cls(
            bucket_id=bucket_id,
            prefix=cfg.get("prefix", ""),
            token=cfg.get("token"),
            private=cfg.get("private", True),
        )

    @property
    def destination(self) -> str:
        if self.prefix:
            return f"hf://buckets/{self.bucket_id}/{self.prefix}"
        return f"hf://buckets/{self.bucket_id}"

    def _remote_path(self, sub: str) -> str:
        sub = sub.lstrip("/")
        return f"{self.prefix}/{sub}" if self.prefix else sub

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        from huggingface_hub import create_bucket

        await asyncio.to_thread(
            create_bucket,
            self.bucket_id,
            private=self.private,
            exist_ok=True,
            token=self.token,
        )
        logger.info("HF bucket ready: %s", self.bucket_id)
        self._ready = True

    async def upload_file(
        self, local_path: str, remote_subpath: str | None = None
    ) -> None:
        from huggingface_hub import batch_bucket_files

        await self.ensure_ready()
        remote = self._remote_path(remote_subpath or local_path.split("/")[-1])
        await asyncio.to_thread(
            batch_bucket_files,
            self.bucket_id,
            add=[(local_path, remote)],
            token=self.token,
        )
        os.remove(local_path)  # staging copy is uploaded; clean it up
        logger.info("Uploaded %s -> hf://buckets/%s/%s", local_path, self.bucket_id, remote)

    async def upload_bytes(self, data: bytes, filename: str) -> None:
        from huggingface_hub import batch_bucket_files

        await self.ensure_ready()
        remote = self._remote_path(filename)
        await asyncio.to_thread(
            batch_bucket_files,
            self.bucket_id,
            add=[(data, remote)],
            token=self.token,
        )
        logger.info("Uploaded bytes -> hf://buckets/%s/%s", self.bucket_id, remote)