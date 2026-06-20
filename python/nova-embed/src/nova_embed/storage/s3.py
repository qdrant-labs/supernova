import logging
import os

import aiobotocore.session
from botocore.exceptions import ClientError

from nova_embed.registry import STORAGE
from nova_embed.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@STORAGE.register("s3")
class S3Backend(StorageBackend):
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self._ready = False

    @classmethod
    def from_config(cls, cfg: dict) -> "S3Backend":
        # `output_dir` (local staging dir) is consumed by the pipeline, not the
        # backend — pull only what S3Backend needs.
        return cls(bucket=cfg["bucket"], prefix=cfg["prefix"])

    @property
    def destination(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        session = aiobotocore.session.get_session()
        async with session.create_client("s3") as client:
            try:
                await client.head_bucket(Bucket=self.bucket)
            except ClientError:
                logger.info("Bucket %s does not exist, creating it", self.bucket)
                await client.create_bucket(Bucket=self.bucket)
        self._ready = True

    async def upload_file(
        self, local_path: str, remote_subpath: str | None = None
    ) -> None:
        remote = remote_subpath or local_path.split("/")[-1]
        key = f"{self.prefix}/{remote}"

        session = aiobotocore.session.get_session()
        async with session.create_client("s3") as client:
            await self.ensure_ready()
            with open(local_path, "rb") as f:
                await client.put_object(Bucket=self.bucket, Key=key, Body=f)

        os.remove(local_path)  # staging copy is uploaded; clean it up
        logger.info("Uploaded s3://%s/%s", self.bucket, key)

    async def upload_bytes(self, data: bytes, filename: str) -> None:
        key = f"{self.prefix}/{filename}"

        session = aiobotocore.session.get_session()
        async with session.create_client("s3") as client:
            await self.ensure_ready()
            await client.put_object(Bucket=self.bucket, Key=key, Body=data)

        logger.info("Uploaded s3://%s/%s", self.bucket, key)
