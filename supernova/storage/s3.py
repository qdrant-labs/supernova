import logging

import aiobotocore.session
from botocore.exceptions import ClientError

from supernova.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class S3Backend(StorageBackend):
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self._ready = False

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

        logger.info("Uploaded s3://%s/%s", self.bucket, key)

    async def upload_bytes(self, data: bytes, filename: str) -> None:
        key = f"{self.prefix}/{filename}"

        session = aiobotocore.session.get_session()
        async with session.create_client("s3") as client:
            await self.ensure_ready()
            await client.put_object(Bucket=self.bucket, Key=key, Body=data)

        logger.info("Uploaded s3://%s/%s", self.bucket, key)
