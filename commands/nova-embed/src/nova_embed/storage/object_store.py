"""Generic object-store backend over `obstore` (Python bindings to Apache's Rust
`object_store` crate).

One backend for **S3, GCS, and Azure Blob** — plus any **S3-compatible** store
(Cloudflare R2, Backblaze B2, MinIO, DigitalOcean Spaces) via a custom `endpoint`.
The destination is a `path` URI whose scheme selects the store:

    s3://bucket/prefix      gs://bucket/prefix      az://container/prefix

`obstore` parses the URI, so the part after the bucket becomes the store's prefix
automatically — uploads just pass the file's sub-path. Uploads use `obstore`'s
async, multipart-capable `put`.

Credentials come from the standard provider chains (env / config file / instance
role). For S3 we plug in `obstore`'s boto3 credential provider so auth matches the
rest of supernova exactly. GCS/Azure use `obstore`'s built-in env/config resolution.

Registered under both `object_store` and `s3` (the latter is just the friendlier
name; both take `path`).
"""

from __future__ import annotations

import asyncio
import logging
import os

from urllib.parse import urlparse

import obstore

from nova_embed.registry import STORAGE
from nova_embed.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@STORAGE.register("object_store", "s3")
class ObjectStoreBackend(StorageBackend):
    def __init__(
        self,
        path: str,
        *,
        endpoint: str | None = None,
        region: str | None = None,
        config: dict | None = None,
    ):
        self.path = path.rstrip("/")
        self.scheme = urlparse(self.path).scheme
        if not self.scheme:
            raise ValueError(
                f"storage path {path!r} has no scheme; expected e.g. s3://bucket/prefix"
            )
        # Store-construction options; endpoint/region are the common S3-compatible
        # knobs, promoted to top-level config keys for convenience.
        self._config = dict(config or {})
        if endpoint:
            self._config["endpoint"] = endpoint
        if region:
            self._config["region"] = region
        # The obstore store + boto3 client are built lazily (see ensure_ready) so
        # constructing the backend never touches the network or requires creds —
        # matching the old aiobotocore path, where creds resolved at upload time.
        self._store = None
        self._ready = False

    @classmethod
    def from_config(cls, cfg: dict) -> "ObjectStoreBackend":
        path = cfg.get("path")
        if not path:
            raise ValueError(
                "storage requires 'path', e.g. s3://bucket/prefix, gs://bucket/prefix, "
                "az://container/prefix"
            )
        return cls(
            path=path,
            endpoint=cfg.get("endpoint"),
            region=cfg.get("region"),
            config=cfg.get("config"),
        )

    @property
    def destination(self) -> str:
        return self.path

    def _build_store(self):
        from obstore.store import from_url

        kwargs = {}
        if self.scheme == "s3":
            # Reuse boto3's credential chain (env / ~/.aws / instance role) so S3
            # and S3-compatible endpoints authenticate exactly like the rest of
            # supernova. GCS/Azure fall through to obstore's own env/config resolution.
            from obstore.auth.boto3 import Boto3CredentialProvider

            kwargs["credential_provider"] = Boto3CredentialProvider()
        elif self.scheme == "file":
            # LocalStore needs the root to exist before it'll open.
            os.makedirs(urlparse(self.path).path, exist_ok=True)
        return from_url(self.path, config=self._config or None, **kwargs)

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        self._store = self._build_store()
        # Object stores don't create buckets via the data API. Keep the old
        # auto-create convenience for real S3 (head → create via boto3); other
        # providers require the bucket/container to already exist.
        if self.scheme == "s3":
            await asyncio.to_thread(self._ensure_s3_bucket)
        self._ready = True

    def _ensure_s3_bucket(self) -> None:
        import boto3
        from botocore.exceptions import ClientError

        bucket = urlparse(self.path).netloc
        client = boto3.client(
            "s3",
            endpoint_url=self._config.get("endpoint"),
            region_name=self._config.get("region"),
        )
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            logger.info("Bucket %s does not exist, creating it", bucket)
            client.create_bucket(Bucket=bucket)

    async def upload_file(
        self, local_path: str, remote_subpath: str | None = None
    ) -> None:
        await self.ensure_ready()
        sub = (remote_subpath or os.path.basename(local_path)).lstrip("/")
        with open(local_path, "rb") as f:
            await obstore.put_async(self._store, sub, f)  # streams + multipart
        os.remove(local_path)  # staging copy is uploaded; clean it up
        logger.info("Uploaded %s/%s", self.path, sub)

    async def upload_bytes(self, data: bytes, filename: str) -> None:
        await self.ensure_ready()
        sub = filename.lstrip("/")
        await obstore.put_async(self._store, sub, data)
        logger.info("Uploaded %s/%s", self.path, sub)
