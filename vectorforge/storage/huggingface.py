import asyncio
import logging

from vectorforge.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class HuggingFaceBackend(StorageBackend):
    """
    Uploads parquet files to a HuggingFace Hub dataset repo.
    Files are uploaded to `data/` so HF auto-detects them as a dataset.
    """

    def __init__(self, repo_id: str, token: str | None = None, private: bool = True):
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self._api = None
        self._ready = False

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    @property
    def destination(self) -> str:
        return f"hf://datasets/{self.repo_id}"

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        api = self._get_api()
        await asyncio.to_thread(
            api.create_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            private=self.private,
            exist_ok=True,
        )
        logger.info("HF dataset repo ready: %s", self.repo_id)
        self._ready = True

    async def upload_file(self, local_path: str, remote_subpath: str | None = None) -> None:
        remote = remote_subpath or local_path.split("/")[-1]
        api = self._get_api()
        await self.ensure_ready()
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=local_path,
            path_in_repo=f"data/{remote}",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        logger.info("Uploaded %s to hf://datasets/%s/data/%s", remote, self.repo_id, remote)

    async def upload_bytes(self, data: bytes, filename: str) -> None:
        api = self._get_api()
        await self.ensure_ready()
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=data,
            path_in_repo=filename,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        logger.info("Uploaded %s to hf://datasets/%s/%s", filename, self.repo_id, filename)