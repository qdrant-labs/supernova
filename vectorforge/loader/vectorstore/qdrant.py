"""Qdrant vector store backend for loading pre-embedded data."""

import asyncio
import logging

from qdrant_client import AsyncQdrantClient, models

from .base import VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection_name: str = "default",
        params: dict | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.params = params or {}
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=60)

    async def ensure_collection(self, dimension: int) -> None:
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name in existing:
            logger.info(f"Collection '{self.collection_name}' already exists, skipping creation")
            return

        await self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Created collection '{self.collection_name}' (dim={dimension})")

    async def defer_indexing(self) -> None:
        """
        Set indexing_threshold to 0 so Qdrant stores vectors without building HNSW.
        """
        await self._client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )
        logger.info(f"Deferred indexing on '{self.collection_name}'")

    async def enable_indexing(self) -> None:
        """Restore default indexing_threshold so Qdrant builds the HNSW graph."""
        await self._client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )
        logger.info(f"Enabled indexing on '{self.collection_name}'")

    async def wait_for_indexing(self, poll_interval: float = 5.0) -> None:
        """Poll collection status until indexing is complete."""
        logger.info(f"Waiting for indexing to complete on '{self.collection_name}'...")
        while True:
            info = await self._client.get_collection(self.collection_name)
            if info.status == models.CollectionStatus.GREEN:
                logger.info(f"Indexing complete on '{self.collection_name}'")
                return
            await asyncio.sleep(poll_interval)

    def _build_quantization(self, params: dict) -> models.QuantizationConfig | None:
        quant_cfg = params.get("quantization", {})
        if not quant_cfg:
            return None

        quant_type = quant_cfg.get("type", "scalar")
        if quant_type == "scalar":
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=quant_cfg.get("always_ram", True),
                ),
            )
        elif quant_type == "binary":
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=quant_cfg.get("always_ram", True),
                ),
            )
        return None

    async def upsert_batch(self, points: list[dict], max_retries: int = 3) -> None:
        qdrant_points = [
            models.PointStruct(
                id=p["id"],
                vector=p["embedding"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        for attempt in range(max_retries):
            try:
                await self._client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points,
                )
                return
            except Exception:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(f"Upsert failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                await asyncio.sleep(wait)

    async def close(self) -> None:
        await self._client.close()

    @property
    def name(self) -> str:
        return f"qdrant({self.collection_name})"
