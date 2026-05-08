"""Qdrant vector store backend for loading pre-embedded data."""

import asyncio
import logging

from qdrant_client import AsyncQdrantClient, models

from .base import VectorStore

logger = logging.getLogger(__name__)


_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}

_COMPARATOR_MAP = {
    "max_sim": models.MultiVectorComparator.MAX_SIM,
}


def _resolve_distance(name: str | None) -> models.Distance:
    return _DISTANCE_MAP[(name or "cosine").lower()]


def _resolve_comparator(name: str | None) -> models.MultiVectorComparator:
    return _COMPARATOR_MAP[(name or "max_sim").lower()]


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        url: str,
        vectors: dict[str, dict],
        api_key: str | None = None,
        collection_name: str = "default",
        params: dict | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vectors = vectors
        self.params = params or {}
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=60)

    def _build_vectors_config(
        self, dimensions: dict[str, int]
    ) -> tuple[dict[str, models.VectorParams], dict[str, models.SparseVectorParams]]:
        vectors_config: dict[str, models.VectorParams] = {}
        sparse_vectors_config: dict[str, models.SparseVectorParams] = {}

        for name, spec in self.vectors.items():
            vtype = spec["type"]
            if vtype == "dense":
                vectors_config[name] = models.VectorParams(
                    size=dimensions[name],
                    distance=_resolve_distance(spec.get("distance")),
                )
            elif vtype == "multivector":
                vectors_config[name] = models.VectorParams(
                    size=dimensions[name],
                    distance=_resolve_distance(spec.get("distance")),
                    multivector_config=models.MultiVectorConfig(
                        comparator=_resolve_comparator(spec.get("comparator")),
                    ),
                )
            elif vtype == "sparse":
                sparse_vectors_config[name] = models.SparseVectorParams()
            else:
                raise ValueError(f"vectors[{name!r}] has unknown type {vtype!r}")

        return vectors_config, sparse_vectors_config

    async def ensure_collection(self, dimensions: dict[str, int]) -> None:
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name in existing:
            logger.info(
                f"Collection '{self.collection_name}' already exists, skipping creation"
            )
            return

        vectors_config, sparse_vectors_config = self._build_vectors_config(dimensions)
        await self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config or None,
        )
        logger.info(
            f"Created collection '{self.collection_name}' with vectors={list(vectors_config)}, "
            f"sparse={list(sparse_vectors_config)}"
        )

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

    def _build_point_vector(self, raw: dict) -> dict:
        out: dict = {}
        for name, val in raw.items():
            vtype = self.vectors[name]["type"]
            if vtype == "sparse":
                out[name] = models.SparseVector(
                    indices=val["indices"],
                    values=val["values"],
                )
            else:
                out[name] = val
        return out

    async def upsert_batch(self, points: list[dict], max_retries: int = 5) -> None:
        qdrant_points = [
            models.PointStruct(
                id=p["id"],
                vector=self._build_point_vector(p["vectors"]),
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        for attempt in range(max_retries):
            try:
                await self._client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points,
                    wait=False,
                )
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                logger.warning(
                    "Upsert failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                await asyncio.sleep(wait)

    async def close(self) -> None:
        await self._client.close()

    @property
    def name(self) -> str:
        return f"qdrant({self.collection_name})"
