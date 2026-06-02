"""Qdrant vector store backend for loading pre-embedded data."""

import asyncio
import logging
import os

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
    # Top-level vectorstore.params keys this backend understands. Anything
    # outside this set at construction time is a typo and raises loudly —
    # silent param drop (e.g. shard_numbr instead of shard_number) is the
    # worst UX. Nested config dicts (hnsw_config, etc.) are validated by
    # qdrant_client's pydantic models, which already forbid extra keys.
    _VALID_PARAM_KEYS = frozenset(
        {
            # non-collection-creation params
            "upsert_wait",
            # create_collection scalar params
            "shard_number",
            "sharding_method",
            "replication_factor",
            "write_consistency_factor",
            "on_disk_payload",
            # create_collection nested-dict params
            "hnsw_config",
            "optimizers_config",
            "wal_config",
            "quantization",
        }
    )

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
        unknown = set(self.params) - self._VALID_PARAM_KEYS
        if unknown:
            raise ValueError(
                f"QdrantVectorStore: unknown params keys {sorted(unknown)}. "
                f"Valid keys: {sorted(self._VALID_PARAM_KEYS)}"
            )
        self.upsert_wait = self._resolve_upsert_wait()
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=180, prefer_grpc=True)

    def _resolve_upsert_wait(self) -> bool:
        # VF_UPSERT_WAIT env var overrides YAML; default false matches the
        # historic hardcoded behavior (fire-and-forget upserts).
        env = os.environ.get("VF_UPSERT_WAIT")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return bool(self.params.get("upsert_wait", False))

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

    def _build_create_collection_kwargs(self, dimensions: dict[str, int]) -> dict:
        """
        Translate ``self.params`` + ``dimensions`` into kwargs for
        ``AsyncQdrantClient.create_collection``.

        Scalar params pass through unchanged. Nested config dicts
        (hnsw_config / optimizers_config / wal_config) are spread into the
        matching ``qdrant_client.models.*ConfigDiff`` — pydantic raises on
        unknown nested keys, so typos at that level surface as
        ``ValidationError`` from qdrant_client without us needing to mirror
        its full schema.
        """
        vectors_config, sparse_vectors_config = self._build_vectors_config(dimensions)
        kwargs: dict = {
            "collection_name": self.collection_name,
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_vectors_config or None,
        }

        p = self.params
        for key in (
            "shard_number",
            "sharding_method",
            "replication_factor",
            "write_consistency_factor",
            "on_disk_payload",
        ):
            if key in p:
                kwargs[key] = p[key]

        if "hnsw_config" in p:
            kwargs["hnsw_config"] = models.HnswConfigDiff(**p["hnsw_config"])
        if "optimizers_config" in p:
            kwargs["optimizers_config"] = models.OptimizersConfigDiff(
                **p["optimizers_config"]
            )
        if "wal_config" in p:
            kwargs["wal_config"] = models.WalConfigDiff(**p["wal_config"])

        quant = self._build_quantization(p)
        if quant is not None:
            kwargs["quantization_config"] = quant

        return kwargs

    async def ensure_collection(self, dimensions: dict[str, int]) -> None:
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name in existing:
            logger.info(
                f"Collection '{self.collection_name}' already exists, skipping creation"
            )
            return

        kwargs = self._build_create_collection_kwargs(dimensions)
        await self._client.create_collection(**kwargs)
        logger.info(
            f"Created collection '{self.collection_name}' with vectors={list(kwargs['vectors_config'])}, "
            f"sparse={list(kwargs.get('sparse_vectors_config') or [])}, "
            f"extra_params={sorted(set(self.params) - {'upsert_wait'})}"
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
        upsert_wait = self.upsert_wait
        # log out the upsert_wait value for debugging
        logger.debug(f"Upsert wait is set to {upsert_wait}")
        for attempt in range(max_retries):
            try:
                await self._client.upsert(
                    collection_name=self.collection_name,
                    points=qdrant_points,
                    # wait=upsert_wait,
                    wait=False # hardcoded for debugging right now
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
