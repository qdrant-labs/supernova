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

# params that tune behavior rather than feed create_collection; excluded from
# the create kwargs and from the "extra_params" creation log line.
_NON_CREATION_PARAMS = frozenset({"upsert_wait", "recreate"})


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
            "recreate",
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
            "strict_mode_config",
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
        self.recreate = bool(self.params.get("recreate", False))
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=180, prefer_grpc=True)

    def _resolve_upsert_wait(self) -> bool:
        # NOVA_UPSERT_WAIT env var overrides YAML; default false matches the
        # historic hardcoded behavior (fire-and-forget upserts).
        env = os.environ.get("NOVA_UPSERT_WAIT")
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
                    datatype=spec.get("datatype"),
                    on_disk=spec.get("on_disk"),
                )
            elif vtype == "multivector":
                vectors_config[name] = models.VectorParams(
                    size=dimensions[name],
                    distance=_resolve_distance(spec.get("distance")),
                    datatype=spec.get("datatype"),
                    on_disk=spec.get("on_disk"),
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
        if "strict_mode_config" in p:
            kwargs["strict_mode_config"] = models.StrictModeConfig(
                **p["strict_mode_config"]
            )

        quant = self._build_quantization(p)
        if quant is not None:
            kwargs["quantization_config"] = quant

        return kwargs

    def _existing_mismatches(self, info, dimensions: dict[str, int]) -> list[str]:
        """Compare a live collection against this config, returning human-readable
        mismatch strings (empty list = compatible).

        Only checks the params that are both *immutable after creation* and
        *benchmark-defining* — vector size+distance, shard_number,
        replication_factor — so a stale collection can't silently masquerade as
        the topology you configured. shard_number/replication_factor are only
        checked when the config pins them (otherwise Qdrant's chosen default is
        not a "mismatch").
        """
        mism: list[str] = []
        params = info.config.params

        live_vectors = params.vectors or {}
        for name, spec in self.vectors.items():
            if spec["type"] == "sparse":
                continue
            live = (
                live_vectors.get(name)
                if isinstance(live_vectors, dict)
                else live_vectors
            )
            if live is None:
                mism.append(f"vector {name!r}: absent in existing collection")
                continue
            if live.size != dimensions[name]:
                mism.append(
                    f"vector {name!r} size: existing {live.size} != config {dimensions[name]}"
                )
            want_dist = _resolve_distance(spec.get("distance"))
            if live.distance != want_dist:
                mism.append(
                    f"vector {name!r} distance: existing {live.distance} != config {want_dist}"
                )

        for key in ("shard_number", "replication_factor"):
            if key in self.params and getattr(params, key) != self.params[key]:
                mism.append(
                    f"{key}: existing {getattr(params, key)} != config {self.params[key]}"
                )

        return mism

    async def ensure_collection(self, dimensions: dict[str, int]) -> None:
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection_name in existing:
            if not self.recreate:
                info = await self._client.get_collection(self.collection_name)
                mismatches = self._existing_mismatches(info, dimensions)
                if mismatches:
                    raise ValueError(
                        f"Collection '{self.collection_name}' already exists but does not "
                        f"match this config:\n"
                        + "\n".join(f"  - {m}" for m in mismatches)
                        + "\n\nThese params are immutable after creation. Set "
                        "`params: {recreate: true}` to drop and recreate it, or point at "
                        "a fresh collection_name."
                    )
                logger.info(
                    f"Collection '{self.collection_name}' already exists and matches "
                    f"config, skipping creation"
                )
                return
            logger.warning(
                f"Collection '{self.collection_name}' exists; recreate=true, dropping it"
            )
            await self._client.delete_collection(self.collection_name)

        kwargs = self._build_create_collection_kwargs(dimensions)
        await self._client.create_collection(**kwargs)
        logger.info(
            f"Created collection '{self.collection_name}' with vectors={list(kwargs['vectors_config'])}, "
            f"sparse={list(kwargs.get('sparse_vectors_config') or [])}, "
            f"extra_params={sorted(set(self.params) - _NON_CREATION_PARAMS)}"
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
