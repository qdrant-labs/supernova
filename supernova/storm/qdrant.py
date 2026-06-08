"""Qdrant implementation of :class:`BaseLoadTester`."""

import time

from supernova.storm.base import BaseLoadTester, QueryResult


class QdrantLoadTester(BaseLoadTester):
    """Fires nearest-neighbor queries at a Qdrant collection over gRPC."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection_name: str | None = None,
        vector_name: str | None = None,
        top_k: int = 10,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.top_k = top_k
        self._client = None

    async def setup(self) -> None:
        from qdrant_client import AsyncQdrantClient

        self._client = AsyncQdrantClient(
            url=self.url, api_key=self.api_key, prefer_grpc=True, timeout=60
        )

    def compile_filter(self, spec: dict | None):
        """The ``query.filter`` block is a Qdrant-native filter (``must`` /
        ``should`` / ``must_not`` with field conditions), parsed straight into a
        ``models.Filter`` so what you write in YAML maps 1:1 to the API.
        """
        if not spec:
            return None
        from qdrant_client import models

        return models.Filter(**spec)

    async def query(self, vector: list[float], query_filter=None) -> QueryResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                using=self.vector_name,
                limit=self.top_k,
                query_filter=query_filter,
                with_payload=False,
            )
            ids = [p.id for p in resp.points]
            return QueryResult(latency_s=time.perf_counter() - t0, ok=True, returned_ids=ids)
        except Exception as e:
            # Any failure is recorded as an error sample rather than aborting the run.
            return QueryResult(latency_s=time.perf_counter() - t0, ok=False, error=str(e))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()