import asyncio
import logging

from nova_embed.embedders.sparse.base import SparseEmbedder
from nova_embed.models import SparseEmbedding
from nova_embed.registry import SPARSE_EMBEDDERS

logger = logging.getLogger(__name__)


@SPARSE_EMBEDDERS.register("fastembed")
class FastEmbedSparseEmbedder(SparseEmbedder):
    """
    Sparse embedder backed by fastembed's SparseTextEmbedding.
    Supports BM25 and other lexical sparse models from Qdrant/fastembed.

    Example config:
        sparse_embedder:
          type: fastembed
          model: Qdrant/bm25
    """

    def __init__(
        self,
        model: str = "Qdrant/bm25",
        batch_size: int = 256,
        cache_dir: str | None = None,
    ):
        from fastembed import SparseTextEmbedding

        logger.info("Loading fastembed sparse model %s", model)
        self._model = SparseTextEmbedding(
            model_name=model,
            cache_dir=cache_dir,
        )
        self._model_name = model
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        # BM25 is lexical (word tokens), no hard transformer limit.
        # Return a large sentinel so the pipeline's max_text_length governs instead.
        return 100_000

    def _encode(self, texts: list[str]) -> list[SparseEmbedding]:
        results = list(self._model.embed(texts, batch_size=self._batch_size))
        return [
            SparseEmbedding(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]

    async def embed(self, texts: list[str]) -> list[SparseEmbedding]:
        return await asyncio.to_thread(self._encode, texts)
