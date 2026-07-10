import asyncio
import logging

from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.models import SparseEmbedding
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)


@EMBEDDERS.register("fastembed")
class FastEmbedSparseEmbedder(Embedder):
    """
    Sparse embedder backed by fastembed's SparseTextEmbedding.
    Supports BM25 and other lexical sparse models from Qdrant/fastembed.

    Example config entry:
        embedders:
          - name: bm25
            kind: sparse
            type: fastembed
            model: Qdrant/bm25
            input_column: text
            modality: text
    """

    output_kind = OutputKind.SPARSE

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
        # Return a large sentinel so per-entry max_length governs instead.
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
