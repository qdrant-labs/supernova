import asyncio
import logging

import torch
from sentence_transformers import SparseEncoder

from nova_embed.embedders.sparse.base import SparseEmbedder
from nova_embed.models import SparseEmbedding
from nova_embed.registry import SPARSE_EMBEDDERS

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@SPARSE_EMBEDDERS.register("sentence_transformer")
class SentenceTransformerSparseEmbedder(SparseEmbedder):
    """
    Sparse embedder using sentence-transformers SparseEncoder.
    Works with models like SPLADE, gte-multilingual-base (sparse mode), etc.
    """

    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        model: str = "Alibaba-NLP/gte-multilingual-base",
        batch_size: int = 32,
        device: str | None = None,
        dtype: str = "float32",
        trust_remote_code: bool = False,
    ):
        self._device = device or _detect_device()
        logger.info(
            "Loading sparse encoder %s on %s (dtype=%s)", model, self._device, dtype
        )
        self._model = SparseEncoder(
            model,
            device=self._device,
            trust_remote_code=trust_remote_code,
        )
        self._model_name = model
        self._batch_size = batch_size
        self._max_tokens = self._model.max_seq_length

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _encode(self, texts: list[str]) -> list[SparseEmbedding]:
        results = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        embeddings = []
        for row in results:
            # SparseEncoder returns scipy sparse matrices or dicts depending on version
            if hasattr(row, "toarray"):
                # scipy sparse matrix — convert to indices/values
                dense = row.toarray().squeeze()
                nonzero = dense.nonzero()[0]
                embeddings.append(
                    SparseEmbedding(
                        indices=nonzero.tolist(),
                        values=dense[nonzero].tolist(),
                    )
                )
            elif isinstance(row, dict):
                embeddings.append(
                    SparseEmbedding(
                        indices=list(row.keys()),
                        values=list(row.values()),
                    )
                )
            else:
                raise TypeError(f"Unexpected sparse output type: {type(row)}")

        return embeddings

    async def embed(self, texts: list[str]) -> list[SparseEmbedding]:
        return await asyncio.to_thread(self._encode, texts)
