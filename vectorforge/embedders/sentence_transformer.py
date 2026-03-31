import asyncio
import logging

import torch
from sentence_transformers import SentenceTransformer

from vectorforge.embedders.base import Embedder

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: str = "Alibaba-NLP/gte-multilingual-base",
        batch_size: int = 256,
        device: str | None = None,
    ):
        self._device = device or _detect_device()
        logger.info("Loading %s on %s", model, self._device)
        self._model = SentenceTransformer(model, device=self._device)
        self._model_name = model
        self._batch_size = batch_size
        self._dimensions_val = self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int | None:
        return self._dimensions_val

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Small wrapper to make sure that the blocking _encode method runs in a thread, allowing for concurrency across batches.
        """
        return await asyncio.to_thread(self._encode, texts)
