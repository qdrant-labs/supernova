import asyncio
import logging
import threading

import torch
from sentence_transformers import SentenceTransformer

from supernova.embedders.dense.base import DenseEmbedder

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerDenseEmbedder(DenseEmbedder):
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
        max_tokens: int | None = None,
    ):
        self._device = device or _detect_device()
        torch_dtype = self.DTYPE_MAP.get(dtype, torch.float32)
        logger.info("Loading %s on %s (dtype=%s)", model, self._device, dtype)
        self._model = SentenceTransformer(
            model,
            device=self._device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"dtype": torch_dtype},
        )
        self._model_name = model
        self._batch_size = batch_size
        self._dimensions_val = self._model.get_embedding_dimension()
        # override the model's seq-length cap if user set one
        if max_tokens is not None:
            # clamp to the model's native max — exceeding it breaks the forward pass
            # (position-embedding table size is fixed at training time)
            # protect against user error here by capping it at the model's max if they set something too high
            self._model.max_seq_length = min(self._model.max_seq_length, max_tokens)
        self._max_tokens = self._model.max_seq_length
        self._encode_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int | None:
        return self._dimensions_val

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with self._encode_lock:
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return embeddings.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)
