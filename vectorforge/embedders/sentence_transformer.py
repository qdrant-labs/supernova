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
        torch_dtype = self.DTYPE_MAP.get(dtype, torch.float32)
        logger.info("Loading %s on %s (dtype=%s)", model, self._device, dtype)
        self._model = SentenceTransformer(
            model,
            device=self._device,
            trust_remote_code=trust_remote_code,
            # TODO torch_dtype is deprecated, use dtpye instead
            model_kwargs={"torch_dtype": torch_dtype},
        )
        self._model_name = model
        self._batch_size = batch_size
        self._dimensions_val = self._model.get_sentence_embedding_dimension()
        self._max_tokens = self._model.max_seq_length
        # Separate tokenizer copy for split_text to avoid "Already borrowed"
        # race with the model's internal tokenizer used during encode()
        from transformers import AutoTokenizer
        self._splitter_tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int | None:
        return self._dimensions_val

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def split_text(self, text: str) -> list[str]:
        """
        Split text using the model's own tokenizer.
        """
        tokens = self._splitter_tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self._max_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), self._max_tokens):
            chunk_tokens = tokens[i : i + self._max_tokens]
            chunks.append(self._splitter_tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        return chunks

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