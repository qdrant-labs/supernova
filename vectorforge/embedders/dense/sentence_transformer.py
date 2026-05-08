import asyncio
import logging
import threading

import torch
from sentence_transformers import SentenceTransformer

from vectorforge.embedders.dense.base import DenseEmbedder

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
        truncate: bool = False,
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
        self._dimensions_val = self._model.get_sentence_embedding_dimension()
        # override the model's seq-length cap if user set one
        if max_tokens is not None:
            # clamp to the model's native max — exceeding it breaks the forward pass
            # (position-embedding table size is fixed at training time)
            # protect against user error here by capping it at the model's max if they set something too high
            self._model.max_seq_length = min(self._model.max_seq_length, max_tokens)
        self._max_tokens = self._model.max_seq_length
        self._truncate = truncate
        self._encode_lock = threading.Lock()
        # Separate tokenizer copy for split_text to avoid "Already borrowed"
        # race with the model's internal tokenizer used during encode()
        from transformers import AutoTokenizer

        self._splitter_tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )

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
        # truncate mode: let the encoder's tokenizer chop overlong input; emit one piece.
        # useful for cutting long documents off to speed up inference
        # at the cost of potentially worse embeddings (losing semantic content after the cutoff)
        # if a dataset has many many long documents and is bottlenecked on encoding, this can be a useful speed/quality tradeoff to consider
        if self._truncate:
            return [text]

        tokens = self._splitter_tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self._max_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), self._max_tokens):
            chunk_tokens = tokens[i : i + self._max_tokens]
            chunks.append(
                self._splitter_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            )
        return chunks

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
