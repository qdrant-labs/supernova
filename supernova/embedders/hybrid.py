"""
Hybrid embedder -- produces both dense and sparse embeddings in a single forward pass.

This is an internal class, never exposed in config. The EmbeddingEngine creates it
automatically when dense_embedder and sparse_embedder point to the same model.
"""

import asyncio
import logging

import torch
from sentence_transformers import SentenceTransformer, SparseEncoder

from supernova.models import SparseEmbedding

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerHybridEmbedder:
    """
    Single model that produces both dense and sparse embeddings in one forward pass.

    Uses sentence-transformers SentenceTransformer for dense and SparseEncoder for
    sparse. When the underlying model supports both (e.g. gte-multilingual-base),
    the SparseEncoder shares the transformer backbone, avoiding redundant computation.
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
        torch_dtype = self.DTYPE_MAP.get(dtype, torch.float32)
        logger.info(
            "Loading hybrid encoder %s on %s (dtype=%s)", model, self._device, dtype
        )

        self._dense_model = SentenceTransformer(
            model,
            device=self._device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"dtype": torch_dtype},
        )
        self._sparse_model = SparseEncoder(
            model,
            device=self._device,
            trust_remote_code=trust_remote_code,
        )

        self._model_name = model
        self._batch_size = batch_size
        self._dimensions_val = self._dense_model.get_sentence_embedding_dimension()
        self._max_tokens = self._dense_model.max_seq_length

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

    def _encode(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseEmbedding]]:
        # Dense embeddings
        dense_np = self._dense_model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        dense = dense_np.tolist()

        # Sparse embeddings
        sparse_raw = self._sparse_model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        sparse = []
        for row in sparse_raw:
            if hasattr(row, "toarray"):
                arr = row.toarray().squeeze()
                nonzero = arr.nonzero()[0]
                sparse.append(
                    SparseEmbedding(
                        indices=nonzero.tolist(),
                        values=arr[nonzero].tolist(),
                    )
                )
            elif isinstance(row, dict):
                sparse.append(
                    SparseEmbedding(
                        indices=list(row.keys()),
                        values=list(row.values()),
                    )
                )
            else:
                raise TypeError(f"Unexpected sparse output type: {type(row)}")

        return dense, sparse

    async def embed(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseEmbedding]]:
        return await asyncio.to_thread(self._encode, texts)
