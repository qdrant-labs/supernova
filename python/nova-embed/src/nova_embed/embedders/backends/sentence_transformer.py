"""
sentence-transformers backends: dense (text + image) and sparse.

No fused backend here, deliberately: sentence-transformers splits dense and
sparse into separate classes (SentenceTransformer / SparseEncoder) with no
shared-backbone API, so a "fused" implementation would still load two models
and run two forward passes — module surgery on ST internals to actually share
the transformer is too version-fragile to be worth it. Models that natively
multi-head (e.g. bge-m3) get real fusion via their own backend.
"""

import asyncio
import logging
import threading

from typing import Any

import torch

from sentence_transformers import SentenceTransformer, SparseEncoder

from nova_embed.embedders.backends.device import detect_device
from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.media import Modality
from nova_embed.models import SparseEmbedding
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _sparse_rows_to_embeddings(rows) -> list[SparseEmbedding]:
    """
    SparseEncoder returns scipy sparse matrices or dicts depending on version.
    """
    embeddings = []
    for row in rows:
        if hasattr(row, "toarray"):
            arr = row.toarray().squeeze()
            nonzero = arr.nonzero()[0]
            embeddings.append(
                SparseEmbedding(indices=nonzero.tolist(), values=arr[nonzero].tolist())
            )
        elif isinstance(row, dict):
            embeddings.append(
                SparseEmbedding(indices=list(row.keys()), values=list(row.values()))
            )
        else:
            raise TypeError(f"Unexpected sparse output type: {type(row)}")
    return embeddings


@EMBEDDERS.register("sentence_transformer")
class SentenceTransformerDenseEmbedder(Embedder):
    output_kind = OutputKind.DENSE
    # ST's encode() accepts PIL images for CLIP-family models (clip-ViT-*,
    # jina-clip, siglip ports, …), so one class serves both modalities. A
    # text-only model fed images fails loudly in the forward pass.
    supported_modalities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(
        self,
        model: str = "Alibaba-NLP/gte-multilingual-base",
        batch_size: int = 32,
        device: str | None = None,
        dtype: str = "float32",
        trust_remote_code: bool = False,
        max_tokens: int | None = None,
        # Pin the Hub revision (a commit sha, tag or branch)
        revision: str | None = None,
    ):
        self._device = device or detect_device()
        torch_dtype = DTYPE_MAP.get(dtype, torch.float32)
        logger.info(
            "Loading %s on %s (dtype=%s, revision=%s)",
            model, self._device, dtype, revision or "default",
        )
        self._model = SentenceTransformer(
            model,
            device=self._device,
            trust_remote_code=trust_remote_code,
            revision=revision,
            model_kwargs={"dtype": torch_dtype},
        )
        self._model_name = model
        self._batch_size = batch_size
        self._dimensions_val = self._model.get_embedding_dimension()
        # override the model's seq-length cap if user set one
        if max_tokens is not None:
            # clamp to the model's native max — exceeding it breaks the forward pass
            # (position-embedding table size is fixed at training time)
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

    def _encode(self, batch: list[Any]) -> list[list[float]]:
        with self._encode_lock:
            embeddings = self._model.encode(
                batch,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        # keep rows as float32 ndarray views, NOT .tolist(): a Python float is
        # ~32B vs 4B — on a 2048-dim model that's 64KB vs 8KB per row sitting
        # in the flush buffer. pyarrow writes ndarray rows directly.
        return list(embeddings)

    async def embed(self, batch: list[Any]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, batch)


@EMBEDDERS.register("sentence_transformer")
class SentenceTransformerSparseEmbedder(Embedder):
    """
    Sparse embedder using sentence-transformers SparseEncoder.
    Works with models like SPLADE, gte-multilingual-base (sparse mode), etc.
    """

    output_kind = OutputKind.SPARSE

    def __init__(
        self,
        model: str = "Alibaba-NLP/gte-multilingual-base",
        batch_size: int = 32,
        device: str | None = None,
        dtype: str = "float32",
        trust_remote_code: bool = False,
        revision: str | None = None,  # see the dense embedder's `revision`
    ):
        self._device = device or detect_device()
        logger.info(
            "Loading sparse encoder %s on %s (dtype=%s, revision=%s)",
            model, self._device, dtype, revision or "default",
        )
        self._model = SparseEncoder(
            model,
            device=self._device,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        self._model_name = model
        self._batch_size = batch_size
        self._max_tokens = self._model.max_seq_length
        # serialize concurrent pipeline workers — same contract as the dense
        # embedder's lock; a torch forward pass is not thread-safe
        self._encode_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _encode(self, texts: list[str]) -> list[SparseEmbedding]:
        with self._encode_lock:
            results = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        return _sparse_rows_to_embeddings(results)

    async def embed(self, texts: list[str]) -> list[SparseEmbedding]:
        return await asyncio.to_thread(self._encode, texts)
