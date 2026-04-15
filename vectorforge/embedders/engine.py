"""
EmbeddingEngine -- orchestrates dense and/or sparse embedding generation.

Thin wrapper around one or two embedders, with some logic to optimize the hybrid case (same underlying model for both dense and sparse, e.g. gte-multilingual-base).

Sits between the pipeline worker and the embedder(s). The worker calls
engine.embed(texts) and gets back an EmbedResult with optional dense
and sparse embeddings. Handles the hybrid optimization internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vectorforge.embedders.dense.base import DenseEmbedder
from vectorforge.embedders.sparse.base import SparseEmbedder
from vectorforge.embedders.hybrid import SentenceTransformerHybridEmbedder
from vectorforge.models import SparseEmbedding

logger = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    """Result from EmbeddingEngine.embed() -- one or both fields will be set."""
    dense: list[list[float]] | None = None
    sparse: list[SparseEmbedding] | None = None


class EmbeddingEngine:
    """
    Wraps one or two embedders, handles hybrid optimization internally.

    The pipeline worker just calls engine.embed(texts) and gets back
    (dense, sparse). It never knows whether that was one forward pass
    or two.
    """

    def __init__(
        self,
        dense: DenseEmbedder | None = None,
        sparse: SparseEmbedder | None = None,
        hybrid: SentenceTransformerHybridEmbedder | None = None,
    ):
        if not dense and not sparse and not hybrid:
            raise ValueError("Must provide at least one of: dense_embedder, sparse_embedder")

        self.dense = dense
        self.sparse = sparse
        self._hybrid = hybrid

    @property
    def has_dense(self) -> bool:
        return self.dense is not None or self._hybrid is not None

    @property
    def has_sparse(self) -> bool:
        return self.sparse is not None or self._hybrid is not None

    @property
    def model_name(self) -> str:
        """Primary model name for manifests/logging."""
        if self._hybrid:
            return self._hybrid.model_name
        if self.dense:
            return self.dense.model_name
        return self.sparse.model_name

    @property
    def dense_model_name(self) -> str | None:
        if self._hybrid:
            return self._hybrid.model_name
        if self.dense:
            return self.dense.model_name
        return None

    @property
    def sparse_model_name(self) -> str | None:
        if self._hybrid:
            return self._hybrid.model_name
        if self.sparse:
            return self.sparse.model_name
        return None

    @property
    def dimensions(self) -> int | None:
        if self._hybrid:
            return self._hybrid.dimensions
        if self.dense:
            return self.dense.dimensions
        return None

    @property
    def max_tokens(self) -> int:
        if self._hybrid:
            return self._hybrid.max_tokens
        if self.dense:
            return self.dense.max_tokens
        return self.sparse.max_tokens

    def split_text(self, text: str) -> list[str]:
        """Use whichever embedder's tokenizer is available."""
        if self._hybrid:
            return self._hybrid.split_text(text)
        if self.dense:
            return self.dense.split_text(text)
        return self.sparse.split_text(text)

    async def embed(self, texts: list[str]) -> EmbedResult:
        if self._hybrid:
            dense, sparse = await self._hybrid.embed(texts)
            return EmbedResult(dense=dense, sparse=sparse)

        dense_out = None
        sparse_out = None

        if self.dense:
            dense_out = await self.dense.embed(texts)
        if self.sparse:
            sparse_out = await self.sparse.embed(texts)

        return EmbedResult(dense=dense_out, sparse=sparse_out)
