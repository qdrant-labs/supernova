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
from vectorforge.embedders.multivector.base import MultiVectorEmbedder
from vectorforge.embedders.hybrid import SentenceTransformerHybridEmbedder
from vectorforge.models import MultiVectorEmbedding, SparseEmbedding

logger = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    """Result from EmbeddingEngine.embed() -- any combination of fields may be set."""
    dense: list[list[float]] | None = None
    sparse: list[SparseEmbedding] | None = None
    multivector: list[MultiVectorEmbedding] | None = None


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
        multivector: MultiVectorEmbedder | None = None,
        hybrid: SentenceTransformerHybridEmbedder | None = None,
    ):
        if not dense and not sparse and not multivector and not hybrid:
            raise ValueError("Must provide at least one of: dense_embedder, sparse_embedder, multivector_embedder")

        self.dense = dense
        self.sparse = sparse
        self.multivector = multivector
        self._hybrid = hybrid

    @property
    def has_dense(self) -> bool:
        return self.dense is not None or self._hybrid is not None

    @property
    def has_sparse(self) -> bool:
        return self.sparse is not None or self._hybrid is not None

    @property
    def has_multivector(self) -> bool:
        return self.multivector is not None

    @property
    def multivector_model_name(self) -> str | None:
        return self.multivector.model_name if self.multivector else None

    @property
    def model_name(self) -> str:
        """Primary model name for manifests/logging."""
        if self._hybrid:
            return self._hybrid.model_name
        if self.dense:
            return self.dense.model_name
        if self.sparse:
            return self.sparse.model_name
        return self.multivector.model_name

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
        if self.multivector:
            return self.multivector.dimensions
        return None

    @property
    def max_tokens(self) -> int:
        if self._hybrid:
            return self._hybrid.max_tokens
        if self.dense:
            return self.dense.max_tokens
        if self.sparse:
            return self.sparse.max_tokens
        return self.multivector.max_tokens

    def split_text(self, text: str) -> list[str]:
        """Use whichever embedder's tokenizer is available."""
        if self._hybrid:
            return self._hybrid.split_text(text)
        if self.dense:
            return self.dense.split_text(text)
        if self.sparse:
            return self.sparse.split_text(text)
        return self.multivector.split_text(text)

    async def embed(self, texts: list[str]) -> EmbedResult:
        dense_out = None
        sparse_out = None
        multivector_out = None

        if self._hybrid:
            dense_out, sparse_out = await self._hybrid.embed(texts)
        else:
            if self.dense:
                dense_out = await self.dense.embed(texts)
            if self.sparse:
                sparse_out = await self.sparse.embed(texts)

        if self.multivector:
            multivector_out = await self.multivector.embed(texts)

        return EmbedResult(dense=dense_out, sparse=sparse_out, multivector=multivector_out)
