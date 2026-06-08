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

import numpy as np

from supernova.embedders.dense.base import DenseEmbedder
from supernova.embedders.sparse.base import SparseEmbedder
from supernova.embedders.multivector.base import MultiVectorEmbedder
from supernova.embedders.hybrid import SentenceTransformerHybridEmbedder
from supernova.models import MultiVectorEmbedding, SparseEmbedding

logger = logging.getLogger(__name__)

POOLING_TYPES = {"mean", "max", "cls", "last"}


def pool_multivector(
    mv: MultiVectorEmbedding, pool_type: str, normalize: bool
) -> list[float]:
    arr = np.asarray(mv.vectors, dtype=np.float32)
    if pool_type == "mean":
        pooled = arr.mean(axis=0)
    elif pool_type == "max":
        pooled = arr.max(axis=0)
    elif pool_type == "cls":
        pooled = arr[0]
    elif pool_type == "last":
        pooled = arr[-1]
    else:
        raise ValueError(
            f"Unknown pooling type: {pool_type!r}. Choose from {sorted(POOLING_TYPES)}."
        )

    if normalize:
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm

    return pooled.tolist()


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
        multivector_pooling: str | None = None,
        multivector_pooling_normalize: bool = True,
    ):
        if not dense and not sparse and not multivector and not hybrid:
            raise ValueError(
                "Must provide at least one of: dense_embedder, sparse_embedder, multivector_embedder"
            )

        if multivector_pooling is not None:
            if multivector_pooling not in POOLING_TYPES:
                raise ValueError(
                    f"Unknown pooling type: {multivector_pooling!r}. Choose from {sorted(POOLING_TYPES)}."
                )
            if not multivector:
                raise ValueError("pooling requires multivector_embedder to be set")
            if dense or hybrid:
                raise ValueError(
                    "pooling produces a dense column from the multivector output; "
                    "it conflicts with a separately-configured dense_embedder. Pick one."
                )

        self.dense = dense
        self.sparse = sparse
        self.multivector = multivector
        self._hybrid = hybrid
        self._multivector_pooling = multivector_pooling
        self._multivector_pooling_normalize = multivector_pooling_normalize

    @property
    def has_dense(self) -> bool:
        return (
            self.dense is not None
            or self._hybrid is not None
            or self._multivector_pooling is not None
        )

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
        if self._multivector_pooling and self.multivector:
            return f"{self.multivector.model_name} ({self._multivector_pooling}-pooled)"
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
            # derive a pooled dense vector from each multivector output, if configured
            if self._multivector_pooling is not None:
                dense_out = [
                    pool_multivector(
                        mv,
                        self._multivector_pooling,
                        self._multivector_pooling_normalize,
                    )
                    for mv in multivector_out
                ]

        return EmbedResult(
            dense=dense_out, sparse=sparse_out, multivector=multivector_out
        )
