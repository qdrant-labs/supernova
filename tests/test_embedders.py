import pytest

from supernova.embedders.dense.base import DenseEmbedder
from supernova.embedders.sparse.base import SparseEmbedder
from supernova.embedders.engine import EmbeddingEngine
from supernova.models import SparseEmbedding


class FakeDenseEmbedder(DenseEmbedder):
    """Returns deterministic dense embeddings for testing."""

    @property
    def model_name(self) -> str:
        return "fake-dense"

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def max_tokens(self) -> int:
        return 100

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class FakeSparseEmbedder(SparseEmbedder):
    """Returns deterministic sparse embeddings for testing."""

    @property
    def model_name(self) -> str:
        return "fake-sparse"

    @property
    def max_tokens(self) -> int:
        return 100

    async def embed(self, texts: list[str]) -> list[SparseEmbedding]:
        return [SparseEmbedding(indices=[0, 1], values=[1.0, 0.5]) for _ in texts]


@pytest.mark.asyncio
async def test_dense_embedder():
    embedder = FakeDenseEmbedder()
    result = await embedder.embed(["hello", "hi"])
    assert len(result) == 2
    assert result[0] == [5.0, 0.0, 1.0]
    assert result[1] == [2.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_sparse_embedder():
    embedder = FakeSparseEmbedder()
    result = await embedder.embed(["hello", "hi"])
    assert len(result) == 2
    assert result[0].indices == [0, 1]
    assert result[0].values == [1.0, 0.5]


def test_dense_embedder_properties():
    embedder = FakeDenseEmbedder()
    assert embedder.model_name == "fake-dense"
    assert embedder.dimensions == 3


def test_base_dense_embedder_requires_max_tokens():
    # A minimal embedder need only implement model_name + embed. dimensions
    # defaults to None (inferred from the first batch); max_tokens must be
    # overridden. Text splitting is NOT an embedder concern — it lives in the
    # chunkers module (issue #12), so there's no split_text to define.
    class MinimalEmbedder(DenseEmbedder):
        @property
        def model_name(self) -> str:
            return "minimal"

        async def embed(self, texts):
            return []

    e = MinimalEmbedder()
    assert e.dimensions is None
    with pytest.raises(NotImplementedError):
        e.max_tokens


@pytest.mark.asyncio
async def test_engine_dense_only():
    engine = EmbeddingEngine(dense=FakeDenseEmbedder())
    assert engine.has_dense
    assert not engine.has_sparse

    result = await engine.embed(["hello"])
    assert result.dense == [[5.0, 0.0, 1.0]]
    assert result.sparse is None


@pytest.mark.asyncio
async def test_engine_sparse_only():
    engine = EmbeddingEngine(sparse=FakeSparseEmbedder())
    assert not engine.has_dense
    assert engine.has_sparse

    result = await engine.embed(["hello"])
    assert result.dense is None
    assert result.sparse[0].indices == [0, 1]


@pytest.mark.asyncio
async def test_engine_both():
    engine = EmbeddingEngine(dense=FakeDenseEmbedder(), sparse=FakeSparseEmbedder())
    assert engine.has_dense
    assert engine.has_sparse

    result = await engine.embed(["hello"])
    assert result.dense == [[5.0, 0.0, 1.0]]
    assert result.sparse[0].indices == [0, 1]


def test_engine_requires_at_least_one():
    with pytest.raises(ValueError):
        EmbeddingEngine()
