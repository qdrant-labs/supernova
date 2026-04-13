import pytest

from vectorforge.embedders.dense.base import DenseEmbedder
from vectorforge.embedders.sparse.base import SparseEmbedder
from vectorforge.embedders.engine import EmbeddingEngine, EmbedResult
from vectorforge.models import SparseEmbedding


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

    def split_text(self, text: str) -> list[str]:
        return [text]

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

    def split_text(self, text: str) -> list[str]:
        return [text]

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
    with pytest.raises(NotImplementedError):
        e.split_text("hello")


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
