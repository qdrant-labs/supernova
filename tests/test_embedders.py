import pytest

from vectorforge.embedders.base import Embedder


class FakeEmbedder(Embedder):
    """Returns deterministic embeddings for testing."""

    @property
    def model_name(self) -> str:
        return "fake-model"

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


@pytest.mark.asyncio
async def test_fake_embedder():
    embedder = FakeEmbedder()
    result = await embedder.embed(["hello", "hi"])
    assert len(result) == 2
    assert result[0] == [5.0, 0.0, 1.0]
    assert result[1] == [2.0, 0.0, 1.0]


def test_embedder_properties():
    embedder = FakeEmbedder()
    assert embedder.model_name == "fake-model"
    assert embedder.dimensions == 3


def test_base_embedder_requires_max_tokens():
    """Base class raises NotImplementedError for max_tokens."""

    class MinimalEmbedder(Embedder):
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