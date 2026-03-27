from vectorforge.embedders.base import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small", dimensions: int | None = None):
        from openai import AsyncOpenAI
        
        self.client = AsyncOpenAI()
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self._model, "input": texts}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        response = await self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
