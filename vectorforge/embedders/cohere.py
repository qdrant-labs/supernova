from vectorforge.embedders.base import Embedder


class CohereEmbedder(Embedder):
    """
    Note: Cohere has an input_type param that affects quality.
    Use "search_document" when embedding corpus data (default here).
    Use "search_query" when embedding queries at search time.
    """

    def __init__(self, model: str = "embed-english-v3.0", input_type: str = "search_document"):
        import cohere
        
        self.client = cohere.AsyncClient()
        self._model = model
        self.input_type = input_type

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embed(
            texts=texts,
            model=self._model,
            input_type=self.input_type,
        )
        return response.embeddings
