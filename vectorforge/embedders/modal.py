from vectorforge.embedders.base import Embedder


class ModalEmbedder(Embedder):
    def __init__(self, app_name: str, function_name: str, model_id: str):
        import modal
        
        self.fn = modal.Function.lookup(app_name, function_name)
        self._model_name = model_id

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.fn.remote.aio(texts)
