import httpx

from vectorforge.embedders.base import Embedder


class BasetenEmbedder(Embedder):
    def __init__(self, deployment_id: str, api_key: str, model_id: str):
        self.deployment_id = deployment_id
        self.api_key = api_key
        self._model_name = model_id

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://model-{self.deployment_id}.api.baseten.co/predict",
                headers={"Authorization": f"Api-Key {self.api_key}"},
                json={"inputs": texts},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["embeddings"]
