import asyncio
import logging

from vectorforge.embedders.base import Embedder

logger = logging.getLogger(__name__)

class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str = "fastembed-english-v1.0", batch_size: int = 128, max_concurrent: int = 2):
        import fastembed
        
        self.client = fastembed.AsyncClient()
        self._model = model
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        async def _embed_batch(batch: list[str]) -> list[list[float]]:
            async with self._semaphore:
                response = await self.client.embed(
                    texts=batch,
                    model=self._model,
                )
                return response.embeddings

        results = await asyncio.gather(*[_embed_batch(batch) for batch in batches])
        return [embedding for batch in results for embedding in batch]