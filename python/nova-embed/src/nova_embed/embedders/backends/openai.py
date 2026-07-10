import asyncio
import logging

from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)


@EMBEDDERS.register("openai")
class OpenAIEmbedder(Embedder):
    output_kind = OutputKind.DENSE

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 128,
        max_concurrent: int = 2,
        max_retries: int = 5,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        from openai import AsyncOpenAI

        # For local servers (llama.cpp, vLLM, Ollama) that don't need auth,
        # pass api_key="none" in config to skip the OPENAI_API_KEY env var check.
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or None)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def max_tokens(self) -> int:
        return 8192

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import RateLimitError

        kwargs_base = {"model": self._model}
        if self._dimensions:
            kwargs_base["dimensions"] = self._dimensions

        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        async def _embed_batch(batch: list[str]) -> list[list[float]]:
            async with self._semaphore:
                for attempt in range(self._max_retries):
                    try:
                        response = await self.client.embeddings.create(
                            input=batch, **kwargs_base
                        )
                        return [item.embedding for item in response.data]
                    except RateLimitError as e:
                        if attempt == self._max_retries - 1:
                            raise
                        wait = 2**attempt
                        retry_after = getattr(e.response, "headers", {}).get(
                            "retry-after"
                        )
                        if retry_after:
                            wait = float(retry_after)
                        logger.warning(
                            "Rate limited, retrying in %.1fs (attempt %d/%d)",
                            wait,
                            attempt + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(wait)

        results = await asyncio.gather(*[_embed_batch(b) for b in batches])
        return [emb for batch_result in results for emb in batch_result]
