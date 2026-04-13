import asyncio
import logging

from vectorforge.embedders.dense.base import DenseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(DenseEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 128,
        max_concurrent: int = 2,
        max_retries: int = 5,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
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

    def split_text(self, text: str) -> list[str]:
        import tiktoken
        encoder = tiktoken.encoding_for_model(self._model)
        tokens = encoder.encode(text, allowed_special="all")

        if len(tokens) <= self.max_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunks.append(encoder.decode(chunk_tokens))
        return chunks

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
                        response = await self.client.embeddings.create(input=batch, **kwargs_base)
                        return [item.embedding for item in response.data]
                    except RateLimitError as e:
                        if attempt == self._max_retries - 1:
                            raise
                        wait = 2 ** attempt
                        retry_after = getattr(e.response, "headers", {}).get("retry-after")
                        if retry_after:
                            wait = float(retry_after)
                        logger.warning("Rate limited, retrying in %.1fs (attempt %d/%d)", wait, attempt + 1, self._max_retries)
                        await asyncio.sleep(wait)

        results = await asyncio.gather(*[_embed_batch(b) for b in batches])
        return [emb for batch_result in results for emb in batch_result]
