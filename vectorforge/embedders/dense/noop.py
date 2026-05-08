"""
No-op dense embedder.

Lets the embed pipeline run end-to-end (source → chunker → writer) without
spending GPU time. Used by `vf partition` to validate sharding/partitioning
logic before committing to a real embed run.

The embed() return is `[None] * len(texts)` -- the worker translates that
into `dense_embedding=None` per record, and write_batch() then omits the
embedding column entirely. So the output parquet has every "real" column
(row_id, source_row_id, text, source columns, etc.) but no float vectors.
"""

from vectorforge.embedders.dense.base import DenseEmbedder


class NoopDenseEmbedder(DenseEmbedder):
    def __init__(
        self,
        max_tokens: int = 1_000_000,
        chunk_chars: int | None = None,
        model: str | None = None,
    ):
        """
        Args:
            max_tokens: reported as the embedder's max token length. Default is
                effectively "no chunking by token limit". Override to a sensible
                value if you want the chunker to actually split long texts during
                partition validation.
            chunk_chars: if set, split_text() splits texts into pieces of
                approximately this many characters. Cheap proxy for tokenizer-based
                chunking when you want the partition output to roughly match the
                eventual real embed run's row count.
            model: ignored, accepted so configs originally pointing at e.g.
                'BAAI/bge-large-en-v1.5' can be edited to type=noop without other
                changes.
        """
        self._max_tokens = max_tokens
        self._chunk_chars = chunk_chars
        self._model = model

    @property
    def model_name(self) -> str:
        return f"noop({self._model})" if self._model else "noop"

    @property
    def dimensions(self) -> int | None:
        return None

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def split_text(self, text: str) -> list[str]:
        if self._chunk_chars is None:
            return [text]
        n = self._chunk_chars
        return [text[i : i + n] for i in range(0, len(text), n)] or [""]

    async def embed(self, texts: list[str]):
        # Returning a list of None per input -- the worker sees a truthy list
        # (so it indexes into it) and ends up with .dense_embedding=None per
        # record, which the writer omits from the parquet schema.
        return [None] * len(texts)
