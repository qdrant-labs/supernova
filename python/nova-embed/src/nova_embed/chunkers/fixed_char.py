from nova_embed.registry import CHUNKERS

from .base import Chunker


@CHUNKERS.register("fixed_char")
class FixedCharChunker(Chunker):
    """Splits text into fixed-size character windows (characters, NOT tokens).

    ``chunk_chars`` is the window size; ``overlap`` repeats the trailing N
    characters at the start of the next window (0 = no overlap), which helps
    avoid cutting context exactly on a boundary. Each window becomes its own
    record and thus its own vector, so one long row fans out into several.

    Character-based splitting is model-agnostic and cheap, but approximate: a
    window's token count varies by tokenizer, so a window may still exceed a
    model's limit and get truncated. Use it when you want predictable, uniform
    pieces without paying for tokenization.
    """

    def __init__(self, chunk_chars: int = 1024, overlap: int = 0):
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be > 0")
        if not 0 <= overlap < chunk_chars:
            raise ValueError("overlap must be in [0, chunk_chars)")
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if len(text) <= self.chunk_chars:
            return [text]
        step = self.chunk_chars - self.overlap
        return [text[i : i + self.chunk_chars] for i in range(0, len(text), step)]
