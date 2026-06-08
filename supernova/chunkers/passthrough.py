from .base import Chunker


class PassthroughChunker(Chunker):
    """No-op chunker: emits the text unchanged as a single piece.

    The default strategy — splitting is opt-in. Text longer than a model's
    token limit is left for the model to truncate at embed time (lossy, one
    vector per row), rather than being fanned out into multiple vectors.
    """

    def chunk(self, text: str) -> list[str]:
        return [text]
