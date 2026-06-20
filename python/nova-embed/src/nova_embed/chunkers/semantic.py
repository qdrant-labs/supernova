from nova_embed.registry import CHUNKERS

from .base import Chunker


@CHUNKERS.register("semantic")
class SemanticChunker(Chunker):
    """Splits on semantic boundaries rather than fixed windows.

    NOT IMPLEMENTED YET — tracked in issue #13. Registered so that
    ``chunking.strategy: semantic`` fails with a clear, actionable message
    instead of an "unknown strategy" error.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "SemanticChunker is not implemented yet (see issue #13). "
            "Use 'passthrough' or 'fixed_char' for now."
        )

    def chunk(self, text: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError
