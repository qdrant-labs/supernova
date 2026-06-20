"""Text chunking strategies, owned independently of the embedding models.

A chunker turns one record's text into one or more pieces before embedding, so
every model in a pipeline sees the same pieces (see issue #12). Selected via the
config's ``chunking:`` block; omit it for the no-op default.
"""

from nova_embed.registry import CHUNKERS

from .base import Chunker

# Import the concrete chunkers so their @CHUNKERS.register decorators run.
from .passthrough import PassthroughChunker
from .fixed_char import FixedCharChunker
from .semantic import SemanticChunker


def build_chunker(cfg: dict | None) -> Chunker:
    """Build a :class:`Chunker` from a ``chunking:`` config block.

    Omitted / empty block → :class:`PassthroughChunker` (no-op), so splitting is
    always an explicit opt-in. ``strategy`` selects the class; remaining keys are
    passed through as constructor kwargs. An unknown strategy raises ``ValueError``.
    """
    return CHUNKERS.build(cfg)


__all__ = [
    "Chunker",
    "PassthroughChunker",
    "FixedCharChunker",
    "SemanticChunker",
    "build_chunker",
]
