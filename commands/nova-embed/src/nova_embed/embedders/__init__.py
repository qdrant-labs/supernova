# Import the backends package for its registration side-effects: importing
# `nova_embed.embedders` populates the EMBEDDERS registry with every backend
# whose dependencies are installed. Concrete classes are importable from their
# backend modules (nova_embed.embedders.backends.*) directly.
import nova_embed.embedders.backends  # noqa: F401
from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.embedders.engine import EmbeddingEngine, build_engine
from nova_embed.embedders.runner import run_embedder

__all__ = [
    "Embedder",
    "OutputKind",
    "EmbeddingEngine",
    "build_engine",
    "run_embedder",
]
