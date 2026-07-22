"""
The single embedder interface.

There is one base class, not one per output kind — "dense vs sparse vs
multivector" is a *declared property* of an implementation (`output_kind`), not
a class hierarchy. The declaration is what downstream consumers use: the parquet
writer picks the column schema from it, the manifest records it, and the engine
routes hybrid fusion / pooling by it. Nothing ever sniffs the runtime shape of a
batch result.

Likewise `supported_modalities` declares what a backend can consume; the engine
validates each config entry's `modality` against it *before* loading weights, so
a wrong-modality launch dies immediately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from nova_embed.media import Modality
from nova_embed.models import Embedding, OutputKind

__all__ = ["Embedder", "OutputKind"]


class Embedder(ABC):
    """
    Abstract base for all embedding backends.

    All implementations must be async — parallelism comes from running many
    embed() calls concurrently via asyncio.gather.
    """

    # What one input becomes: list[float] / SparseEmbedding / MultiVectorEmbedding.
    output_kind: ClassVar[OutputKind]
    # What this backend can consume. The engine hands embed() *canonical* objects
    # for the configured modality (str for text, PIL.Image for image; multimodal
    # entries get part dicts like {"text": str, "image": PIL.Image} with at least
    # one key present) — transport decoding already happened in nova_embed.media.
    supported_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.TEXT})

    @abstractmethod
    async def embed(self, batch: list[Any]) -> list[Embedding]:
        """
        Takes a batch of canonical inputs. Returns one embedding per input,
        in the same order.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier, written into the manifest."""
        pass

    @property
    def dimensions(self) -> int | None:
        """
        Vector dimension (per-vector D for multivector). Override if known
        statically; None means "inferred from data / not applicable".
        """
        return None

    @property
    def max_tokens(self) -> int | None:
        """
        Max input token length, for text-modality backends. None for backends
        where the concept doesn't apply (e.g. image encoders). Written to the
        manifest; text splitting itself is owned by the chunkers module.
        """
        return None

    @property
    def instruction(self) -> str | None:
        """
        Instruction/prompt text baked into every embedding by this backend
        (e.g. the chat-template system turn of instruction-tuned embedding
        models). None when the concept doesn't apply. Written to the manifest
        because it CHANGES the embedding space: query-side embedding at search
        time must reproduce the exact same instruction, or recall quietly tanks.
        """
        return None
