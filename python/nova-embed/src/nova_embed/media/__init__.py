"""Per-modality input handling: decoding raw source values into canonical objects.

The str-vs-bytes-vs-path question is *transport*, not modality — and it's solved
here exactly once, per modality, instead of inside every embedder backend. The
pipeline decodes each input column through these loaders, so embedders always
receive canonical objects (``str`` for text, ``PIL.Image`` for image) and never
re-implement the dispatch grid.

Each modality module exposes:

* ``decode(value) -> canonical`` — accept every transport form the wild throws
  at us (str, bytes, path, HF ``{"bytes":…, "path":…}`` dicts) or raise a clear
  TypeError.
* ``is_empty(value) -> bool`` — the modality's definition of "nothing to embed"
  (whitespace-only text, missing/empty image payload, …). Drives the pipeline's
  ``on_empty_input`` policy.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from nova_embed.media import image, text


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


_LOADERS = {
    Modality.TEXT: text,
    Modality.IMAGE: image,
}


def decode(value: Any, modality: Modality) -> Any:
    """Decode one raw source value into the modality's canonical form."""
    return _LOADERS[Modality(modality)].decode(value)


def is_empty(value: Any, modality: Modality) -> bool:
    """True when there is nothing to embed in `value` for this modality."""
    return _LOADERS[Modality(modality)].is_empty(value)


__all__ = ["Modality", "decode", "is_empty"]
