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
    # Entry-level declaration only: a multimodal embedder entry maps PART
    # modalities to columns via `input_columns`, and each part column is decoded
    # with its own loader. There is deliberately no multimodal loader —
    # decode()/is_empty() reject it.
    MULTIMODAL = "multimodal"


_LOADERS = {
    Modality.TEXT: text,
    Modality.IMAGE: image,
}

# Modalities a multimodal entry's `input_columns` may map (i.e. the decodable ones).
PART_MODALITIES = frozenset(_LOADERS)


def _loader(modality: Modality):
    loader = _LOADERS.get(Modality(modality))
    if loader is None:
        raise ValueError(
            f"modality {Modality(modality).value!r} has no decoder: it is an "
            "entry-level declaration; decode each part column with its own "
            "part modality"
        )
    return loader


def decode(value: Any, modality: Modality) -> Any:
    """Decode one raw source value into the modality's canonical form."""
    return _loader(modality).decode(value)


def is_empty(value: Any, modality: Modality) -> bool:
    """True when there is nothing to embed in `value` for this modality."""
    return _loader(modality).is_empty(value)


__all__ = ["Modality", "PART_MODALITIES", "decode", "is_empty"]
