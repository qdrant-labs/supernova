"""Image modality: canonical form is a ``PIL.Image.Image``.

Accepted transport forms:

* ``PIL.Image.Image`` — passed through.
* ``bytes`` — decoded in-memory (png/jpeg/webp/…, whatever PIL reads).
* ``str`` — treated as a local file path.
* ``dict`` with ``bytes`` and/or ``path`` keys — the HuggingFace ``Image``
  feature's raw form when read straight from parquet; ``bytes`` wins when both
  are present (the path often refers to a file inside the original archive).
"""

from __future__ import annotations

import io
from typing import Any


def decode(value: Any):
    from PIL import Image  # heavyweight-ish; only needed for image pipelines

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, str):
        return Image.open(value)
    if isinstance(value, dict):
        data = value.get("bytes")
        if data:
            return Image.open(io.BytesIO(bytes(data)))
        path = value.get("path")
        if path:
            return Image.open(path)
        raise TypeError("image dict has neither 'bytes' nor 'path' set")
    raise TypeError(
        f"image modality can't decode {type(value).__name__!r}; expected "
        "PIL.Image, bytes, path str, or {'bytes':…, 'path':…} dict"
    )


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (bytes, bytearray, str)):
        return len(value) == 0
    if isinstance(value, dict):
        return not value.get("bytes") and not value.get("path")
    return False
