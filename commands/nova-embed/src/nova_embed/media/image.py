"""Image modality: canonical form is a ``PIL.Image.Image``.

Accepted transport forms:

* ``PIL.Image.Image`` — passed through.
* ``bytes`` — decoded in-memory (png/jpeg/webp/…, whatever PIL reads).
* ``str`` — an ``http(s)://`` URL is fetched; anything else is a local file
  path. Fetch timeout defaults to 10s, override via ``NOVA_IMAGE_FETCH_TIMEOUT``
  (seconds). A failed fetch raises — the pipeline's empty-input policy is about
  MISSING data, not broken transport, and silently dropping rows on a flaky
  CDN would skew the output.
* ``dict`` with ``bytes`` and/or ``path`` keys — the HuggingFace ``Image``
  feature's raw form when read straight from parquet; ``bytes`` wins when both
  are present (the path often refers to a file inside the original archive).
"""

from __future__ import annotations

import io
import os
from typing import Any

_URL_PREFIXES = ("http://", "https://")


def _fetch_url(url: str):
    import urllib.request

    from PIL import Image

    timeout = float(os.environ.get("NOVA_IMAGE_FETCH_TIMEOUT", "10"))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:
        raise ValueError(f"failed to fetch image from {url!r}: {e}") from e
    return Image.open(io.BytesIO(data))


def _open_str(value: str):
    from PIL import Image

    if value.startswith(_URL_PREFIXES):
        return _fetch_url(value)
    return Image.open(value)


def decode(value: Any):
    from PIL import Image  # heavyweight-ish; only needed for image pipelines

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, str):
        return _open_str(value)
    if isinstance(value, dict):
        data = value.get("bytes")
        if data:
            return Image.open(io.BytesIO(bytes(data)))
        path = value.get("path")
        if path:
            return _open_str(path)
        raise TypeError("image dict has neither 'bytes' nor 'path' set")
    raise TypeError(
        f"image modality can't decode {type(value).__name__!r}; expected "
        "PIL.Image, bytes, path/URL str, or {'bytes':…, 'path':…} dict"
    )


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (bytes, bytearray, str)):
        return len(value) == 0
    if isinstance(value, dict):
        return not value.get("bytes") and not value.get("path")
    return False
