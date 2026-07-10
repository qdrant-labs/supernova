"""Text modality: canonical form is ``str``."""

from __future__ import annotations

from typing import Any


def decode(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    raise TypeError(
        f"text modality can't decode {type(value).__name__!r}; expected str or utf-8 bytes"
    )


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (bytes, bytearray)):
        return len(value) == 0
    return False
