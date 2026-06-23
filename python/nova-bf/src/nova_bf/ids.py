"""
Point-id derivation, kept byte-identical to nova-load's `vf_point_id` macro.

The loader builds ids in DuckDB as:
    vf_uuid_from_hex(md5(filename || ':' || row))
i.e. md5 of "filename:row" as hex, formatted as a UUID. We replicate it here so
brute-force hit ids equal the Qdrant point ids the loader produced — letting you
evaluate recall by comparing id sets directly.
"""

from __future__ import annotations

import hashlib


def make_point_id(filename: str, row: int) -> str:
    """
    UUID derived from md5('{filename}:{row}'), matching `vf_point_id`.
    """
    h = hashlib.md5(f"{filename}:{row}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
