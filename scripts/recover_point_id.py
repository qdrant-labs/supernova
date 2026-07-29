#!/usr/bin/env python3
"""
Recover Supernova/Qdrant point IDs from (filename, file_row_number).

This mirrors nova-load's DuckDB macro:
  vf_point_id(fname, rnum) = uuid(md5(fname || ':' || CAST(rnum AS VARCHAR)))

Examples:
  python scripts/recover_point_id.py --filename "resharded/00/train-part0__1695.parquet" --row-number 0

  python scripts/recover_point_id.py \
    --filename "resharded/00/train-part0__1695.parquet" \
    --row-number 0 \
    --verify \
    --collection fineweb_gte_100m_probe
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys


def normalize_http_url(raw: str) -> str:
    url = raw.strip()
    if "-grpc." in url:
        url = url.replace("-grpc.", ".")
    return url.rstrip("/")


def vf_point_id(filename: str, row_number: int) -> str:
    digest = hashlib.md5(f"{filename}:{row_number}".encode("utf-8")).hexdigest()  # noqa: S324
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover deterministic point IDs used by nova-load (vf_point_id)."
    )
    p.add_argument("--filename", required=True, help="Logical file key used by loader")
    p.add_argument("--row-number", type=int, required=True, help="DuckDB file_row_number")
    p.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "fineweb_gte_100m_probe"),
        help="Collection name for verification",
    )
    p.add_argument("--verify", action="store_true", help="Verify ID exists in Qdrant")
    p.add_argument(
        "--url",
        default=os.getenv("QDRANT_HTTP_URL")
        or os.getenv("QDRANT_URL")
        or "https://qdrant-fineweb-10b.mavcode.io",
        help="Qdrant URL (HTTP endpoint)",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="Qdrant API key (or set QDRANT_API_KEY)",
    )
    p.add_argument("--timeout", type=float, default=20.0, help="Qdrant client timeout")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    point_id = vf_point_id(args.filename, args.row_number)

    print("filename=", args.filename)
    print("row_number=", args.row_number)
    print("point_id=", point_id)

    if not args.verify:
        return 0

    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # noqa: BLE001
        print(f"verify_error=missing_qdrant_client ({exc})", file=sys.stderr)
        return 2

    url = normalize_http_url(args.url)
    client = QdrantClient(
        url=url,
        api_key=args.api_key,
        prefer_grpc=False,
        timeout=args.timeout,
    )

    records = client.retrieve(
        collection_name=args.collection,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )
    found = len(records) > 0

    print("verify_url=", url)
    print("verify_collection=", args.collection)
    print("found=", found)
    if found:
        rec = records[0]
        payload_keys = sorted((rec.payload or {}).keys())
        print("payload_keys=", ",".join(payload_keys))

    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())

