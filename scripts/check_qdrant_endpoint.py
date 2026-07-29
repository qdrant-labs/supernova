#!/usr/bin/env python3
"""
Simple Qdrant HTTPS checker.

Examples:
  python scripts/check_qdrant_endpoint.py
  python scripts/check_qdrant_endpoint.py --collection fineweb_gte_100m_probe
  python scripts/check_qdrant_endpoint.py --watch --interval 10
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_URL = "https://qdrant-fineweb-10b.mavcode.io"
DEFAULT_COLLECTION = "fineweb_gte_100m_probe"


def _default_url() -> str:
    # Prefer explicit HTTP URL when both endpoints exist.
    raw = os.getenv("QDRANT_HTTP_URL") or os.getenv("QDRANT_URL") or DEFAULT_URL
    if "-grpc." in raw:
        return raw.replace("-grpc.", ".")
    return raw


@dataclass
class CheckResult:
    ok: bool
    http_code: int | None
    elapsed_ms: int | None
    payload: dict[str, Any] | None
    error: str | None


def _request_json(url: str, api_key: str | None, timeout: float) -> CheckResult:
    headers = {}
    if api_key:
        headers["api-key"] = api_key
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    context = ssl.create_default_context()

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            body = resp.read().decode("utf-8")
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            payload = None
            if body:
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {"raw": body}
            return CheckResult(
                ok=(200 <= resp.status < 300),
                http_code=resp.status,
                elapsed_ms=elapsed_ms,
                payload=payload,
                error=None,
            )
    except urllib.error.HTTPError as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        body = e.read().decode("utf-8", errors="replace")
        return CheckResult(
            ok=False,
            http_code=e.code,
            elapsed_ms=elapsed_ms,
            payload={"raw": body} if body else None,
            error=f"HTTPError: {e}",
        )
    except Exception as e:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return CheckResult(
            ok=False,
            http_code=None,
            elapsed_ms=elapsed_ms,
            payload=None,
            error=f"{type(e).__name__}: {e}",
        )


def _print_snapshot(base: CheckResult, collection: CheckResult | None, collection_name: str) -> bool:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"\n[{now}]")
    print(
        f"base_endpoint: ok={base.ok} http={base.http_code} latency_ms={base.elapsed_ms}"
        + (f" error={base.error}" if base.error else "")
    )

    if collection is None:
        return base.ok

    points = None
    queue = None
    col_status = None
    status = None
    if collection.payload and isinstance(collection.payload, dict):
        status = collection.payload.get("status")
        result = collection.payload.get("result") or {}
        if isinstance(result, dict):
            points = result.get("points_count")
            col_status = result.get("status")
            update_queue = result.get("update_queue") or {}
            if isinstance(update_queue, dict):
                queue = update_queue.get("length")
    raw_snippet = None
    if collection.payload and isinstance(collection.payload, dict) and "raw" in collection.payload:
        raw_val = str(collection.payload.get("raw", ""))
        raw_snippet = raw_val.replace("\n", " ")[:140]

    print(
        f"collection[{collection_name}]: ok={collection.ok} http={collection.http_code} "
        f"latency_ms={collection.elapsed_ms} status={status} collection_status={col_status} "
        f"points={points} queue={queue}"
        + (f" error={collection.error}" if collection.error else "")
    )
    if raw_snippet:
        print(f"collection_raw_snippet: {raw_snippet}")
    return base.ok and collection.ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple Qdrant HTTPS endpoint checker.")
    parser.add_argument("--url", default=_default_url(), help="Qdrant base URL")
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
        help="Collection to inspect",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="Qdrant API key (or set QDRANT_API_KEY)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    parser.add_argument("--watch", action="store_true", help="Continuously poll endpoint")
    parser.add_argument("--interval", type=float, default=15.0, help="Watch interval seconds")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only check base endpoint, skip collection details",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    collection_url = f"{base_url}/collections/{args.collection}"
    overall_ok = True

    while True:
        base = _request_json(base_url + "/", api_key=None, timeout=args.timeout)
        collection = None
        if not args.base_only:
            collection = _request_json(collection_url, api_key=args.api_key, timeout=args.timeout)

        ok = _print_snapshot(base, collection, args.collection)
        overall_ok = overall_ok and ok

        if not args.watch:
            return 0 if ok else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())

