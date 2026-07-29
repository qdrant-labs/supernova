#!/usr/bin/env python3
"""
Build recall-analysis ID maps from (filename, row_number) pairs.

Input formats:
  - CSV/TSV with headers (default expected columns: filename,row_number)
  - JSONL with keys (default expected keys: filename,row_number)

Output:
  - CSV or JSONL including computed point_id
  - Optional Qdrant verification columns when --verify is set

Examples:
  python scripts/build_recall_id_map.py \
    --input pairs.csv \
    --output recall_map.csv

  uv run --directory python/nova-sweep python /home/mvasquez/workspace/supernova/scripts/build_recall_id_map.py \
    --input pairs.csv \
    --output recall_map.jsonl \
    --output-format jsonl \
    --verify \
    --collection fineweb_gte_100m_probe
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any, Iterable


def normalize_http_url(raw: str) -> str:
    url = raw.strip()
    if "-grpc." in url:
        url = url.replace("-grpc.", ".")
    return url.rstrip("/")


def vf_point_id(filename: str, row_number: int) -> str:
    digest = hashlib.md5(f"{filename}:{row_number}".encode("utf-8")).hexdigest()  # noqa: S324
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build recall ID map from filename+row pairs.")
    p.add_argument("--input", required=True, help="Input CSV/TSV/JSONL path")
    p.add_argument("--output", required=True, help="Output CSV/JSONL path")
    p.add_argument(
        "--input-format",
        choices=("csv", "tsv", "jsonl"),
        help="Input format (default: infer from extension)",
    )
    p.add_argument(
        "--output-format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Output format",
    )
    p.add_argument("--filename-col", default="filename", help="Filename column/key")
    p.add_argument("--row-col", default="row_number", help="Row-number column/key")
    p.add_argument(
        "--query-id-col",
        default=None,
        help="Optional input query ID column/key to preserve in output",
    )
    p.add_argument("--verify", action="store_true", help="Verify IDs in Qdrant")
    p.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "fineweb_gte_100m_probe"),
        help="Qdrant collection when --verify",
    )
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
        help="Qdrant API key",
    )
    p.add_argument("--timeout", type=float, default=20.0, help="Qdrant timeout")
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for Qdrant retrieve when --verify",
    )
    return p.parse_args()


def infer_input_format(path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".tsv"):
        return "tsv"
    return "csv"


def read_rows(
    path: str,
    input_format: str,
    filename_col: str,
    row_col: str,
    query_id_col: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if input_format in ("csv", "tsv"):
        delimiter = "," if input_format == "csv" else "\t"
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for idx, row in enumerate(reader, start=1):
                try:
                    filename = str(row[filename_col])
                    row_number = int(row[row_col])
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"invalid row {idx}: {exc}") from exc
                rec: dict[str, Any] = {
                    "filename": filename,
                    "row_number": row_number,
                }
                if query_id_col:
                    rec["query_id"] = row.get(query_id_col)
                out.append(rec)
        return out

    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                filename = str(obj[filename_col])
                row_number = int(obj[row_col])
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid JSONL line {idx}: {exc}") from exc
            rec = {"filename": filename, "row_number": row_number}
            if query_id_col:
                rec["query_id"] = obj.get(query_id_col)
            out.append(rec)
    return out


def chunked(items: list[dict[str, Any]], n: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def verify_in_qdrant(
    rows: list[dict[str, Any]],
    collection: str,
    url: str,
    api_key: str | None,
    timeout: float,
    batch_size: int,
) -> None:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "qdrant-client is required for --verify. "
            "Run with uv in an environment that has qdrant-client."
        ) from exc

    client = QdrantClient(
        url=normalize_http_url(url),
        api_key=api_key,
        prefer_grpc=False,
        timeout=timeout,
    )
    found_ids: set[str] = set()
    for part in chunked(rows, batch_size):
        ids = [r["point_id"] for r in part]
        recs = client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        for rec in recs:
            found_ids.add(str(rec.id))

    for r in rows:
        r["found"] = r["point_id"] in found_ids


def write_output(path: str, fmt: str, rows: list[dict[str, Any]]) -> None:
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return

    # csv
    fields: list[str] = []
    for k in ("query_id", "filename", "row_number", "point_id", "found"):
        if any(k in r for r in rows):
            fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def main() -> int:
    args = parse_args()
    in_fmt = infer_input_format(args.input, args.input_format)

    rows = read_rows(
        path=args.input,
        input_format=in_fmt,
        filename_col=args.filename_col,
        row_col=args.row_col,
        query_id_col=args.query_id_col,
    )
    if not rows:
        print("no rows found in input", file=sys.stderr)
        return 1

    for r in rows:
        r["point_id"] = vf_point_id(r["filename"], int(r["row_number"]))

    if args.verify:
        verify_in_qdrant(
            rows=rows,
            collection=args.collection,
            url=args.url,
            api_key=args.api_key,
            timeout=args.timeout,
            batch_size=args.batch_size,
        )

    write_output(args.output, args.output_format, rows)

    print(f"rows={len(rows)}")
    print(f"output={args.output}")
    if args.verify:
        found = sum(1 for r in rows if r.get("found"))
        print(f"verified_found={found}")
        print(f"verified_missing={len(rows) - found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

