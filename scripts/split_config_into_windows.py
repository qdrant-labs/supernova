#!/usr/bin/env python3
"""
Split an embedder YAML config into N windows of a fixed row count.

Produces one config per window with `source.offset` / `source.limit` set
and `storage.s3_prefix` suffixed by the row range so each window writes
to a distinct S3 destination.

Usage:
  python scripts/split_config_into_windows.py \\
      configs/embedder/standford_oval_ccnews_bge_large.yaml \\
      --window-size 100_000_000 --num-windows 10

Output files land next to the template:
  configs/embedder/standford_oval_ccnews_bge_large.rows-0000000000-0100000000.yaml
  configs/embedder/standford_oval_ccnews_bge_large.rows-0100000000-0200000000.yaml
  ...
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import yaml


def format_range(offset: int, limit: int, width: int = 10) -> str:
    end = offset + limit
    return f"rows-{offset:0{width}d}-{end:0{width}d}"


def main():
    parser = argparse.ArgumentParser(description="Split an embedder YAML into N row-window configs.")
    parser.add_argument("template", type=Path, help="Path to a template embedder YAML.")
    parser.add_argument("--window-size", type=int, required=True,
                        help="Rows per window, e.g. 100_000_000.")
    parser.add_argument("--num-windows", type=int, required=True,
                        help="Number of windows to generate.")
    parser.add_argument("--start-offset", type=int, default=0,
                        help="Starting offset for window 0. Defaults to 0.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated, don't write files.")
    args = parser.parse_args()

    if not args.template.exists():
        print(f"Template not found: {args.template}", file=sys.stderr)
        sys.exit(1)

    with open(args.template) as f:
        template = yaml.safe_load(f)

    original_s3_prefix = template.get("storage", {}).get("s3_prefix", "")
    if not original_s3_prefix:
        print("Template's storage.s3_prefix is empty; can't derive per-window prefixes.", file=sys.stderr)
        sys.exit(1)

    stem = args.template.stem
    out_dir = args.template.parent

    print(f"Template: {args.template}")
    print(f"Windows: {args.num_windows} × {args.window_size:,} rows starting at offset {args.start_offset:,}")
    print()

    for i in range(args.num_windows):
        offset = args.start_offset + i * args.window_size
        limit = args.window_size
        rng = format_range(offset, limit)

        # deep-ish copy via YAML round-trip so we don't mutate the template dict
        cfg = yaml.safe_load(yaml.safe_dump(template))
        cfg.setdefault("source", {})
        cfg["source"]["offset"] = offset
        cfg["source"]["limit"] = limit
        cfg.setdefault("storage", {})
        cfg["storage"]["s3_prefix"] = f"{original_s3_prefix}/{rng}"

        out_path = out_dir / f"{stem}.{rng}.yaml"
        print(f"  [{i+1:>3}/{args.num_windows}] {rng}  ->  {out_path}")

        if not args.dry_run:
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    if args.dry_run:
        print("\n[dry-run] No files written.")


if __name__ == "__main__":
    main()