"""
Import a pre-embedded HuggingFace dataset into vectorforge's S3 format.

Streams parquet data from HuggingFace via DuckDB COPY, remaps columns to match
the vectorforge schema, and uploads batch_*.parquet files + _manifest.json.

Uses pure DuckDB SQL for all transformations (no Python row iteration) and
processes multiple language configs in parallel.

Usage:
    PYTHONPATH=. uv run python scripts/import_cohere_to_s3.py \
        --dataset CohereLabs/wikipedia-2023-11-embed-multilingual-v3 \
        --prefix cohere--wikipedia/embed-multilingual-v3 \
        --batch-size 100000 \
        --workers 4
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock

import duckdb

from vectorforge.storage.s3 import S3Backend

# Shared state for batch ID assignment across threads
_batch_lock = Lock()
_next_batch_id = 0


def _allocate_batch_id() -> int:
    global _next_batch_id
    with _batch_lock:
        bid = _next_batch_id
        _next_batch_id += 1
        return bid


def get_configs(dataset_name: str) -> list[str]:
    from datasets import get_dataset_config_names
    return get_dataset_config_names(dataset_name)


def process_config(
    dataset_name: str,
    config: str,
    batch_size: int,
    model_name: str,
    storage: S3Backend,
    output_dir: str,
) -> dict:
    """Stream one config from HF using DuckDB COPY, upload batches to S3."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    hf_path = f"hf://datasets/{dataset_name}/{config}/*.parquet"

    total_rows = con.execute(f"SELECT count(*) FROM '{hf_path}'").fetchone()[0]
    print(f"  [{config}] {total_rows:,} rows")

    records_written = 0
    batches = []
    offset = 0

    while offset < total_rows:
        t_batch = time.time()
        batch_id = _allocate_batch_id()
        limit = min(batch_size, total_rows - offset)

        filename = f"batch_{batch_id:08d}.parquet"
        local_path = os.path.join(output_dir, filename)

        # Pure DuckDB: transform + write parquet in one shot
        con.execute(f"""
            COPY (
                SELECT
                    (row_number() OVER () - 1 + {offset})::BIGINT   AS row_id,
                    (row_number() OVER () - 1 + {offset})::BIGINT   AS source_row_id,
                    {batch_id}::INT                                 AS chunk_id,
                    0::INT                                          AS chunk_index,
                    text,
                    '{dataset_name}'                                AS source,
                    emb                                             AS embedding,
                    '{model_name}'                                  AS model,
                    json_object(
                        '_id', _id,
                        'url', url,
                        'title', title,
                        'lang', '{config}'
                    )                                               AS payload
                FROM '{hf_path}'
                LIMIT {limit} OFFSET {offset}
            ) TO '{local_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

        # Upload and clean up
        asyncio.run(storage.upload_file(local_path))
        file_size_mb = round(os.path.getsize(local_path) / 1024 / 1024, 1)
        os.remove(local_path)

        elapsed = round(time.time() - t_batch, 1)
        records_written += limit
        print(f"    [{config}] batch_{batch_id:08d}: {limit:,} records, {file_size_mb}MB, {elapsed}s")

        batches.append({
            "batch_id": batch_id,
            "num_records": limit,
            "elapsed": elapsed,
        })

        offset += limit

    con.close()

    return {
        "config": config,
        "total_rows": total_rows,
        "records_written": records_written,
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Import pre-embedded HF dataset to S3")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--bucket", default="qdrant---vectorforge")
    parser.add_argument("--prefix", required=True, help="S3 prefix")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--model-name", default="embed-multilingual-v3.0",
                        help="Model name to record in parquet")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel config workers")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Specific configs to process (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Discover configs
    configs = args.configs if args.configs else get_configs(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Configs: {len(configs)} languages")

    storage = S3Backend(args.bucket, args.prefix)

    print("\n" + "=" * 60)
    print("vectorforge pre-embedded import plan")
    print("=" * 60)
    print(f"  Dataset:      {args.dataset}")
    print(f"  Configs:      {len(configs)} languages")
    print(f"  Batch size:   {args.batch_size:,}")
    print(f"  Workers:      {args.workers}")
    print(f"  Model:        {args.model_name}")
    print(f"  Destination:  s3://{args.bucket}/{args.prefix}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry run] No jobs submitted.")
        return

    output_dir = tempfile.mkdtemp(prefix="vectorforge_import_")
    print(f"Temp dir: {output_dir}")
    t0 = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_config,
                dataset_name=args.dataset,
                config=cfg,
                batch_size=args.batch_size,
                model_name=args.model_name,
                storage=storage,
                output_dir=output_dir,
            ): cfg
            for cfg in configs
        }

        for future in as_completed(futures):
            cfg = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"\n  Completed {cfg}: {result['records_written']:,} records")
            except Exception as e:
                print(f"\n  FAILED {cfg}: {e}")

    total_time = round(time.time() - t0, 1)
    total_records = sum(r["records_written"] for r in all_results)
    total_rows = sum(r["total_rows"] for r in all_results)
    total_batches = sum(len(r["batches"]) for r in all_results)

    # Upload manifest
    manifest = {
        "dataset": args.dataset,
        "split": "train",
        "total_rows": total_rows,
        "total_records": total_records,
        "chunk_size": args.batch_size,
        "num_slices": total_batches,
        "gpu": False,
        "embedder": {
            "type": "cohere",
            "model": args.model_name,
        },
        "configs": {r["config"]: {
            "total_rows": r["total_rows"],
            "records_written": r["records_written"],
        } for r in all_results},
        "slices": [
            batch
            for r in all_results
            for batch in r["batches"]
        ],
        "total_time_seconds": total_time,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    asyncio.run(
        storage.upload_bytes(
            json.dumps(manifest, indent=2).encode(),
            "_manifest.json",
        )
    )

    print("\n" + "=" * 60)
    print("import complete")
    print("=" * 60)
    print(f"  Total records:   {total_records:,}")
    print(f"  Total batches:   {total_batches}")
    print(f"  Total time:      {total_time}s")
    print("  Manifest:        _manifest.json uploaded")
    print("=" * 60)

    os.rmdir(output_dir)


if __name__ == "__main__":
    main()
