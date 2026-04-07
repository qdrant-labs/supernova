"""
Modal-parallelized import of CohereLabs/wikipedia pre-embedded dataset to S3.

Fans out one container per language config (~323 containers), each streaming
from HuggingFace via DuckDB and uploading batch parquet files to S3.

Usage:
  # Dry run
  modal run modal_import_cohere.py --dry-run

  # Full run (all configs)
  modal run modal_import_cohere.py

  # Specific configs only
  modal run modal_import_cohere.py --configs en de fr es

  # Fire-and-forget
  modal run --detach modal_import_cohere.py
"""

import modal

app = modal.App("vectorforge-import-cohere")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("duckdb", "pyarrow", "aiobotocore", "datasets")
    .add_local_dir("vectorforge", "/app/vectorforge")
)

DATASET = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
BUCKET = "qdrant--vectorforge"
PREFIX = "cohere--wikipedia/embed-multilingual-v3"
MODEL_NAME = "embed-multilingual-v3.0"
BATCH_SIZE = 100_000


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("vectorforge-secrets")],
    timeout=7200,
    retries=1,
)
def import_config(config: str) -> dict:
    """Import a single language config from HF to S3."""
    import asyncio
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/app")

    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vectorforge.storage.s3 import S3Backend

    SCHEMA = pa.schema([
        pa.field("row_id", pa.int64()),
        pa.field("source_row_id", pa.int64()),
        pa.field("chunk_id", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
        pa.field("model", pa.string()),
        pa.field("payload", pa.string()),
    ])

    storage = S3Backend(BUCKET, f"{PREFIX}/{config}")
    t0 = time.time()

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    hf_path = f"hf://datasets/{DATASET}/{config}/*.parquet"
    total_rows = con.execute(f"SELECT count(*) FROM '{hf_path}'").fetchone()[0]
    print(f"[{config}] {total_rows:,} rows")

    output_dir = "/tmp/vectorforge"
    os.makedirs(output_dir, exist_ok=True)

    batches = []
    offset = 0
    batch_idx = 0

    while offset < total_rows:
        t_batch = time.time()
        limit = min(BATCH_SIZE, total_rows - offset)

        filename = f"batch_{batch_idx:06d}.parquet"
        local_path = os.path.join(output_dir, filename)

        con.execute(f"""
            COPY (
                SELECT
                    (row_number() OVER () - 1 + {offset})::BIGINT  AS row_id,
                    (row_number() OVER () - 1 + {offset})::BIGINT  AS source_row_id,
                    {batch_idx}::INT                                AS chunk_id,
                    0::INT                                          AS chunk_index,
                    text,
                    '{DATASET}'                                     AS source,
                    emb                                             AS embedding,
                    '{MODEL_NAME}'                                  AS model,
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

        asyncio.run(storage.upload_file(local_path))
        file_mb = round(os.path.getsize(local_path) / 1024 / 1024, 1)
        os.remove(local_path)

        elapsed = round(time.time() - t_batch, 1)
        print(f"  [{config}] {filename}: {limit:,} records, {file_mb}MB, {elapsed}s")

        batches.append({
            "filename": filename,
            "num_records": limit,
            "elapsed": elapsed,
        })

        batch_idx += 1
        offset += limit

    con.close()
    total_elapsed = round(time.time() - t0, 1)

    return {
        "config": config,
        "total_rows": total_rows,
        "num_batches": len(batches),
        "batches": batches,
        "elapsed": total_elapsed,
    }


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    configs: str = "",
):
    import json
    import time
    import asyncio

    from datasets import get_dataset_config_names

    if configs:
        config_list = configs.split(",")
    else:
        config_list = get_dataset_config_names(DATASET)

    print(f"Dataset: {DATASET}")
    print(f"Configs: {len(config_list)} languages")
    print(f"Destination: s3://{BUCKET}/{PREFIX}")

    if dry_run:
        print("\n[dry run] Would process:")
        for c in config_list:
            print(f"  {c}")
        return

    print(f"\nSubmitting {len(config_list)} jobs...")
    t0 = time.time()

    results = []
    for result in import_config.map(config_list, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"  FAILED: {result}")
        else:
            results.append(result)
            print(f"  Completed {result['config']}: "
                  f"{result['total_rows']:,} records in {result['elapsed']}s")

    total_time = round(time.time() - t0, 1)
    total_records = sum(r["total_rows"] for r in results)
    total_batches = sum(r["num_batches"] for r in results)

    # Upload manifest
    from vectorforge.storage.s3 import S3Backend

    manifest = {
        "dataset": DATASET,
        "split": "train",
        "total_rows": total_records,
        "total_records": total_records,
        "chunk_size": BATCH_SIZE,
        "num_slices": total_batches,
        "gpu": False,
        "embedder": {
            "type": "cohere",
            "model": MODEL_NAME,
        },
        "configs": {r["config"]: {
            "total_rows": r["total_rows"],
            "num_batches": r["num_batches"],
            "elapsed": r["elapsed"],
        } for r in results},
        "total_time_seconds": total_time,
    }

    storage = S3Backend(BUCKET, PREFIX)
    asyncio.run(
        storage.upload_bytes(
            json.dumps(manifest, indent=2).encode(),
            "_manifest.json",
        )
    )

    print("\n" + "=" * 60)
    print("import complete")
    print("=" * 60)
    print(f"  Configs:       {len(results)}/{len(config_list)}")
    print(f"  Total records: {total_records:,}")
    print(f"  Total batches: {total_batches}")
    print(f"  Total time:    {total_time}s")
    print("=" * 60)
