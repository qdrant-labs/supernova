"""
Modal batch runner for vectorforge — massively parallel slice-based embedding.

Instead of one job with internal workers, this spawns thousands of independent
Modal jobs, each processing a slice of the dataset.

Usage:
  # Dry run (print plan only)
  modal run modal_batch.py --config configs/arxiv.yaml --dry-run

  # CPU (API-based embedders like OpenAI)
  modal run modal_batch.py --config configs/arxiv_openai.yaml

  # GPU (sentence-transformers)
  modal run modal_batch.py --config configs/arxiv_gte.yaml --gpu

  # Custom chunk size
  modal run modal_batch.py --config configs/arxiv.yaml --gpu --chunk-size 50000

  # Fire-and-forget
  modal run --detach modal_batch.py --config configs/arxiv.yaml --gpu
"""

import modal

app = modal.App("vectorforge-batch")

DEPS = [
    "datasets",
    "pyarrow",
    "openai",
    "tiktoken",
    "tqdm",
    "aiobotocore",
    "huggingface_hub",
    "pyyaml",
    "httpx",
    "sentence-transformers",
    "torch",
    "transformers<5",
]

LOCAL_DIRS = [
    ("vectorforge", "/app/vectorforge"),
    ("scripts", "/app/scripts"),
    ("configs", "/app/configs"),
]

image = modal.Image.debian_slim(python_version="3.13").pip_install(*DEPS)
for src, dst in LOCAL_DIRS:
    image = image.add_local_dir(src, dst)

gpu_image = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-runtime-ubuntu24.04", add_python="3.13"
).pip_install(*DEPS)
for src, dst in LOCAL_DIRS:
    gpu_image = gpu_image.add_local_dir(src, dst)


def _process_slice(slice_args: dict) -> dict:
    """
    Process a single dataset slice: stream rows, embed, write parquet, upload.

    slice_args keys:
        source_cfg, embedder_cfg, storage_cfg, offset, limit, slice_id
    """
    import sys
    sys.path.insert(0, "/app")

    import asyncio
    import time

    from datasets import load_dataset
    from scripts.run_pipeline import build_embedder, build_storage
    from vectorforge.models import Record, EmbeddedRecord
    from vectorforge.storage.writer import write_batch
    from vectorforge.sources.huggingface import _build_text_extractor

    source_cfg = slice_args["source_cfg"]
    embedder_cfg = slice_args["embedder_cfg"]
    storage_cfg = slice_args["storage_cfg"]
    offset = slice_args["offset"]
    limit = slice_args["limit"]
    slice_id = slice_args["slice_id"]
    held_out_indices = set(slice_args.get("held_out_indices", []))

    t0 = time.time()

    # Build embedder and storage
    embedder = build_embedder(dict(embedder_cfg))
    storage = build_storage(dict(storage_cfg))

    # Load only our slice using HF split range syntax
    base_split = source_cfg.get("split", "train")
    ds = load_dataset(
        source_cfg["dataset_name"],
        source_cfg.get("config"),
        split=f"{base_split}[{offset}:{offset + limit}]",
    )
    stream = ds

    extract_text = _build_text_extractor(
        source_cfg.get("text_field"),
        source_cfg.get("text_template"),
    )
    payload_fields = source_cfg.get("payload_fields", [])

    # Collect records, splitting text as needed
    from tqdm import tqdm

    records: list[Record] = []
    row_counter = 0
    skipped = 0
    for local_index, row in enumerate(tqdm(ds, total=limit, desc=f"[slice {slice_id}] Streaming")):
        source_row_id = offset + local_index

        if source_row_id in held_out_indices:
            skipped += 1
            continue

        text = extract_text(row)
        if not text or not text.strip():
            continue

        payload = {k: row[k] for k in payload_fields if k in row} if payload_fields else {}
        chunks = embedder.split_text(text)

        for chunk_index, chunk_text in enumerate(chunks):
            records.append(Record(
                row_id=row_counter,
                source_row_id=source_row_id,
                chunk_id=slice_id,
                chunk_index=chunk_index,
                text=chunk_text,
                source=source_cfg["dataset_name"],
                payload=payload,
            ))
            row_counter += 1

    if not records:
        return {
            "slice_id": slice_id,
            "num_records": 0,
            "held_out": skipped,
            "elapsed": round(time.time() - t0, 1),
        }

    # Embed — batch the texts so we get a progress bar over batches
    texts = [r.text for r in records]
    batch_size = embedder_cfg.get("batch_size", 32)
    embeddings: list[list[float]] = []
    embed_progress = tqdm(total=len(texts), desc=f"[slice {slice_id}] Embedding", unit=" texts")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = asyncio.run(embedder.embed(batch))
        embeddings.extend(batch_embs)
        embed_progress.update(len(batch))
    embed_progress.close()

    # Build EmbeddedRecords
    embedded: list[EmbeddedRecord] = []
    for rec, emb in zip(records, embeddings):
        embedded.append(EmbeddedRecord(
            row_id=rec.row_id,
            source_row_id=rec.source_row_id,
            chunk_id=rec.chunk_id,
            chunk_index=rec.chunk_index,
            text=rec.text,
            source=rec.source,
            embedding=emb,
            model=embedder.model_name,
            payload=rec.payload,
        ))

    # Write parquet
    output_dir = "/tmp/vectorforge"
    parquet_path = write_batch(embedded, output_dir, slice_id)

    # Upload to storage
    asyncio.run(storage.upload_file(parquet_path))

    elapsed = round(time.time() - t0, 1)
    print(f"[slice {slice_id:08d}] {len(embedded)} records, {skipped} held out, {elapsed}s")

    return {
        "slice_id": slice_id,
        "num_records": len(embedded),
        "held_out": skipped,
        "elapsed": elapsed,
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("vectorforge-secrets")],
    timeout=7200,
)
def embed_slice(slice_args: dict) -> dict:
    """
    CPU slice processor (for API-based embedders like OpenAI).
    """
    return _process_slice(slice_args)


@app.function(
    image=gpu_image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("vectorforge-secrets")],
    timeout=7200,
)
def embed_slice_gpu(slice_args: dict) -> dict:
    """
    GPU slice processor (for sentence-transformers).
    """
    return _process_slice(slice_args)


@app.local_entrypoint()
def main(
    config: str,
    gpu: bool = False,
    dry_run: bool = False,
    num_queries: int = 10_000,
):
    import json
    import math
    import random
    import time

    import yaml

    from datasets import load_dataset_builder

    # Read config
    with open(config) as f:
        cfg = yaml.safe_load(f)

    source_cfg = cfg["source"]
    embedder_cfg = cfg["embedder"]
    storage_cfg = cfg["storage"]
    pipeline_cfg = cfg.get("pipeline", {})

    # config overrides default
    chunk_size = pipeline_cfg.get("chunk_size", 100_000)

    dataset_name = source_cfg["dataset_name"]
    hf_config = source_cfg.get("config")
    split = source_cfg.get("split", "train")

    # Get dataset size
    builder = load_dataset_builder(dataset_name, hf_config)
    total_rows = builder.info.splits[split].num_examples

    # Generate deterministic held-out indices
    num_queries = min(num_queries, total_rows // 10)  # cap at 10% of dataset
    random.seed(42)
    held_out_indices = sorted(random.sample(range(total_rows), num_queries))

    num_jobs = math.ceil(total_rows / chunk_size)

    print("=" * 60)
    print("vectorforge batch plan")
    print("=" * 60)
    print(f"  Dataset:      {dataset_name}")
    print(f"  Split:        {split}")
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Held out:     {num_queries:,} queries")
    print(f"  Corpus rows:  ~{total_rows - num_queries:,}")
    print(f"  Chunk size:   {chunk_size:,}")
    print(f"  Num jobs:     {num_jobs}")
    print(f"  GPU:          {gpu}")
    print(f"  Embedder:     {embedder_cfg.get('type')} / {embedder_cfg.get('model', 'default')}")
    print(f"  Storage:      {storage_cfg.get('type')} / {storage_cfg.get('s3_bucket', storage_cfg.get('repo_id', 'local'))}")
    print("=" * 60)

    if dry_run:
        print("\n[dry run] No jobs submitted.")
        return

    # Build slice arguments, distributing held-out indices to their respective slices
    slices = []
    for i in range(num_jobs):
        offset = i * chunk_size
        limit = min(chunk_size, total_rows - offset)
        slice_held_out = [idx for idx in held_out_indices if offset <= idx < offset + limit]
        slices.append({
            "source_cfg": dict(source_cfg),
            "embedder_cfg": dict(embedder_cfg),
            "storage_cfg": dict(storage_cfg),
            "offset": offset,
            "limit": limit,
            "slice_id": i,
            "held_out_indices": slice_held_out,
        })

    # Dispatch
    fn = embed_slice_gpu if gpu else embed_slice
    print(f"\nSubmitting {num_jobs} jobs...")
    t0 = time.time()

    results = []
    for result in fn.map(slices):
        results.append(result)
        print(f"  Completed slice {result['slice_id']:08d}: "
              f"{result['num_records']:,} records, {result.get('held_out', 0)} held out, {result['elapsed']}s")

    total_time = round(time.time() - t0, 1)
    total_records = sum(r["num_records"] for r in results)

    # Upload manifest
    from scripts.run_pipeline import build_storage

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "total_rows": total_rows,
        "total_records": total_records,
        "held_out_indices": held_out_indices,
        "num_queries": num_queries,
        "queries_hydrated": False,
        "chunk_size": chunk_size,
        "num_slices": num_jobs,
        "gpu": gpu,
        "embedder": embedder_cfg,
        "slices": results,
        "total_time_seconds": total_time,
    }

    storage = build_storage(dict(storage_cfg))
    import asyncio
    asyncio.run(
        storage.upload_bytes(
            json.dumps(manifest, indent=2).encode(),
            "_manifest.json",
        )
    )

    print("\n" + "=" * 60)
    print("batch complete")
    print("=" * 60)
    print(f"  Corpus records:  {total_records:,}")
    print(f"  Held out:        {num_queries:,} (stored in manifest)")
    print(f"  Total time:      {total_time}s")
    print(f"  Queries hydrated: False")
    print(f"  Manifest:        _manifest.json uploaded")
    print("=" * 60)
