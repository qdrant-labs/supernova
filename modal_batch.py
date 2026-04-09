"""
Modal batch runner for vectorforge — massively parallel slice-based embedding.

Instead of one job with internal workers, this spawns thousands of independent
Modal jobs, each processing a slice of the dataset.

Usage:
  # Dry run (print plan only)
  modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --dry-run

  # CPU (API-based embedders like OpenAI)
  modal run modal_batch.py --config configs/embedder/mteb_tweets_openai.yaml

  # GPU (sentence-transformers)
  modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu

  # Custom chunk size
  modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu --chunk-size 50000

  # Fire-and-forget
  modal run --detach modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu
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
    from scripts.run_embedder import build_embedder, build_storage
    from vectorforge.models import Record, EmbeddedRecord
    from vectorforge.storage.writer import write_batch
    from vectorforge.sources.huggingface import _build_text_extractor

    source_cfg = slice_args["source_cfg"]
    embedder_cfg = slice_args["embedder_cfg"]
    storage_cfg = slice_args["storage_cfg"]
    offset = slice_args["offset"]
    limit = slice_args["limit"]
    slice_id = slice_args["slice_id"]
    max_text_length = slice_args.get("max_text_length")
    t0 = time.time()

    # Build embedder and storage
    embedder = build_embedder(dict(embedder_cfg))
    storage = build_storage(dict(storage_cfg))

    # Stream only our slice — avoids downloading the full dataset per container
    base_split = source_cfg.get("split", "train")
    ds = load_dataset(
        source_cfg["dataset_name"],
        source_cfg.get("config"),
        split=base_split,
        streaming=True,
    )
    ds = ds.skip(offset).take(limit)

    extract_text = _build_text_extractor(
        source_cfg.get("text_field"),
        source_cfg.get("text_template"),
    )
    exclude_columns = set(source_cfg.get("exclude_columns", []))

    # Collect records, splitting text as needed
    from tqdm import tqdm

    records: list[Record] = []
    row_counter = 0
    for local_index, row in enumerate(tqdm(ds, total=limit, desc=f"[slice {slice_id}] Streaming")):
        source_row_id = offset + local_index

        text = extract_text(row)
        if not text or not text.strip():
            continue

        if max_text_length and len(text) > max_text_length:
            text = text[:max_text_length]

        columns = {k: v for k, v in row.items() if k not in exclude_columns}
        chunks = embedder.split_text(text)

        for chunk_index, chunk_text in enumerate(chunks):
            records.append(Record(
                row_id=row_counter,
                source_row_id=source_row_id,
                chunk_id=slice_id,
                chunk_index=chunk_index,
                text=chunk_text,
                columns=columns,
            ))
            row_counter += 1

    if not records:
        return {
            "slice_id": slice_id,
            "num_records": 0,
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
            embedding=emb,
            columns=rec.columns,
        ))

    # Write parquet
    output_dir = "/tmp/vectorforge"
    parquet_path = write_batch(embedded, output_dir, slice_id)

    # Upload to storage
    asyncio.run(storage.upload_file(parquet_path))

    elapsed = round(time.time() - t0, 1)
    print(f"[slice {slice_id:08d}] {len(embedded)} records, {elapsed}s")

    return {
        "slice_id": slice_id,
        "num_records": len(embedded),
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
):
    import json
    import math
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
    max_text_length = pipeline_cfg.get("max_text_length")

    dataset_name = source_cfg["dataset_name"]
    hf_config = source_cfg.get("config")
    split = source_cfg.get("split", "train")

    # Get dataset size
    builder = load_dataset_builder(dataset_name, hf_config)
    total_rows = builder.info.splits[split].num_examples

    num_jobs = math.ceil(total_rows / chunk_size)

    print("=" * 60)
    print("vectorforge batch plan")
    print("=" * 60)
    print(f"  Dataset:      {dataset_name}")
    print(f"  Split:        {split}")
    print(f"  Total rows:   {total_rows:,}")
    print(f"  Chunk size:   {chunk_size:,}")
    print(f"  Num jobs:     {num_jobs}")
    print(f"  GPU:          {gpu}")
    print(f"  Embedder:     {embedder_cfg.get('type')} / {embedder_cfg.get('model', 'default')}")
    print(f"  Storage:      {storage_cfg.get('type')} / {storage_cfg.get('s3_bucket', storage_cfg.get('repo_id', 'local'))}")
    print("=" * 60)

    if dry_run:
        print("\n[dry run] No jobs submitted.")
        return

    slices = []
    for i in range(num_jobs):
        offset = i * chunk_size
        limit = min(chunk_size, total_rows - offset)
        slices.append({
            "source_cfg": dict(source_cfg),
            "embedder_cfg": dict(embedder_cfg),
            "storage_cfg": dict(storage_cfg),
            "offset": offset,
            "limit": limit,
            "slice_id": i,
            "max_text_length": max_text_length,
        })

    # Dispatch
    fn = embed_slice_gpu if gpu else embed_slice
    print(f"\nSubmitting {num_jobs} jobs...")
    t0 = time.time()

    results = []
    for result in fn.map(slices):
        results.append(result)
        print(f"  Completed slice {result['slice_id']:08d}: "
              f"{result['num_records']:,} records, {result['elapsed']}s")

    total_time = round(time.time() - t0, 1)
    total_records = sum(r["num_records"] for r in results)

    # Upload manifest
    from scripts.run_embedder import build_storage

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "total_rows": total_rows,
        "total_records": total_records,
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
    print(f"  Total records:   {total_records:,}")
    print(f"  Total time:      {total_time}s")
    print(f"  Manifest:        _manifest.json uploaded")
    print("=" * 60)
