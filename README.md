# vectorforge

Generate massive pre-embedded datasets, then load them into vector databases.

## Overview

vectorforge has two pipelines:

1. **Embedding** -- stream data from HuggingFace, embed with dense and/or sparse models, write parquet to S3
2. **Loading** -- stream pre-embedded parquet from S3/HuggingFace into vector stores (Qdrant)

Both pipelines are streaming (never loads the full dataset into memory), pluggable (add new sources/embedders/stores by subclassing), and parallelizable (SkyPilot for distributed embedding and loading).

## Quickstart

```bash
uv sync

# 1. Embed a dataset locally
vf embed configs/embedder/nick007x_arxiv_papers.yaml

# 2. Embed distributed across SkyPilot GPU pool
vf embed-dist configs/embedder/nick007x_arxiv_papers.yaml

# 3. Load into Qdrant
vf load configs/loader/cohere200M.yaml

# 4. Distributed loading (SkyPilot)
vf load-dist configs/dispatch/cohere200M.yaml
```

## Project structure

```
vectorforge/
  sources/            # Data sources (HuggingFace)
  embedders/
    dense/            # Dense embedding backends (OpenAI, sentence-transformers)
    sparse/           # Sparse embedding backends (sentence-transformers SparseEncoder)
    engine.py         # EmbeddingEngine -- orchestrates dense/sparse/hybrid
    hybrid.py         # HybridEmbedder -- single forward pass for both
  storage/            # Output backends (S3, HuggingFace Hub, local)
  pipeline/           # Embedding orchestration (runner, worker, buffer)
  loader/
    datasource/       # Parquet readers (S3, HuggingFace)
    vectorstore/      # Vector store backends (Qdrant)
    runner.py         # Loading orchestration

configs/
  embedder/           # Embedding pipeline configs
  loader/             # Loading pipeline configs
  dispatch/           # Distributed loading configs

scripts/
  run_embedder.py           # vectorforge CLI
  run_embed_distributed.py  # vf embed-dist CLI
  run_loader.py             # vf load CLI
  run_load_distributed.py   # vf load-dist CLI
```

---

## Pipeline 1: Embedding

### Configuration

```yaml
source:
  type: huggingface
  dataset_name: nick007x/arxiv-papers
  split: train
  text_field: abstract

dense_embedder:
  type: sentence_transformer    # or openai
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16

pipeline:
  chunk_size: 100000
  num_workers: 2

storage:
  type: s3                      # or hf, local
  s3_bucket: qdrant--vectorforge
  s3_prefix: arxiv-papers/gte-multilingual-base
  output_dir: /tmp/vectorforge
```

### Sparse embeddings

Add a `sparse_embedder` section to produce sparse vectors alongside dense:

```yaml
dense_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16

sparse_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  batch_size: 64
  dtype: bfloat16
```

When both point to the same model, vectorforge automatically uses a hybrid encoder to minimize forward passes. You must specify at least one of `dense_embedder` or `sparse_embedder`.

### Dense embedders

| Type | Config key | Notes |
|------|-----------|-------|
| OpenAI | `openai` | `model`, `dimensions`, `batch_size`, `max_concurrent`, `base_url`, `api_key` |
| Sentence Transformers | `sentence_transformer` | `model`, `batch_size`, `dtype`. Auto-detects CUDA/MPS/CPU |

The OpenAI embedder supports any OpenAI-compatible API via `base_url` (llama.cpp, vLLM, Ollama, etc). Set `api_key: none` for local servers that don't require auth.

### Storage backends

| Type | Config key | Notes |
|------|-----------|-------|
| S3 | `s3` | `s3_bucket`, `s3_prefix` |
| HuggingFace Hub | `hf` | `repo_id`, `private` |
| Local | `local` | `output_dir` |

### Running locally

```bash
vf embed configs/embedder/nick007x_arxiv_papers.yaml
```

### Running at scale with SkyPilot

SkyPilot pools create GPU workers and distribute embedding jobs across them. Workers are reused -- setup happens once, not per-slice.

```bash
# Preview the plan
vf embed-dist configs/embedder/nick007x_arxiv_papers.yaml --dry-run

# Run (default: A10G spot, autoscaling)
vf embed-dist configs/embedder/nick007x_arxiv_papers.yaml

# Custom parallelism
vf embed-dist configs/embedder/nick007x_arxiv_papers.yaml --num-jobs 20
```

Override resources in your config:

```yaml
resources:
  accelerators: A10G:1
  cloud: aws
  use_spot: true
```

### Output format

Parquet files with this schema:

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int64 | Auto-incrementing record ID |
| `source_row_id` | int64 | Original row in the source dataset |
| `chunk_id` | int32 | Pipeline batch / slice ID |
| `chunk_index` | int32 | Position within a text split (0 if not split) |
| `text` | string | The embedded text |
| `dense_embedding` | list\<float32\> | Dense embedding vector (when configured) |
| `sparse_embedding` | struct{indices, values} | Sparse embedding (when configured) |

Query with DuckDB:

```sql
SELECT row_id, text[:80] AS preview, length(dense_embedding) AS dim
FROM 's3://qdrant--vectorforge/dataset/model/**/*.parquet'
LIMIT 10;
```

---

## Pipeline 2: Loading

### Configuration

```yaml
datasource:
  type: s3                          # s3 or huggingface
  s3_bucket: qdrant--vectorforge
  s3_prefix: cohere--wikipedia/embed-multilingual-v3
  columns:                          # optional: override parquet column names
    id: _id                         # default: row_id
    embedding: dense_embedding      # default: dense_embedding
  payload_fields:                   # what goes into the vector store payload
    text: text                      # payload key: parquet column name
    source: source

vectorstore:
  type: qdrant
  collection_name: cohere-wikipedia
  url: ${QDRANT_URL}                # env var substitution
  api_key: ${QDRANT_API_KEY}

loader:
  batch_size: 1000                  # points per upsert
  prefetch_size: 100000             # rows per DuckDB fetch (default: batch_size * 10)
  concurrency: 8                    # parallel upsert tasks
```

### Running

```bash
vf load configs/loader/cohere200M.yaml
```

### Datasources

| Type | Config key | Notes |
|------|-----------|-------|
| S3 | `s3` | `s3_bucket`, `s3_prefix`. Streams via DuckDB httpfs |
| HuggingFace | `huggingface` | `repo_id`, optional `subdir`. Streams via DuckDB `hf://` protocol |

### Vector stores

| Type | Config key | Notes |
|------|-----------|-------|
| Qdrant | `qdrant` | `url`, `api_key`, `collection_name`. Retry with backoff on timeouts |

### How it works

1. DuckDB streams parquet data in large prefetch chunks (minimizes S3 round trips)
2. Chunks are sliced into upsert-sized batches and written concurrently via asyncio
3. **Deferred indexing** -- HNSW construction is disabled during load, then enabled for one efficient batch build
4. Failed upserts are retried with exponential backoff

### Distributed loading with SkyPilot

For terabyte-scale datasets, fan out across SkyPilot spot instances:

```bash
vf load-dist configs/dispatch/cohere200M.yaml
vf load-dist configs/dispatch/cohere200M.yaml --dry-run
vf load-dist configs/dispatch/cohere200M.yaml --num-shards 20
```

---

## Environment variables

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI embedder |
| `HF_TOKEN` | HuggingFace Hub storage / datasource |
| `AWS_ACCESS_KEY_ID` | S3 storage / datasource |
| `AWS_SECRET_ACCESS_KEY` | S3 storage / datasource |
| `AWS_SESSION_TOKEN` | S3 with AWS SSO |
| `QDRANT_URL` | Qdrant vector store |
| `QDRANT_API_KEY` | Qdrant vector store |

## Tests

```bash
uv run pytest tests/ -v
```

## Documentation

- [Introduction](docs/introduction.md) -- concepts, mental model, architecture diagrams
- [Installation](docs/installation.md) -- setup, environment variables, SkyPilot configuration
- [Quickstart](docs/quickstart.md) -- embed a dataset and load it into Qdrant end-to-end
- [Embedding Generation](docs/embedding-generation.md) -- dense/sparse embedders, SkyPilot at scale, output format
- [Data Loading](docs/data-loading.md) -- column mapping, payload composition, distributed loading
- [Loader Architecture](docs/loader.md) -- internal design docs
- [AWS SSO Setup](docs/aws-sso-setup.md) -- configuring AWS SSO credentials
- [SkyPilot](docs/skypilot-migration.md) -- distributed compute setup and cost estimates
