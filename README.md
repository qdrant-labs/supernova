# vectorforge

Stream massive datasets, embed at scale, load into vector databases.

## Overview

vectorforge is a pipeline for generating and loading pre-embedded vector datasets. It streams data from sources (HuggingFace datasets, S3, etc.), embeds text using configurable backends (OpenAI, sentence-transformers), writes the results as parquet files to S3 or HuggingFace Hub, and loads them into vector stores (Qdrant, etc.) for benchmarking.

Key properties:
- **Streaming** -- never loads the full dataset into memory
- **Massively parallel** -- splits datasets into slices and processes them across hundreds of GPUs via Modal
- **Text splitting** -- long texts are automatically split using each embedder's native tokenizer
- **Pluggable** -- add new sources, embedders, storage backends, or vector stores by subclassing
- **Database loading** -- stream pre-embedded parquet data into vector stores with deferred indexing for bulk performance
- **Distributed loading** -- fan out loading jobs across SkyPilot spot instances for terabyte-scale datasets

## Quickstart

```bash
# Install
uv sync

# Embed a dataset locally
vectorforge configs/embedder/mteb_tweets_openai.yaml

# Load pre-embedded data into Qdrant
vectorforge-load configs/loader/arxiv_papers_qdrant.yaml

# Distributed loading across SkyPilot spot instances
vectorforge-load-distributed configs/dispatch/cohere200M.yaml --dry-run
```

## Embedding configuration

Embedding pipelines are defined as YAML configs:

```yaml
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train
  text_field: text

embedder:
  type: openai
  model: text-embedding-3-small
  dimensions: 1536

pipeline:
  chunk_size: 10000
  num_workers: 4
  flush_threshold: 100000

storage:
  type: s3
  s3_bucket: qdrant---vectorforge
  s3_prefix: mteb--tweet-sentiment/openai-3-small
  output_dir: /tmp/vectorforge
```

### Storage backends

**S3:**
```yaml
storage:
  type: s3
  s3_bucket: qdrant---vectorforge
  s3_prefix: dataset-name/model-name
```

**HuggingFace Hub:**
```yaml
storage:
  type: hf
  repo_id: Qdrant/dataset-name--model-name
  private: true
```

**Local (no upload):**
```yaml
storage:
  type: local
  output_dir: /tmp/vectorforge
```

### Embedders

| Type | Config key | Notes |
|------|-----------|-------|
| OpenAI | `openai` | Supports `model`, `dimensions`, `batch_size`, `max_concurrent` |
| Sentence Transformers | `sentence_transformer` | Supports `model`, `batch_size`, `device`. Auto-detects CUDA/MPS/CPU |

### Sentence Transformers example

```yaml
embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 32
```

## Output format

Output is parquet with this schema:

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int64 | Auto-incrementing record ID |
| `source_row_id` | int64 | Original row in the source dataset |
| `chunk_id` | int32 | Pipeline batch / slice ID |
| `chunk_index` | int32 | Position within a text split (0 if not split) |
| `text` | string | The embedded text |
| `source` | string | Dataset identifier |
| `embedding` | list\<float32\> | The embedding vector |
| `model` | string | Model used for embedding |
| `payload` | string | JSON metadata from the source |

Query with DuckDB:

```sql
SELECT * FROM 's3://qdrant---vectorforge/dataset/model/*.parquet' LIMIT 10;
```

## Running at scale with Modal

vectorforge uses [Modal](https://modal.com) for massively parallel embedding. The dataset is split into slices, and each slice runs as an independent job on its own GPU/CPU. For a 10M row dataset with `chunk_size=100_000`, that's 100 jobs running concurrently.

### Setup

1. Install Modal and authenticate:

```bash
pip install modal
modal setup
```

2. Create a secret group with your credentials:

```bash
modal secret create vectorforge-secrets \
  OPENAI_API_KEY=$OPENAI_API_KEY \
  AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  AWS_DEFAULT_REGION=us-east-1 \
  HF_TOKEN=$HF_TOKEN
```

### Running jobs

```bash
# Preview the plan (no jobs submitted)
modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu --dry-run

# Run with GPU (sentence-transformers)
modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu

# Run with CPU (API-based embedders like OpenAI)
modal run modal_batch.py --config configs/embedder/mteb_tweets_openai.yaml

# Custom chunk size (smaller = more parallelism)
modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu --chunk-size 50000

# Fire and forget
modal run --detach modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu
```

### How it works

1. **Planner** (runs locally): reads config, queries dataset size, divides into slices
2. **`spawn_map`**: kicks off N independent jobs on Modal
3. **Each job**: streams its slice from HuggingFace, embeds, writes one parquet, uploads to storage
4. **Manifest**: after all jobs complete, a `_manifest.json` is uploaded with run metadata

### Local runner

For development and small datasets, run locally without Modal:

```bash
uv run python scripts/run_pipeline.py configs/embedder/mteb_tweets_openai.yaml
```

The local runner uses async workers with a priority queue buffer for ordered output.

## Loading into vector stores

Once you have pre-embedded parquet data on S3 or HuggingFace, load it into a vector store:

```bash
vectorforge-load configs/loader/cohere200M.yaml
```

### Loader configuration

```yaml
datasource:
  type: s3                          # s3 or huggingface
  s3_bucket: qdrant---vectorforge
  s3_prefix: cohere--wikipedia/embed-multilingual-v3
  columns:                          # override parquet column names
    id: _id                         # default: row_id
    embedding: emb                  # default: embedding
  payload_fields:                   # parquet columns to include as payload
    text: text                      # payload key: parquet column
    url: url
    title: title

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

### Datasources

| Type | Config key | Notes |
|------|-----------|-------|
| S3 | `s3` | Requires `s3_bucket`, `s3_prefix`. Streams via DuckDB httpfs |
| HuggingFace | `huggingface` | Requires `repo_id`. Streams via DuckDB `hf://` protocol |

**HuggingFace example:**
```yaml
datasource:
  type: huggingface
  repo_id: CohereLabs/wikipedia-2023-11-embed-multilingual-v3
  subdir: en                        # optional, scope to a subfolder
```

### Vector stores

| Type | Config key | Notes |
|------|-----------|-------|
| Qdrant | `qdrant` | Requires `url`, `api_key`, `collection_name` |

### How it works

1. DuckDB streams parquet data in large prefetch chunks (minimizes remote I/O)
2. Chunks are sliced into upsert-sized batches and written concurrently
3. **Deferred indexing** -- HNSW indexing is disabled during load, then enabled for a single efficient batch build
4. Progress bar tracks points loaded, with throughput logging

### Column mapping

The `columns` field maps logical names to actual parquet column names. Defaults:

| Logical name | Default parquet column |
|-------------|----------------------|
| `id` | `row_id` |
| `embedding` | `embedding` |

### Payload composition

The `payload_fields` field controls what goes into the vector store payload. Each entry maps a payload key (what gets stored) to a parquet column name (where the data comes from):

```yaml
payload_fields:
  abstract: text        # store parquet "text" column as "abstract"
  source: source
  metadata: payload     # JSON columns are auto-unpacked
```

If omitted, defaults to `{text: text}`.

## Distributed loading with SkyPilot

For terabyte-scale datasets, distribute loading across SkyPilot spot instances:

```bash
vectorforge-load-distributed configs/dispatch/cohere200M.yaml
```

### Dispatch configuration

Extends the loader config with `dispatch` and `resources` sections:

```yaml
dispatch:
  num_shards: 10                    # number of parallel workers
  run_name: cohere200M              # optional, used in run directory name

resources:                          # SkyPilot VM spec
  cpus: 8
  memory: 32
  cloud: aws
  use_spot: true                    # 60-90% cheaper than on-demand

# Standard loader config (passed through to workers)
datasource:
  type: s3
  s3_bucket: qdrant---vectorforge
  s3_prefix: cohere--wikipedia/embed-multilingual-v3
  columns:
    id: _id
    embedding: emb
  payload_fields:
    text: text
    url: url
    title: title

vectorstore:
  type: qdrant
  collection_name: cohere-wikipedia
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}

loader:
  batch_size: 1000
  prefetch_size: 100000
  concurrency: 8
```

### How it works

1. **Discover** -- lists all parquet files at the S3 prefix via boto3
2. **Shard** -- divides files round-robin across N workers
3. **Generate** -- writes per-shard loader + SkyPilot YAML configs to `runs/<run_id>/`
4. **Setup** -- creates Qdrant collection and defers indexing
5. **Launch** -- fans out N SkyPilot spot instance jobs (env vars injected at runtime, never written to disk)
6. **Wait** -- polls `sky jobs queue` until all complete
7. **Index** -- enables indexing and waits for HNSW build
8. **Report** -- writes `report.json` with timing and success/failure counts

### Dry run

Preview the plan without launching:

```bash
vectorforge-load-distributed configs/dispatch/cohere200M.yaml --dry-run
```

### Generated artifacts

Each run produces a directory with a full paper trail:

```
runs/2026-04-07T14-30_cohere200M/
  manifest.json              # file list + shard assignments
  shard_000_loader.yaml      # per-shard loader config (with file_list)
  shard_000_sky.yaml         # per-shard SkyPilot job config
  shard_001_loader.yaml
  shard_001_sky.yaml
  ...
  report.json                # timing, success/failure counts
```

### Prerequisites

1. [SkyPilot](https://skypilot.readthedocs.io/) installed and configured with AWS credentials
2. Required env vars set: `QDRANT_URL`, `QDRANT_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

See [docs/aws-sso-setup.md](docs/aws-sso-setup.md) for AWS SSO credential setup.

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

## Dashboard

A static Next.js site that displays all completed embedding runs by reading `_manifest.json` files from S3.

```bash
cd dashboard
npm install
npm run fetch-manifests  # pull manifests from S3
npm run build            # static export to out/
```

## Tests

```bash
uv run pytest tests/ -v
```