# vectorforge

Generate massive pre-embedded datasets, then load them into vector databases.

## Overview

vectorforge has two pipelines:

1. **Embedding** -- stream data from HuggingFace, embed with OpenAI or sentence-transformers, write parquet to S3
2. **Loading** -- stream pre-embedded parquet from S3/HuggingFace into vector stores (Qdrant)

Both pipelines are streaming (never loads the full dataset into memory), pluggable (add new sources/embedders/stores by subclassing), and parallelizable (Modal for embedding, SkyPilot for loading).

## Quickstart

```bash
uv sync

# 1. Embed a dataset
vectorforge configs/embedder/mteb_tweets_openai.yaml

# 2. Load into Qdrant
vectorforge-load configs/loader/cohere200M.yaml

# 3. Distributed loading (SkyPilot)
vectorforge-load-distributed configs/dispatch/cohere200M.yaml --dry-run
```

## Project structure

```
vectorforge/
  sources/            # Data sources (HuggingFace, S3)
  embedders/          # Embedding backends (OpenAI, sentence-transformers)
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
  run_embedder.py     # vectorforge CLI entrypoint
  run_loader.py       # vectorforge-load CLI entrypoint
  run_dispatch.py     # vectorforge-load-distributed CLI entrypoint

modal_batch.py        # Modal distributed embedding
modal_import_cohere.py # Modal Cohere dataset import
```

---

## Pipeline 1: Embedding

### Configuration

```yaml
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train
  text_field: text

embedder:
  type: openai                    # or sentence_transformer
  model: text-embedding-3-small
  dimensions: 1536

pipeline:
  chunk_size: 10000
  num_workers: 4
  flush_threshold: 100000

storage:
  type: s3                        # or hf, local
  s3_bucket: qdrant--vectorforge
  s3_prefix: dataset-name/model-name
  output_dir: /tmp/vectorforge
```

### Embedders

| Type | Config key | Notes |
|------|-----------|-------|
| OpenAI | `openai` | `model`, `dimensions`, `batch_size`, `max_concurrent` |
| Sentence Transformers | `sentence_transformer` | `model`, `batch_size`, `dtype`. Auto-detects CUDA/MPS/CPU |

### Storage backends

| Type | Config key | Notes |
|------|-----------|-------|
| S3 | `s3` | `s3_bucket`, `s3_prefix` |
| HuggingFace Hub | `hf` | `repo_id`, `private` |
| Local | `local` | `output_dir` |

### Running locally

```bash
vectorforge configs/embedder/mteb_tweets_openai.yaml
```

### Running at scale with Modal

Modal splits the dataset into slices and processes each on its own GPU/CPU:

```bash
# Setup
pip install modal && modal setup
modal secret create vectorforge-secrets \
  OPENAI_API_KEY=$OPENAI_API_KEY \
  AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  AWS_DEFAULT_REGION=us-east-1 \
  HF_TOKEN=$HF_TOKEN

# Run
modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu
modal run modal_batch.py --config configs/embedder/mteb_tweets_openai.yaml          # CPU for API embedders
modal run modal_batch.py --config configs/embedder/nick007x_arxiv_papers.yaml --gpu --dry-run  # preview only
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
| `source` | string | Dataset identifier |
| `embedding` | list\<float32\> | The embedding vector |
| `model` | string | Model used for embedding |

Query with DuckDB:

```sql
SELECT * FROM 's3://qdrant--vectorforge/dataset/model/**/*.parquet' LIMIT 10;
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
    embedding: emb                  # default: embedding
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
vectorforge-load configs/loader/cohere200M.yaml
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

### Column mapping

The `columns` field maps logical names to actual parquet column names:

| Logical name | Default | Description |
|-------------|---------|-------------|
| `id` | `row_id` | Point ID in the vector store |
| `embedding` | `embedding` | Vector column |

### Payload composition

`payload_fields` maps payload keys to parquet columns. Only the fields you list end up in the vector store payload:

```yaml
payload_fields:
  abstract: text        # parquet "text" column stored as "abstract" in payload
  source: source
```

Default when omitted: `{text: text}`.

### How it works

1. DuckDB streams parquet data in large prefetch chunks (minimizes S3 round trips)
2. Chunks are sliced into upsert-sized batches and written concurrently via asyncio
3. **Deferred indexing** -- HNSW construction is disabled during load, then enabled for one efficient batch build
4. Failed upserts are retried with exponential backoff

### Distributed loading with SkyPilot

For terabyte-scale datasets, fan out across SkyPilot spot instances:

```bash
vectorforge-load-distributed configs/dispatch/cohere200M.yaml
vectorforge-load-distributed configs/dispatch/cohere200M.yaml --dry-run      # preview only
vectorforge-load-distributed configs/dispatch/cohere200M.yaml --num-shards 20  # override shard count
```

Dispatch config adds `dispatch` and `resources` sections to the standard loader config:

```yaml
dispatch:
  num_shards: 10
  run_name: cohere200M

resources:
  cpus: 2
  memory: 8
  cloud: aws
  use_spot: true
```

The dispatch flow:
1. Lists all parquet files at the S3 prefix
2. Divides files round-robin across N shards
3. Generates per-shard loader + SkyPilot YAML configs in `runs/<timestamp>_<name>/`
4. Creates Qdrant collection and defers indexing
5. Launches N SkyPilot spot instance jobs (credentials injected at runtime, never written to disk)
6. Waits for all jobs to complete
7. Enables indexing and measures HNSW build time
8. Writes `report.json`

Requires [SkyPilot](https://skypilot.readthedocs.io/) configured with AWS credentials. See [docs/aws-sso-setup.md](docs/aws-sso-setup.md).

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
- [Installation](docs/installation.md) -- setup, environment variables, Modal and SkyPilot configuration
- [Quickstart](docs/quickstart.md) -- embed a dataset and load it into Qdrant end-to-end
- [Embedding Generation](docs/embedding-generation.md) -- embedder options, Modal at scale, output format
- [Data Loading](docs/data-loading.md) -- column mapping, payload composition, distributed loading
- [Loader Architecture](docs/loader.md) -- internal design docs
- [AWS SSO Setup](docs/aws-sso-setup.md) -- configuring AWS SSO for local and Modal usage
- [SkyPilot Migration](docs/skypilot-migration.md) -- plan for moving sustained workloads to SkyPilot
