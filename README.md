# vectorforge

Stream massive datasets, embed at scale, store as parquet in S3 or HuggingFace Hub.

## Overview

vectorforge is a pipeline for generating pre-embedded vector datasets. It streams data from sources (HuggingFace datasets, S3, etc.), embeds text using configurable backends (OpenAI, sentence-transformers), and writes the results as parquet files to S3, HuggingFace Hub, or local disk.

Key properties:
- **Streaming** -- never loads the full dataset into memory
- **Massively parallel** -- splits datasets into slices and processes them across hundreds of GPUs via Modal
- **Text splitting** -- long texts are automatically split using each embedder's native tokenizer
- **Pluggable** -- add new sources, embedders, or storage backends by subclassing

## Quickstart

```bash
# Install
uv sync

# Run a pipeline locally
uv run python scripts/run_pipeline.py configs/mteb_tweets_openai.yaml
```

## Configuration

Pipelines are defined as YAML configs:

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

## Environment variables

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI embedder |
| `HF_TOKEN` | HuggingFace Hub storage |
| `AWS_ACCESS_KEY_ID` | S3 storage |
| `AWS_SECRET_ACCESS_KEY` | S3 storage |

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
modal run modal_batch.py --config configs/nick007x_arxiv_papers.yaml --gpu --dry-run

# Run with GPU (sentence-transformers)
modal run modal_batch.py --config configs/nick007x_arxiv_papers.yaml --gpu

# Run with CPU (API-based embedders like OpenAI)
modal run modal_batch.py --config configs/mteb_tweets_openai.yaml

# Custom chunk size (smaller = more parallelism)
modal run modal_batch.py --config configs/nick007x_arxiv_papers.yaml --gpu --chunk-size 50000

# Fire and forget
modal run --detach modal_batch.py --config configs/nick007x_arxiv_papers.yaml --gpu
```

### How it works

1. **Planner** (runs locally): reads config, queries dataset size, divides into slices
2. **`spawn_map`**: kicks off N independent jobs on Modal
3. **Each job**: streams its slice from HuggingFace, embeds, writes one parquet, uploads to storage
4. **Manifest**: after all jobs complete, a `_manifest.json` is uploaded with run metadata

### Local runner

For development and small datasets, run locally without Modal:

```bash
uv run python scripts/run_pipeline.py configs/mteb_tweets_openai.yaml
```

The local runner uses async workers with a priority queue buffer for ordered output.

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
