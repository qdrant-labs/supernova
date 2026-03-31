# vectorforge

Stream massive datasets, embed at scale, store as parquet in S3 or HuggingFace Hub.

## Overview

vectorforge is a pipeline for generating pre-embedded vector datasets. It streams data from sources (HuggingFace datasets, S3, etc.), embeds text using configurable backends (OpenAI, Cohere, Baseten, Modal), and writes the results as parquet files to S3 or HuggingFace Hub.

Key properties:
- **Streaming** -- never loads the full dataset into memory
- **Async** -- embedding, flushing, and chunking all run concurrently
- **Ordered output** -- a priority queue buffer ensures parquet files are written in order even though workers finish out of order
- **Batched flushes** -- accumulates records up to a configurable threshold before writing, producing fewer and larger parquet files
- **Text splitting** -- long texts are automatically split by token count (via tiktoken) so each record fits within model limits
- **Pluggable** -- add new sources, embedders, or storage backends by subclassing

## Quickstart

```bash
# Install
uv sync

# Run a pipeline
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
  max_tokens: 8192

storage:
  type: s3
  s3_bucket: qdrant--vectorforge
  s3_prefix: mteb--tweet-sentiment/openai-3-small
  output_dir: /tmp/vectorforge
```

### Storage backends

**S3:**
```yaml
storage:
  type: s3
  s3_bucket: qdrant--vectorforge
  s3_prefix: dataset-name/model-name
```

**HuggingFace Hub:**
```yaml
storage:
  type: hf
  repo_id: Qdrant/dataset-name--model-name
  private: true
```

### Embedders

| Type | Config key | Notes |
|------|-----------|-------|
| OpenAI | `openai` | Supports `model`, `dimensions`, `batch_size`, `max_concurrent` |
| Cohere | `cohere` | Supports `model`, `input_type` |
| Baseten | `baseten` | Requires `deployment_id`, `api_key`, `model_id` |
| Modal | `modal` | Requires `app_name`, `function_name`, `model_id` |

## Environment variables

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI embedder |
| `COHERE_API_KEY` | Cohere embedder |
| `HF_TOKEN` | HuggingFace Hub storage |
| `AWS_ACCESS_KEY_ID` | S3 storage |
| `AWS_SECRET_ACCESS_KEY` | S3 storage |

## Output format

Output is parquet with this schema:

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int64 | Auto-incrementing record ID |
| `source_row_id` | int64 | Original row in the source dataset |
| `chunk_id` | int32 | Pipeline batch ID |
| `chunk_index` | int32 | Position within a text split (0 if not split) |
| `text` | string | The embedded text |
| `source` | string | Dataset identifier |
| `embedding` | list\<float32\> | The embedding vector |
| `model` | string | Model used for embedding |
| `payload` | string | JSON metadata from the source |

Query with DuckDB:

```sql
SELECT * FROM 's3://qdrant--vectorforge/dataset/model/*.parquet' LIMIT 10;
```

## Running at scale with AWS Batch

Dockerize and run multiple configs in parallel on Fargate:

```bash
# One-time setup
cd terraform
terraform init
terraform apply -var="openai_api_key=$OPENAI_API_KEY" -var="hf_token=$HF_TOKEN"

# Submit all configs as parallel batch jobs
./scripts/submit_batch.sh

# Or submit specific configs
./scripts/submit_batch.sh configs/mteb_tweets_openai.yaml configs/nick007x_arxiv_papers.yaml
```

This builds the Docker image, pushes to ECR, and submits one Batch job per config. Jobs run concurrently on Fargate with logs in CloudWatch.

## Dashboard

A static Next.js site that displays all completed embedding runs by reading `_manifest.json` files from S3.

```bash
cd dashboard
npm install
npm run fetch-manifests  # pull manifests from S3
npm run build            # static export to out/
```

Auto-deploys to GitHub Pages via the included workflow on push to `master`.

## Tests

```bash
uv run pytest tests/ -v
```
