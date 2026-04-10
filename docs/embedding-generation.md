# Embedding Generation

vectorforge's embedding pipeline streams data from a source, embeds text with a configurable model, and writes the results as parquet files to S3 or HuggingFace Hub.

![Embedding Pipeline](fig/embedding_pipeline.svg)

The dataset is split into N chunks. Each chunk is processed independently -- either locally (async workers) or on Modal (one container per chunk). Each chunk produces a parquet file that gets uploaded to storage.

## Configuration

Embedding configs live in `configs/embedder/` and have four sections:

```yaml
source:
  type: huggingface
  dataset_name: nick007x/arxiv-papers
  split: train
  text_field: abstract             # single field to embed
  # text_template: "{title}: {abstract}"  # or a format string

embedder:
  type: sentence_transformer       # or openai
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16                  # float32, float16, bfloat16

pipeline:
  chunk_size: 100000               # records per batch
  num_workers: 2                   # async workers (local mode)
  flush_threshold: 100000          # records before writing parquet

storage:
  type: s3                         # s3, hf, or local
  s3_bucket: my-bucket
  s3_prefix: arxiv-papers/gte-multilingual-base
  output_dir: /tmp/vectorforge
```

## Sources

### HuggingFace

Streams from any HuggingFace dataset:

```yaml
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train
  text_field: text
```

**Text extraction** -- two options:

- `text_field: abstract` -- use a single column
- `text_template: "{title}: {abstract}"` -- format string combining multiple columns

**Text splitting** -- if a text exceeds the embedder's max token limit, it's automatically split using the embedder's native tokenizer. Each piece becomes a separate record with an incrementing `chunk_index`.

## Embedders

### OpenAI

Uses the OpenAI API. Best for smaller datasets or when you don't have GPUs.

```yaml
embedder:
  type: openai
  model: text-embedding-3-small    # or text-embedding-3-large
  dimensions: 1536                 # optional dimension selection
  batch_size: 128
  max_concurrent: 8                # parallel API calls
```

- Rate limiting with exponential backoff
- Text splitting via tiktoken
- Max 8192 tokens per text

### Sentence Transformers

Runs models locally. Best for large datasets with GPU access.

```yaml
embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16                  # float32, float16, bfloat16
```

- Auto-detects CUDA, MPS (Apple Silicon), or CPU
- Supports bfloat16/float16 for faster inference
- Text splitting via the model's tokenizer

## Storage backends

### S3

```yaml
storage:
  type: s3
  s3_bucket: my-bucket
  s3_prefix: dataset-name/model-name
```

Each chunk/slice produces one parquet file, uploaded as `s3://bucket/prefix/batch_00000000.parquet`, `batch_00000001.parquet`, etc. A large dataset might produce hundreds of parquet files under the same prefix. Auto-creates the bucket if it doesn't exist.

### HuggingFace Hub

```yaml
storage:
  type: hf
  repo_id: your-org/dataset-name--model-name
  private: true
```

Creates a dataset repo and uploads parquet to the `data/` subfolder for auto-detection by HF.

### Local

```yaml
storage:
  type: local
  output_dir: /tmp/vectorforge
```

Writes parquet files locally. Useful for development.

## Output format

Every parquet file has the same flat schema:

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
SELECT row_id, text[:80] AS preview, length(embedding) AS dim
FROM 's3://my-bucket/dataset/model/**/*.parquet'
LIMIT 10;
```

## Running locally

```bash
vectorforge configs/embedder/my_dataset.yaml
```

The local runner uses async workers with a priority queue buffer to ensure ordered output. Good for development and small datasets.

## Running at scale with Modal

Modal splits the dataset into slices and processes each on its own container:

```bash
# Preview the plan
modal run modal_batch.py --config configs/embedder/arxiv_papers.yaml --gpu --dry-run

# Run with GPU (sentence-transformers)
modal run modal_batch.py --config configs/embedder/arxiv_papers.yaml --gpu

# Run with CPU (API-based embedders)
modal run modal_batch.py --config configs/embedder/tweets_openai.yaml

# Custom chunk size (smaller = more parallelism)
modal run modal_batch.py --config configs/embedder/arxiv_papers.yaml --gpu --chunk-size 50000

# Fire and forget
modal run --detach modal_batch.py --config configs/embedder/arxiv_papers.yaml --gpu
```

### How Modal works

1. **Planner** (runs locally): reads config, queries HuggingFace for dataset size, divides into slices
2. **Dispatch**: kicks off N independent jobs on Modal via `function.map()`
3. **Each job**: streams its slice from HuggingFace, embeds, writes one parquet file, uploads to storage
4. **Manifest**: after all jobs complete, a `_manifest.json` is uploaded with run metadata (dataset, model, timing, record counts)

For a 10M row dataset with `chunk_size=100,000`, that's 100 jobs running concurrently -- each on its own GPU.

## Cost and time estimation

Before embedding a dataset, estimate how long it will take and what it will cost.

### Step 1: Profile the dataset

Use `token_stats.py` to sample token lengths:

```bash
python scripts/token_stats.py HuggingFaceFW/finewiki --config en --sample 100000
```

This gives you the **mean tokens per row** from a random sample. That's the key input.

### Step 2: Estimate total tokens

```
total_tokens = rows × mean_tokens_per_row
```

The mean already accounts for skewed distributions (most datasets have a long tail of very long texts). For a confidence interval on the estimate:

```
standard_error = stdev / sqrt(sample_size)
total_tokens = rows × mean ± rows × 1.96 × standard_error  (95% CI)
```

With a 100K sample, the standard error is typically small enough to ignore.

### Step 3: Estimate time and cost

```
gpu_hours = total_tokens / throughput_tok_s / 3600
cost      = gpu_hours × price_per_gpu_hour
wall_time = gpu_hours / num_gpus
```

### Assumptions and reference values

| Parameter | Value | Notes |
|-----------|-------|-------|
| Throughput (gte-multilingual-base, A10G, bfloat16) | 50,000 tok/s | Budget estimate; raw benchmarks show 45K-87K depending on text length |
| Throughput (snowflake-arctic-embed-l-v2.0, A10G, bfloat16) | ~35,000 tok/s | Single GPU, benchmarked on real data |
| A10G price (Modal) | $0.60/hr | On-demand |
| A10G price (AWS) | ~$0.38/hr | Spot, g5.xlarge |
| T4 price (GCP) | ~$0.18/hr | Spot |
| OpenAI text-embedding-3-small | $0.02/1M tokens | API pricing, no GPU needed |

Throughput varies with text length: short texts (~20 tok avg) achieve ~58K tok/s due to padding waste, while medium-to-long texts (~700+ tok avg) hit ~80-87K tok/s. Always use bfloat16 — it's ~2x faster than float32 with no quality loss.

### Example

finewiki (English): 1.82M rows, mean 676 tok/row:

```
total_tokens = 1,820,000 × 676 = 1.23B tokens
gpu_hours    = 1,230,000,000 / 50,000 / 3600 = 6.8 GPU-hours
cost (Modal) = 6.8 × $0.60 = $4.10
cost (AWS)   = 6.8 × $0.38 = $2.60  (spot g5.xlarge)
wall_time    = 6.8 / 10 GPUs = 41 min
```

Compare with OpenAI API:

```
cost (OpenAI) = 1,230,000,000 / 1,000,000 × $0.02 = $24.60
```

Self-hosted GPU embedding is ~10x cheaper than OpenAI at this scale. The cost gap widens with larger datasets.

### Caveats

- **Throughput varies by model and GPU.** The 10K tok/s figure is for gte-multilingual-base on A10G with bfloat16. Larger models or float32 will be slower. Measure your own throughput from a test run.
- **Long texts create multiple chunks.** A 50K-token text becomes ~6 chunks at 8192 max_seq_length. The total tokens already accounts for this since we measure at the source row level.
- **Use `max_text_length` to cap outliers.** Setting `pipeline.max_text_length` in your config prevents the long tail from dominating compute. A value of ~80K characters caps texts to ~10 chunks.
- **GPU utilization matters.** The formula assumes 100% utilization. Real runs have overhead from data loading, uploads, and container startup. Add ~20% buffer.
- **Cost scales linearly, wall time doesn't.** Adding more GPUs reduces wall time but total cost stays the same (minus fixed overhead).

## Importing pre-embedded datasets

Some datasets on HuggingFace are already embedded (e.g. Cohere Wikipedia). Use `modal_import_cohere.py` to import them to S3 in vectorforge's parquet format:

```bash
# Import all languages
modal run modal_import_cohere.py

# Import specific languages
modal run modal_import_cohere.py --configs en,de,fr

# Preview
modal run modal_import_cohere.py --dry-run
```

Each language config runs as a separate Modal container, streaming from HuggingFace and uploading to S3 organized by language subfolder:

```
s3://bucket/cohere--wikipedia/embed-multilingual-v3/
  en/batch_000000.parquet
  en/batch_000001.parquet
  de/batch_000000.parquet
  ...
```
