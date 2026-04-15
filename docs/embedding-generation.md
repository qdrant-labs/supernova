# Embedding Generation

vectorforge's embedding pipeline streams data from a source, embeds text with configurable dense and/or sparse models, and writes the results as parquet files to S3 or HuggingFace Hub.

![Embedding Pipeline](fig/embedding_pipeline.svg)

The dataset is split into N chunks. Each chunk is processed independently -- either locally (async workers) or on SkyPilot GPU instances (one per chunk). Each chunk produces a parquet file that gets uploaded to storage.

## Configuration

Embedding configs live in `configs/embedder/` and have four sections:

```yaml
source:
  type: huggingface
  dataset_name: nick007x/arxiv-papers
  split: train
  text_field: abstract             # single field to embed
  # text_template: "{title}: {abstract}"  # or a format string

dense_embedder:
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

You must specify at least one of `dense_embedder` or `sparse_embedder`. When both point to the same model (same type and model name), vectorforge automatically uses a hybrid encoder that produces both in fewer forward passes.

### Column naming

The embedding columns default to `dense_embedding` and `sparse_embedding`. Override with:

```yaml
pipeline:
  dense_embedding_column: my_dense   # default: dense_embedding
  sparse_embedding_column: my_sparse # default: sparse_embedding
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

## Dense embedders

### OpenAI

Uses the OpenAI API. Best for smaller datasets or when you don't have GPUs.

```yaml
dense_embedder:
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
dense_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  trust_remote_code: true
  batch_size: 64
  dtype: bfloat16                  # float32, float16, bfloat16
```

- Auto-detects CUDA, MPS (Apple Silicon), or CPU
- Supports bfloat16/float16 for faster inference
- Text splitting via the model's tokenizer

## Sparse embedders

### Sentence Transformers (SparseEncoder)

Uses sentence-transformers' `SparseEncoder` for models like SPLADE and gte-multilingual-base:

```yaml
sparse_embedder:
  type: sentence_transformer
  model: Alibaba-NLP/gte-multilingual-base
  batch_size: 64
  dtype: bfloat16
```

Sparse embeddings are stored as a struct with two parallel arrays (`indices` and `values`) in the output parquet files.

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
| `dense_embedding` | list\<float32\> | Dense embedding (when configured) |
| `sparse_embedding` | struct{indices, values} | Sparse embedding (when configured) |

Query with DuckDB:

```sql
SELECT row_id, text[:80] AS preview, length(dense_embedding) AS dim
FROM 's3://my-bucket/dataset/model/**/*.parquet'
LIMIT 10;
```

## Running locally

```bash
vf embed configs/embedder/my_dataset.yaml
```

The local runner uses async workers with a priority queue buffer to ensure ordered output. Good for development and small datasets.

## Running at scale with SkyPilot

SkyPilot pools create a set of GPU workers and distribute embedding jobs across them. Workers are reused across jobs -- setup (uv sync, model download) happens once, not per-slice.

```bash
# Preview the plan
vf embed-dist configs/embedder/arxiv_papers.yaml --dry-run

# Run with default resources (A10G spot)
vf embed-dist configs/embedder/arxiv_papers.yaml

# Custom number of jobs
vf embed-dist configs/embedder/arxiv_papers.yaml --num-jobs 20

# Custom chunk size (smaller = more parallelism)
vf embed-dist configs/embedder/arxiv_papers.yaml --chunk-size 50000

# Named pool (for reuse across runs)
vf embed-dist configs/embedder/arxiv_papers.yaml --pool-name my-gpu-pool
```

### How it works

1. **Plan** (runs locally): reads config, queries HuggingFace for dataset size
2. **Pool**: creates a SkyPilot pool with autoscaling GPU workers (`min_workers: 0`, `max_workers: N`)
3. **Submit**: submits N jobs to the pool via `sky jobs launch --num-jobs N`
4. **Each job**: SkyPilot sets `$SKYPILOT_JOB_RANK` and `$SKYPILOT_NUM_JOBS`. The `vf` CLI uses these to auto-compute offset/limit and process its slice
5. **Autoscale**: workers scale up to handle the queue, scale back to zero when done

### Custom resources

Add a `resources` section to your config to override SkyPilot VM specs:

```yaml
resources:
  accelerators: A10G:1
  cloud: aws
  use_spot: true
```

Default: A10G GPU on AWS spot instances.

### Manual pool usage

You can also manage pools directly:

```bash
# Create a pool
sky jobs pool apply -p my-pool runs/<run-dir>/pool.yaml

# Submit jobs
sky jobs launch -p my-pool --num-jobs 20 runs/<run-dir>/job.yaml

# Monitor
sky jobs pool status my-pool

# View logs
sky jobs pool logs my-pool

# Tear down
sky jobs pool down my-pool
```

### Generated artifacts

Each run creates a directory:

```
runs/2026-04-13T14-30_vf-embed-en/
  pool.yaml                  # pool config (resources, setup)
  job.yaml                   # job config (run command)
  manifest.json              # plan metadata
```

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
total_tokens = rows * mean_tokens_per_row
```

The mean already accounts for skewed distributions (most datasets have a long tail of very long texts). For a confidence interval on the estimate:

```
standard_error = stdev / sqrt(sample_size)
total_tokens = rows * mean +/- rows * 1.96 * standard_error  (95% CI)
```

With a 100K sample, the standard error is typically small enough to ignore.

### Step 3: Estimate time and cost

```
gpu_hours = total_tokens / throughput_tok_s / 3600
cost      = gpu_hours * price_per_gpu_hour
wall_time = gpu_hours / num_gpus
```

### Assumptions and reference values

| Parameter | Value | Notes |
|-----------|-------|-------|
| Throughput (gte-multilingual-base, A10G, bfloat16) | 50,000 tok/s | Budget estimate; raw benchmarks show 45K-87K depending on text length |
| Throughput (snowflake-arctic-embed-l-v2.0, A10G, bfloat16) | ~35,000 tok/s | Single GPU, benchmarked on real data |
| A10G price (AWS) | ~$0.38/hr | Spot, g5.xlarge |
| T4 price (GCP) | ~$0.18/hr | Spot |
| OpenAI text-embedding-3-small | $0.02/1M tokens | API pricing, no GPU needed |

Throughput varies with text length: short texts (~20 tok avg) achieve ~58K tok/s due to padding waste, while medium-to-long texts (~700+ tok avg) hit ~80-87K tok/s. Always use bfloat16 -- it's ~2x faster than float32 with no quality loss.

### Example

finewiki (English): 1.82M rows, mean 676 tok/row:

```
total_tokens = 1,820,000 * 676 = 1.23B tokens
gpu_hours    = 1,230,000,000 / 50,000 / 3600 = 6.8 GPU-hours
cost (AWS)   = 6.8 * $0.38 = $2.60  (spot g5.xlarge)
wall_time    = 6.8 / 10 GPUs = 41 min
```

Compare with OpenAI API:

```
cost (OpenAI) = 1,230,000,000 / 1,000,000 * $0.02 = $24.60
```

Self-hosted GPU embedding is ~10x cheaper than OpenAI at this scale. The cost gap widens with larger datasets.

### How much compute are you wasting on padding?

Transformer models pad every batch to the length of the longest sequence. If your dataset has a heavy tail (most do), a single long text forces an entire batch to pad to its length -- wasting the majority of GPU compute on empty tokens.

Run the padding simulator to find out:

```bash
# analyze any dataset -- no GPU needed, runs on CPU in ~2 min
python throughput_exp/padding_sim.py --dataset HuggingFaceFW/finewiki --hf-config en

# outputs: padding efficiency heatmap, tradeoff plot, and JSON results
```

This samples 100K rows, fits the token length distribution, and runs a Monte Carlo simulation across truncation cutoffs and batch sizes. The output tells you exactly what percentage of your GPU is spent on real tokens vs padding at each configuration.

Typical findings on real datasets:
- **No truncation** (cutoff=8192): 15-25% efficiency -- 75-85% of compute is wasted
- **Cutoff=1024**: ~70% efficiency with ~70% of content retained
- **Cutoff=512**: ~87% efficiency with ~50% of content retained

Set `pipeline.max_text_length` in your config to apply a truncation cutoff. For most retrieval use cases, 512-1024 tokens captures the semantically dense portion of documents while avoiding the long tail that destroys throughput.

See `throughput_exp/ANALYSIS.md` for the full methodology and benchmark results.

### Caveats

- **Throughput varies by model and GPU.** Measure your own throughput from a test run. Use `throughput_exp/bench.py` to sweep batch sizes, dtypes, and cutoffs.
- **Long texts create multiple chunks.** A 50K-token text becomes ~6 chunks at 8192 max_seq_length. The total tokens already accounts for this since we measure at the source row level.
- **Use `max_text_length` to cap outliers.** Setting `pipeline.max_text_length` in your config prevents the long tail from dominating compute.
- **GPU utilization matters.** The formula assumes 100% utilization. Real runs have overhead from data loading, uploads, and container startup. Add ~20% buffer.
- **Cost scales linearly, wall time doesn't.** Adding more GPUs reduces wall time but total cost stays the same (minus fixed overhead).
