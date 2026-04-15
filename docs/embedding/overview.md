# Embedding Overview

vectorforge's embedding pipeline streams data from a source, embeds text with configurable dense and/or sparse models, and writes the results as parquet files to S3 or HuggingFace Hub.

![Embedding Pipeline](../fig/embedding_pipeline.svg)

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
  dtype: bfloat16

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

You must specify at least one of `dense_embedder` or `sparse_embedder`. See [Dense Embedders](dense-embedders.md) and [Sparse Embedders](sparse-embedders.md) for details.

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

## Storage backends

### S3

```yaml
storage:
  type: s3
  s3_bucket: my-bucket
  s3_prefix: dataset-name/model-name
```

Each chunk produces one parquet file, uploaded as `batch_00000000.parquet`, `batch_00000001.parquet`, etc. Auto-creates the bucket if it doesn't exist.

### HuggingFace Hub

```yaml
storage:
  type: hf
  repo_id: your-org/dataset-name--model-name
  private: true
```

### Local

```yaml
storage:
  type: local
  output_dir: /tmp/vectorforge
```

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
