# Loading Overview

vectorforge's loading pipeline streams pre-embedded parquet files from S3 or HuggingFace into vector stores like Qdrant. An embedding run typically produces many parquet files (one per chunk/slice) under a shared S3 prefix -- the loader reads all of them. It uses DuckDB for efficient remote parquet reads and async concurrency for parallel upserts.

![Loading Pipeline](../fig/ingestion_pipelione.svg)

## Configuration

Loader configs live in `configs/loader/` and have three sections:

```yaml
datasource:
  type: s3                          # s3 or huggingface
  s3_bucket: my-bucket
  s3_prefix: dataset/model
  payload_fields:                   # what ends up in the vector store payload
    text: text                      # payload key: parquet column name
    source: source

vectorstore:
  type: qdrant
  collection_name: my-collection
  url: ${QDRANT_URL}                # env var substitution with ${VAR}
  api_key: ${QDRANT_API_KEY}

loader:
  batch_size: 1000                  # points per upsert call
  prefetch_size: 100000             # rows per DuckDB fetch
  concurrency: 8                    # parallel upsert tasks
```

## Running

```bash
vf load configs/loader/my_dataset.yaml
```

## Datasources

### S3

Streams parquet files via DuckDB's httpfs extension. No local download.

```yaml
datasource:
  type: s3
  s3_bucket: my-bucket
  s3_prefix: cohere--wikipedia/embed-multilingual-v3
```

Reads all parquet files matching `s3://bucket/prefix/**/*.parquet`.

### HuggingFace

Streams directly from HuggingFace Hub via DuckDB's `hf://` protocol.

```yaml
datasource:
  type: huggingface
  repo_id: CohereLabs/wikipedia-2023-11-embed-multilingual-v3
  subdir: en
```

## Column mapping

Override default column names:

```yaml
datasource:
  columns:
    id: _id                         # parquet column "_id" is the point ID
    embedding: dense_embedding      # parquet column for the vector
```

| Logical name | Default | Used for |
|-------------|---------|----------|
| `id` | `row_id` | Point ID in the vector store |
| `embedding` | `embedding` | The vector |

## Payload composition

`payload_fields` controls what data gets stored alongside each vector:

```yaml
payload_fields:
  text: text              # store parquet "text" column as "text" in payload
  abstract: text          # ...or rename it to "abstract"
  source: source
  url: url
```

JSON string columns are automatically unpacked. Default when omitted: `{text: text}`.

## How it works

1. **DuckDB streams** parquet data in large chunks (`prefetch_size` rows per fetch)
2. **Chunks are sliced** into `batch_size` upsert batches locally
3. **Async upserts** run concurrently, controlled by a semaphore (`concurrency`)
4. **Deferred indexing** -- HNSW construction is disabled during load, then built in one pass
5. **Retry with backoff** -- failed upserts are retried up to 3 times

## Tuning

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `batch_size` | 1000 | Larger = fewer HTTP calls. 1000 is good for 768-1024 dim vectors. |
| `prefetch_size` | batch_size * 10 | Larger = fewer S3 round trips. 100k works well. |
| `concurrency` | 8 | Lower if you're getting timeouts. |
