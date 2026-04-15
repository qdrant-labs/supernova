# Quickstart

Embed a small dataset and load it into Qdrant.

## 1. Embed a dataset

Create `configs/embedder/my_dataset.yaml`:

```yaml
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train
  text_field: text

dense_embedder:
  type: openai
  model: text-embedding-3-small
  dimensions: 1536
  batch_size: 128

pipeline:
  chunk_size: 10000
  num_workers: 4
  flush_threshold: 100000

storage:
  type: s3
  s3_bucket: my-bucket
  s3_prefix: tweet-sentiment/openai-3-small
  output_dir: /tmp/vectorforge
```

Run locally:

```bash
export OPENAI_API_KEY=sk-...
vf embed configs/embedder/my_dataset.yaml
```

For larger datasets, use SkyPilot to parallelize:

```bash
vf embed-dist configs/embedder/my_dataset.yaml
```

## 2. Verify the output

Query the parquet files with DuckDB:

```bash
uv run python -c "
import duckdb
con = duckdb.connect()
con.execute('INSTALL httpfs; LOAD httpfs;')
print(con.sql(\"SELECT count(*) FROM 's3://my-bucket/tweet-sentiment/openai-3-small/**/*.parquet'\"))
"
```

## 3. Load into Qdrant

Create `configs/loader/my_dataset.yaml`:

```yaml
datasource:
  type: s3
  s3_bucket: my-bucket
  s3_prefix: tweet-sentiment/openai-3-small
  payload_fields:
    text: text

vectorstore:
  type: qdrant
  collection_name: tweet-sentiment
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}

loader:
  batch_size: 1000
  prefetch_size: 100000
  concurrency: 8
```

Run:

```bash
export QDRANT_URL=https://your-cluster.qdrant.io
export QDRANT_API_KEY=your-key
vf load configs/loader/my_dataset.yaml
```

The loader automatically creates the Qdrant collection, defers HNSW indexing during load, and builds the index after all data is loaded.

## 4. Query

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="your-key")

results = client.query_points(
    collection_name="tweet-sentiment",
    query=[0.1, 0.2, ...],  # your query vector
    limit=10,
)
```
