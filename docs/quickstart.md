# Quickstart

This guide walks through embedding a small dataset and loading it into Qdrant.

## 1. Embed a dataset

Create a config file `configs/embedder/my_dataset.yaml`:

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
vectorforge configs/embedder/my_dataset.yaml
```

This streams the dataset from HuggingFace, embeds each text with OpenAI, and uploads parquet files to S3.

For larger datasets, use SkyPilot to parallelize across GPU instances:

```bash
vectorforge-embed-distributed configs/embedder/my_dataset.yaml
```

## 2. Verify the output

Query the parquet files directly with DuckDB:

```bash
uv run python -c "
import duckdb
con = duckdb.connect()
con.execute('INSTALL httpfs; LOAD httpfs;')
# configure S3 creds if needed
print(con.sql(\"SELECT count(*) FROM 's3://my-bucket/tweet-sentiment/openai-3-small/**/*.parquet'\"))
"
```

## 3. Load into Qdrant

Create a loader config `configs/loader/my_dataset.yaml`:

```yaml
datasource:
  type: s3
  s3_bucket: my-bucket
  s3_prefix: tweet-sentiment/openai-3-small
  payload_fields:
    text: text
    source: source

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
vectorforge-load configs/loader/my_dataset.yaml
```

You'll see a progress bar as points are upserted:

```
Loading:  45%|██████████████████▎                      | 4,500/10,000 [00:12<00:15, 360 pts/s]
```

The loader automatically:
- Creates the Qdrant collection if it doesn't exist
- Defers HNSW indexing during the load for speed
- Enables indexing and waits for the HNSW graph to build after all data is loaded

## 4. Query Qdrant

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="your-key")

results = client.query_points(
    collection_name="tweet-sentiment",
    query=[0.1, 0.2, ...],  # your query vector
    limit=10,
)
```

## Next steps

- [Embedding Generation](embedding-generation.md) -- configuration reference, SkyPilot at scale, dense/sparse embedder options
- [Data Loading](data-loading.md) -- column mapping, payload composition, distributed loading with SkyPilot
