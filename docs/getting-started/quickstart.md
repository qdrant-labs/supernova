# Quickstart

Embed a dataset, load it into Qdrant, and load-test it.

## 1. Embed a dataset

Create `configs/embedder/my_dataset.yaml`:

```yaml
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train
  text_field: text

dense_embedder:
  type: sentence_transformer
  model: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 128

storage:
  type: local
  output_dir: /tmp/tweets

pipeline:
  chunk_size: 1000
  flush_threshold: 2000
```

Run it:

```bash
nova embed configs/embedder/my_dataset.yaml
```

`nova embed --dry-run <config>` prints the resolved plan without running. For a
big dataset, shard it across machines with `--num-jobs` / `--job-rank` (see
[Distributed](#distributed)).

## 2. Load into Qdrant

Create `configs/loader/my_dataset.yaml`:

```yaml
datasource:
  type: local
  path: /tmp/tweets
  id_expression: "vf_point_id(filename, file_row_number)"
  payload_fields:
    text: text

vectors:                 # which parquet column feeds each named vector
  dense:
    type: dense
    column: dense_embedding
    distance: cosine

vectorstore:
  type: qdrant
  collection_name: tweets
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}

loader:
  batch_size: 256
  concurrency: 8
  file_look_ahead: 2
```

The top-level `vectors:` block tells the loader which parquet column carries each
vector and tells Qdrant how to configure the collection. `id_expression` is a
DuckDB SQL expression yielding the point id per row; `vf_point_id(filename,
file_row_number)` produces stable, deterministic UUIDs.

Run it (single machine — creates the collection, loads, builds the index):

```bash
export QDRANT_URL=https://your-cluster.qdrant.io
export QDRANT_API_KEY=your-key
nova load run configs/loader/my_dataset.yaml
```

`nova load inspect <config>` shows the file list and resolved config without
connecting or loading.

## 3. Load-test it

Create `configs/storm/my_dataset.yaml`:

```yaml
target:
  type: qdrant
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}
  collection_name: tweets

query:
  vector_name: dense
  top_k: 10
  source:
    uri: /tmp/tweets/part-0000.parquet   # reuse some vectors as queries
    column: dense_embedding
    limit: 1000

load:
  concurrency: 32
  duration_s: 60
  # batch_size: 8   # queries per query_batch dispatch (1 = one query per round-trip)
  # rps: 500        # omit for closed-loop (max throughput); set for paced open-loop
```

```bash
nova storm configs/storm/my_dataset.yaml
```

It prints a latency summary (p50/p95/p99, throughput) at the end.

## Distributed

Each tool partitions its own work; a fleet is N copies with a rank. Your
orchestrator (SkyPilot, etc.) provisions nodes and runs the same command on each,
passing that node's `--job-rank`.

**Embed** — each rank embeds its slice of the dataset:

```bash
nova embed configs/embedder/my_dataset.yaml --num-jobs 50 --job-rank $SKYPILOT_JOB_RANK
```

**Load** — one master prepares + finalizes; every worker loads its slice of the
files:

```bash
# master, once:
nova load prepare configs/loader/my_dataset.yaml
# every worker:
nova load load    configs/loader/my_dataset.yaml --num-jobs 50 --job-rank $RANK
# master, after all workers finish:
nova load finalize configs/loader/my_dataset.yaml
```

(`nova load run` is the single-machine shorthand for prepare + load + finalize.)

**Storm** — replicated, not partitioned: every worker runs the *same* profile, so
total offered load ≈ `num_workers × concurrency`.
