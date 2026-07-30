# Quickstart

Embed a dataset, load it into Qdrant, load-test it, then run the new web service
for API-driven operations. In this example, we embed the
`mteb/tweet_sentiment_extraction` dataset with a sentence-transformer. This is a
small dataset, and we embed the `text` field. The embedding is stored in parquet
files, which are then loaded into a Qdrant collection. Finally, we load-test the
collection with queries drawn from the same dataset and start `nova web`.

## 1. Embed a dataset

Create `configs/embedder/my_dataset.yaml`:

```yaml
# get data from HuggingFace datasets hub
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train

# each entry declares WHAT it embeds (input_column + modality) and WHERE the
# result lands (output_column, default "{name}_embedding")
embedders:
  - name: dense
    kind: dense
    type: sentence_transformer
    model: sentence-transformers/all-MiniLM-L6-v2
    input_column: text
    modality: text
    batch_size: 32

  - name: sparse
    kind: sparse
    type: fastembed
    model: Qdrant/bm25
    input_column: text
    modality: text
    batch_size: 256

storage:
  type: local
  output_dir: /tmp/mteb_tweets

pipeline:
  flush_threshold: 10000
```

Run it:

```bash
nova embed configs/embedder/my_dataset.yaml
```

`nova embed --dry-run <config>` prints the resolved plan without running. For a
big dataset, shard it across machines with `--num-jobs` / `--job-rank` (see
[Distributed](#distributed)).

Once finished, the parquet files are in `/tmp/mteb_tweets`. Each file has a
`dense_embedding` column (384 floats) and a `sparse_embedding` column (a
sparse vector) — one column per embedder entry, named `{name}_embedding` by
default. The `text` field and every other source column are preserved in the
parquet files, so we can use them as payload in the vector store.

## 2. Load into Qdrant

Create `configs/loader/my_dataset.yaml`:

```yaml
datasource:
  type: local
  path: /tmp/mteb_tweets
  id: id
  payload_fields:
    text: text
    label: label
    label_text: label_text

vectors:                 # which parquet column feeds each named vector
  dense:
    type: dense
    column: dense_embedding
    distance: cosine
  sparse:
    type: sparse
    column: sparse_embedding

vectorstore:
  type: qdrant
  collection_name: mteb_tweets
  url: ${QDRANT_URL:-http://localhost:6334}
  api_key: ${QDRANT_API_KEY:-}

loader:
  batch_size: 256
  concurrency: 2
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
  collection_name: mteb_tweets

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

## 4. Run the web service (`nova web`)

Build the Angular frontend once, then start the Axum backend:

```bash
cd web/supernova-dashboard
npm install
npm run build

cd ../..
cargo run -p nova-web
```

Open `http://localhost:8080` to use the dashboard.

Useful API endpoints:

```bash
curl -s http://localhost:8080/health
curl -s http://localhost:8080/api/v1/jobs
```

Submit a `nova load run` job by passing config path or inline YAML:

```bash
curl -X POST http://localhost:8080/api/v1/load/run \
  -H 'content-type: application/json' \
  -d '{"config_path":"configs/loader/my_dataset.yaml"}'
```

Distributed orchestration via SkyPilot is also exposed:

```bash
curl -X POST http://localhost:8080/api/v1/dist/storm \
  -H 'content-type: application/json' \
  -d '{"config_path":"configs/storm/my_dataset.yaml","num_jobs":4,"dry_run":true}'
```

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
