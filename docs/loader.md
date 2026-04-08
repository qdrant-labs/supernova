# Loader Architecture

## Overview

The loader streams pre-embedded parquet data from S3 or HuggingFace into vector stores (Qdrant, etc.). It's designed for bulk loading terabyte-scale datasets with minimal memory usage.

```
Source (S3/HF parquet) → DuckDB streaming → async upserts → Vector Store (Qdrant)
```

Three CLI tools operate at different scales:

| Command | What it does | When to use |
|---------|-------------|-------------|
| `vectorforge-load` | Single-machine loader | Dev, small datasets, single VM |
| `vectorforge-load-distributed` | Fan out across SkyPilot spot instances | Large datasets (100GB+) |

## Module Structure

```
vectorforge/loader/
├── datasource/
│   ├── base.py            # DataReader ABC — DuckDB streaming, batch iteration
│   ├── s3.py              # S3DataReader — httpfs, AWS creds, file_list support
│   └── huggingface.py     # HuggingFaceDataReader — hf:// protocol
├── vectorstore/
│   ├── base.py            # VectorStore ABC — upsert, indexing lifecycle
│   └── qdrant.py          # QdrantVectorStore — async client, deferred indexing
└── runner.py              # Async orchestrator — prefetch, slice, concurrent upsert
```

## Data Flow

### Single machine (`vectorforge-load`)

```
1. DuckDB fetchmany(prefetch_size)     # large read from S3/HF, one I/O op
2. Slice into upsert batches            # in-memory, no network
3. asyncio tasks with semaphore          # concurrent writes to vector store
4. Repeat until exhausted
```

### Distributed (`vectorforge-load-distributed`)

```
Master (your laptop):
  1. boto3 list_objects → discover parquet files
  2. Round-robin assign files to N shards
  3. Generate per-shard YAML configs (paper trail in runs/<run_id>/)
  4. Create Qdrant collection + defer indexing
  5. sky jobs launch --async × N (env vars injected, not written to disk)
  6. Poll sky jobs queue until all complete
  7. Enable indexing → wait for HNSW build → report

Workers (SkyPilot spot instances):
  - Run vectorforge-load --no-manage-indexing
  - Read only their assigned parquet files (file_list)
  - Upsert to shared Qdrant collection
  - No indexing lifecycle — master handles that
```

## DataReader

Base class for all parquet data sources. Handles DuckDB connection, SQL generation, batch streaming, and column/payload mapping.

### Key concepts

**`columns`** — maps logical names to parquet column names:

```python
columns = {
    "id": "row_id",        # default
    "embedding": "embedding" # default
}
```

Override when your parquet has different column names:

```yaml
columns:
  id: _id
  embedding: emb
```

**`payload_fields`** — controls what goes into the vector store payload:

```yaml
payload_fields:
  abstract: text      # parquet "text" col → stored as "abstract"
  source: source      # parquet "source" col → stored as "source"
  metadata: payload   # JSON string columns are auto-unpacked
```

Default: `{text: text}`.

**`source_sql`** — the DuckDB FROM clause. Defaults to `'<glob_path>'` but S3DataReader overrides it with `read_parquet([...])` when `file_list` is set. This is how distributed workers read only their assigned files.

### S3DataReader

- Glob path: `s3://{bucket}/{prefix}/**/*.parquet`
- Configures DuckDB httpfs with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`
- Supports `file_list: list[str]` for explicit file assignment (used by distributed dispatch)

### HuggingFaceDataReader

- Glob path: `hf://datasets/{repo_id}/**/*.parquet`
- Streams directly via DuckDB's native HF protocol (no local download)
- Optional `subdir` to scope to a subfolder
- Requires `HF_TOKEN` env var for authenticated access

## VectorStore

### Lifecycle methods

| Method | Purpose | Called by |
|--------|---------|----------|
| `ensure_collection(dim)` | Create collection if it doesn't exist | Master or single-machine loader |
| `defer_indexing()` | Set indexing_threshold=0 for fast bulk writes | Master or single-machine loader |
| `upsert_batch(points)` | Insert points into collection | Every worker |
| `enable_indexing()` | Restore threshold, trigger HNSW build | Master or single-machine loader |
| `wait_for_indexing()` | Poll until collection status is GREEN | Master or single-machine loader |
| `close()` | Clean up connections | Everyone |

### Deferred indexing

The key optimization for bulk loading. Without it, every upsert triggers incremental HNSW graph updates (expensive). With deferred indexing:

1. `defer_indexing()` → Qdrant stores vectors flat, no graph construction
2. Blast data in as fast as possible (parallel upserts)
3. `enable_indexing()` → Qdrant builds HNSW in one efficient batch pass
4. `wait_for_indexing()` → block until complete

This is dramatically faster for bulk loads.

### QdrantVectorStore

- Uses `qdrant-client` AsyncQdrantClient
- `defer_indexing()` sets `indexing_threshold=0` via `update_collection`
- `enable_indexing()` sets `indexing_threshold=20000` (Qdrant default)
- `wait_for_indexing()` polls `get_collection` every 5s until status is GREEN
- Supports scalar (INT8) and binary quantization via `params`

## Runner

`run_loader()` is the core async function. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1000 | Points per upsert call |
| `prefetch_size` | batch_size * 10 | Rows per DuckDB fetch (reduces remote I/O) |
| `concurrency` | 8 | Max parallel upsert tasks (semaphore) |
| `manage_indexing` | True | If False, skip collection creation and indexing lifecycle |

### Prefetch strategy

DuckDB reads `prefetch_size` rows per fetch — one remote I/O operation that reads full parquet row groups. Those rows are sliced into `batch_size` upsert batches locally (no network). This minimizes S3/HF round trips for large datasets.

```
prefetch_size=100,000  →  DuckDB reads 100k rows (one S3 request)
batch_size=1,000       →  sliced into 100 upsert batches (local)
concurrency=8          →  8 upserts running in parallel
```

## Configuration Reference

### Loader config (`vectorforge-load`)

```yaml
datasource:
  type: s3                    # s3 | huggingface
  # S3 options
  s3_bucket: my-bucket
  s3_prefix: my-dataset
  # HuggingFace options
  repo_id: org/dataset
  subdir: en                  # optional subfolder
  # Common options
  columns:                    # optional column name overrides
    id: row_id                # default
    embedding: embedding      # default
  payload_fields:             # optional payload composition
    text: text                # default
  file_list:                  # optional explicit file list (used by dispatch)
    - s3://bucket/file1.parquet
    - s3://bucket/file2.parquet

vectorstore:
  type: qdrant
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}
  collection_name: my-collection

loader:
  batch_size: 1000            # default
  prefetch_size: 10000        # default: batch_size * 10
  concurrency: 8              # default
```

### Dispatch config (`vectorforge-load-distributed`)

Same as loader config, plus:

```yaml
dispatch:
  num_shards: 10              # number of parallel SkyPilot workers
  run_name: my-run            # optional, defaults to config filename

resources:                    # SkyPilot VM spec
  cpus: 8
  memory: 32
  cloud: aws
  use_spot: true
```

## Adding a new vector store

1. Create `vectorforge/loader/vectorstore/my_store.py`
2. Subclass `VectorStore` from `base.py`
3. Implement `ensure_collection`, `upsert_batch`, `close`, `name`
4. Optionally implement `defer_indexing`, `enable_indexing`, `wait_for_indexing`
5. Register in `VECTORSTORE_REGISTRY` in `scripts/run_loader.py`

## Adding a new datasource

1. Create `vectorforge/loader/datasource/my_source.py`
2. Subclass `DataReader` from `base.py`
3. Implement `glob_path` property
4. Optionally override `_configure_connection` for auth setup
5. Register in `DATASOURCE_REGISTRY` in `scripts/run_loader.py`
