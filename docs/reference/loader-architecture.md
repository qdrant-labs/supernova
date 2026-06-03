# Loader Architecture

## Overview

The loader streams pre-embedded parquet data from S3 or HuggingFace into vector stores (Qdrant, etc.). It's designed for bulk loading terabyte-scale datasets with minimal memory usage.

```
Source (S3/HF parquet) → DuckDB streaming → async upserts → Vector Store (Qdrant)
```

Three CLI tools operate at different scales:

| Command | What it does | When to use |
|---------|-------------|-------------|
| `nova load` | Single-machine loader | Dev, small datasets, single VM |
| `nova load-dist` | Fan out across SkyPilot spot instances | Large datasets (100GB+) |

## Module Structure

```
supernova/loader/
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

### Single machine (`nova load`)

```
1. DuckDB fetchmany(prefetch_size)     # large read from S3/HF, one I/O op
2. Slice into upsert batches            # in-memory, no network
3. asyncio tasks with semaphore          # concurrent writes to vector store
4. Repeat until exhausted
```

### Distributed (`nova load-dist`)

```
Master (your laptop):
  1. boto3 list_objects → discover parquet files
  2. Round-robin assign files to N shards
  3. Generate per-shard YAML configs (paper trail in ~/.nova/runs/<run_id>/)
  4. Create Qdrant collection + defer indexing
  5. sky jobs launch --async × N (env vars injected, not written to disk)
  6. Poll sky jobs queue until all complete
  7. Enable indexing → wait for HNSW build → report

Workers (SkyPilot spot instances):
  - Run nova load --no-manage-indexing
  - Read only their assigned parquet files (file_list)
  - Upsert to shared Qdrant collection
  - No indexing lifecycle — master handles that
```

## DataReader

Base class for all parquet data sources. Handles DuckDB connection, SQL generation, batch streaming, and column/payload mapping.

### Key concepts

**`id_expression`** — DuckDB SQL expression that yields the point id per row. Defaults to the bare column name `row_id` (works if your parquets carry a `row_id` column). Any DuckDB expression that returns `UBIGINT` or a UUID string is valid:

- `row_id` — use a pre-baked id column
- `hash(text)` — content-deduplicated ids
- `uuid()` — random per-row UUIDs
- `vf_point_id(filename, file_row_number)` — **recommended** for supernova-produced corpora; matches `nova brute-force` and `nova generate-queries` so recall ground truth aligns

The base reader (`supernova/loader/datasource/base.py`) registers three macros at connection time:

- `vf_uuid_from_hex(h)` — formats a 32-char hex string as a canonical UUID.
- `make_point_id(source_file, source_row)` — `vf_uuid_from_hex(md5(source_file || ':' || source_row))`. Mirrors `supernova.utils.make_point_id` exactly.
- `vf_point_id(fname, rnum)` — `make_point_id(substr(fname, prefix_len + 1), rnum)`, where `prefix_len` is the URI-prefix length each subclass declares via `_root_uri_prefix()`.

If the `id_expression` references `filename` or `file_row_number`, the base reader auto-injects `read_parquet(..., filename=true, file_row_number=true)` so those virtual columns are available. **Use `file_row_number`, not `ROW_NUMBER() OVER (PARTITION BY filename)`** — `file_row_number` is the physical row index and is stable under DuckDB's parallel parquet scan; window-function row numbers reflect scan order and can produce different IDs from the brute-force side. There's an explicit regression test in `tests/test_loader_id_expression.py` that documents both behaviours.

**`vectors`** (top-level config) — declares one or more named vectors. Passed to both the reader (which knows the parquet column to read) and the vector store (which knows how to configure the collection):

```yaml
vectors:
  dense:
    type: dense
    column: dense_embedding
    distance: cosine            # cosine | dot | euclid | manhattan
  sparse:
    type: sparse
    column: sparse_embedding
  colbert:
    type: multivector
    column: multivector_embedding
    distance: cosine
    comparator: max_sim
```

Each entry's key becomes the vector name in Qdrant. Records emitted by `read_batches` carry `vectors: {name: value}` rather than a single `embedding`.

**`payload_fields`** — controls what goes into the vector store payload:

```yaml
payload_fields:
  abstract: text      # parquet "text" col → stored as "abstract"
  source: source      # parquet "source" col → stored as "source"
  metadata: payload   # JSON string columns that parse to a dict are unpacked
```

Default: `{}` (no payload).

**`source_sql`** — the DuckDB FROM clause. Defaults to `'<glob_path>'` but subclasses override it with `read_parquet([...])` when `file_list` is set. This is how distributed workers read only their assigned files.

**`_root_uri_prefix()`** — the URI prefix the base class strips from DuckDB's `filename` column to recover the bare key passed into `make_point_id`. Each subclass declares this once (`f"s3://{bucket}/"` for S3, `f"hf://datasets/{repo_id}/"` for HF) and the base class registers `vf_point_id` automatically. Both this loader and the brute-force / generate-queries pipelines must agree on the bare-key form, otherwise IDs won't match — see `supernova/destinations.py:bare_key_for_uri`.

### S3DataReader

- Glob path: `s3://{bucket}/{prefix}/**/*.parquet`
- Configures DuckDB httpfs with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`
- Supports `file_list: list[str]` for explicit file assignment (used by distributed dispatch)

### HuggingFaceDataReader

- Glob path: `hf://datasets/{repo_id}/**/*.parquet`
- Streams directly via DuckDB's native HF protocol (no local download)
- Optional `subdir` to scope to a subfolder
- Requires `HF_TOKEN` env var for authenticated access

Note: the embed-side storage backend now writes to `hf://buckets/...` (HF Storage Buckets), but the loader still reads from `hf://datasets/...` because DuckDB's `httpfs` extension only supports the dataset/space form of `hf://` ("DuckDB only supports querying datasets or spaces"). Legacy corpora already in dataset repos load fine; new corpora written to buckets cannot be loaded into Qdrant via `nova load` until DuckDB adds bucket support upstream.

## ID space anchoring

Point IDs are `md5(bare_key + ":" + row_index)` as a UUID. The same `bare_key` form is computed by *three* places that must agree: the loader's `vf_point_id` macro (when writing to Qdrant), `generate-queries` (when stamping `__source_file__` on sampled rows), and `brute-force` (when emitting hit IDs). If any of those drifts, recall@k breaks silently — payloads still match but UUIDs don't.

The bare key is anchored at the **top-level container**: the S3 bucket, or the HF dataset repo. So:

- `s3://bucket/prefix/path/file.parquet` → bare key `prefix/path/file.parquet`
- `hf://datasets/ns/repo/data/path/file.parquet` → bare key `data/path/file.parquet`

Two consequences fall out of this anchor choice:

**Stable across scope within a container.** Loading just `s3://b/fineweb/cc-2025-26/...` and loading the wider `s3://b/fineweb/...` produce the *same* IDs for the same physical rows. You can do incremental or partial loads, then later widen the scope, without invalidating earlier ground-truth or fragmenting the ID space.

**Reset across containers.** Migrating S3 bucket A → S3 bucket B, or S3 → HF, changes the anchor and therefore the IDs. The recall ground-truth (`queries_*.parquet`, `brute_force_*.parquet`) must be regenerated on the new side; you cannot reuse a Qdrant collection across container migrations.

This is a deliberate trade-off, not an oversight. There is no unforced way to make a hash function span containers — they're literally different ID universes — without introducing an external "logical dataset name" registry that someone has to set correctly per run. The current scheme prioritises the workflow that's actually common (scoped loads within one bucket/repo) over the one that's rare (cross-backend migrations). The seam where this is implemented is `DataReader._root_uri_prefix()` and `supernova.destinations.bare_key_for_uri()` — both must strip the same prefix.

## VectorStore

### Lifecycle methods

| Method | Purpose | Called by |
|--------|---------|----------|
| `ensure_collection(dimensions)` | Create collection if it doesn't exist (dimensions: dict[name, int] for dense+multivector) | Master or single-machine loader |
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

`nova load` and `nova load-dist` consume the **same** config file from `configs/loader/`. The `dispatch:` and `resources:` blocks are read by `nova load-dist` only and ignored by the single-machine loader.

```yaml
vectors:                      # required, at least one entry
  dense:
    type: dense               # dense | sparse | multivector
    column: dense_embedding
    distance: cosine          # dense/multivector: cosine | dot | euclid | manhattan
    # multivector only:
    # comparator: max_sim

datasource:
  type: s3                    # s3 | huggingface
  # S3 options
  bucket: my-bucket
  prefix: my-dataset
  # HuggingFace options
  repo_id: org/dataset
  subdir: en                  # optional subfolder
  # Common options
  id_expression: "vf_point_id(filename, file_row_number)"   # DuckDB SQL; default: "row_id"
  payload_fields:             # optional payload composition
    text: text
  file_list:                  # optional explicit file list (used by dispatch workers)
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

# Optional, only consumed by `nova load-dist`:
dispatch:
  num_shards: 10              # number of parallel SkyPilot workers
  run_name: my-run            # optional, defaults to config filename

resources:                    # SkyPilot VM spec
  cpus: 8
  memory: 32
  cloud: aws
  use_spot: true
```

## Adding a new component

See [Extending supernova](extending.md) for concrete walkthroughs of:

- Adding a raw input source (e.g. Common Crawl)
- Adding a new corpus backend (e.g. GCS — covers `Destination`, `StorageBackend`, and `DataReader` together)
- Adding a new vector store (e.g. Weaviate)
