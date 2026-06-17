# Loader Architecture

## Overview

The loader streams pre-embedded parquet data from S3 (or local disk) into vector stores (Qdrant, etc.). It's designed for bulk loading terabyte-scale datasets with minimal memory usage.

```
Source (S3/local parquet) → DuckDB streaming → async upserts → Vector Store (Qdrant)
```

The loader is a **Rust binary**, `nova-load` (crate `crates/nova-load`). The `nova` CLI is a thin dispatcher that locates the binary and execs it (`nova load …` → `nova-load …`); see [the CLI module](../../supernova/cli/cli.py). Two entry points operate at different scales:

| Command | What it does | When to use |
|---------|-------------|-------------|
| `nova load` | Single-machine loader | Dev, small datasets, single VM |
| `nova dist load` | Fan out across SkyPilot spot instances | Large datasets (100GB+) |

> **HF datasources are not yet ported to Rust.** The Rust loader implements `s3` and `local`; `huggingface` is stubbed and errors. Loading `hf://datasets/...` corpora is temporarily unsupported (the previous Python loader handled it). Tracked as a follow-up.

## Crate structure

```
crates/nova-load/src/
├── config.rs              # YAML config + ${VAR} resolution; vector specs → CollectionSchema
├── sources/
│   ├── engine.rs           # DuckDbReader — shared DuckDB streaming engine + id macros
│   ├── s3.rs               # S3 backend — httpfs, AWS creds, file_list support
│   ├── local.rs            # local filesystem backend
│   ├── huggingface.rs      # hf:// backend (NOT YET IMPLEMENTED — stub)
│   └── mod.rs              # DataReader trait, DataSourceConfig, ReaderOptions
├── stores/
│   ├── qdrant.rs           # QdrantVectorStore — upsert, indexing lifecycle
│   └── mod.rs             # VectorStore trait, Point / PointId / VectorValue
├── runner.rs              # producer/consumer orchestrator; setup_collection / finalize
└── main.rs                # clap CLI
```

The seam is two traits: **`SourceBackend`** (per-source hooks — where the files are, how to authenticate) that the shared `DuckDbReader` engine drives, and **`VectorStore`** (the upsert + indexing-lifecycle contract). Adding a backend means implementing one trait, not touching the engine.

## Data Flow

### Single machine (`nova load`)

```
1. DuckDB reads prefetch_size rows        # large read from S3/local, one I/O op (blocking thread)
2. Slice into upsert batches               # in-memory, no network
3. tokio tasks, semaphore-capped            # concurrent writes to the vector store
4. Repeat until exhausted
```

A blocking producer task drives the (synchronous) DuckDB reader and feeds upsert-sized batches into a bounded channel; a pool of async workers, capped by a semaphore, upserts them concurrently. The bounded channel is the backpressure valve — when all workers are busy the producer parks, so memory stays bounded and the achieved rate reflects what the store can sustain.

### Distributed (`nova dist load`)

```
Controller (your laptop / Hetzner — Python orchestrator):
  1. discover_corpus_parquets → list parquet files (for the plan + sharding count)
  2. nova-load --setup-only → create collection + defer indexing (it probes dims itself)
  3. stage the raw config to ~/.nova/runs/<run_id>/; secrets ride as forwarded env vars
  4. sky jobs pool_apply + launch × N
  5. (after all workers complete) nova load --finalize → enable indexing + wait for HNSW

Workers (SkyPilot spot instances):
  - setup: curl the prebuilt nova-load binary from the GitHub Release (statically links qdrant/duckdb)
  - run:   nova-load <cfg> --num-jobs N --no-manage-indexing
  - shard by $SKYPILOT_JOB_RANK → read only their slice of files
  - upsert to the shared Qdrant collection; no indexing lifecycle (controller owns it)
```

The controller is pure orchestration — it owns the Qdrant lifecycle but runs no load logic itself, delegating both the one-time setup and the finalize to the same `nova-load` binary in control-plane mode.

## Reading: DuckDbReader + SourceBackend

The shared `DuckDbReader` engine (`sources/engine.rs`) handles the DuckDB connection, SQL generation, batch streaming, and column/payload mapping. Each backend (`s3.rs`, `local.rs`) supplies a thin `SourceBackend` with the per-source bits.

### Key concepts

**`id_expression`** — DuckDB SQL expression that yields the point id per row. Defaults to the bare column name `row_id` (works if your parquets carry a `row_id` column). Any DuckDB expression that returns `UBIGINT` or a UUID string is valid:

- `row_id` — use a pre-baked id column
- `hash(text)` — content-deduplicated ids
- `uuid()` — random per-row UUIDs
- `vf_point_id(filename, file_row_number)` — **recommended** for supernova-produced corpora; produces deterministic IDs so recall ground truth from the eval pipeline aligns

The engine registers three macros at connection time (`DuckDbReader::register_macros`):

- `vf_uuid_from_hex(h)` — formats a 32-char hex string as a canonical UUID.
- `make_point_id(source_file, source_row)` — `vf_uuid_from_hex(md5(source_file || ':' || source_row))`.
- `vf_point_id(fname, rnum)` — `make_point_id(substr(fname, prefix_len + 1), rnum)`, where `prefix_len` is the URI-prefix length each backend declares via `root_uri_prefix()`.

If the `id_expression` references `filename` or `file_row_number`, the engine auto-injects `read_parquet(..., filename=true, file_row_number=true)` so those virtual columns are available. **Use `file_row_number`, not `ROW_NUMBER() OVER (PARTITION BY filename)`** — `file_row_number` is the physical row index and is stable under DuckDB's parallel parquet scan; window-function row numbers reflect scan order and can produce different IDs from one run to the next. There's a regression test (`filename_derived_ids_are_stable_uuids` in `sources/local.rs`) covering this.

**`vectors`** (top-level config) — declares one or more named vectors. Shared by the reader (which parquet column to read) and the store (how to configure the collection):

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

Each entry's key becomes the vector name in Qdrant; each `Point` the reader emits carries `vectors: {name: value}`.

**`payload_fields`** — controls what goes into the vector store payload:

```yaml
payload_fields:
  abstract: text      # parquet "text" col → stored as "abstract"
  source: source      # parquet "source" col → stored as "source"
  metadata: payload   # JSON string columns that parse to a dict are unpacked
```

Default: `{}` (no payload).

**`source_sql`** — the DuckDB FROM clause. Defaults to the bare glob path but a backend wraps it in `read_parquet([...])` when `file_list` is set. This is how distributed workers read only their assigned files.

**`root_uri_prefix()`** — the URI prefix stripped from DuckDB's `filename` column to recover the bare key fed into `make_point_id`. Each backend declares it once (`s3://{bucket}/` for S3) and the engine registers `vf_point_id` automatically. Both the loader and any recall-eval tooling must agree on the bare-key form, or IDs won't match — the Python side computes it in `supernova/destinations.py:bare_key_for_uri` (used for corpus discovery + eval).

### S3 backend (`sources/s3.rs`)

- Glob path: `s3://{bucket}/{prefix}/**/*.parquet`
- Configures DuckDB httpfs with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`
- Supports `file_list: list[str]` for explicit file assignment (used by distributed sharding)

### Local backend (`sources/local.rs`)

- A directory is read recursively (`**/*.parquet`); an explicit glob or single `.parquet` is used as-is
- No credentials; DuckDB reads local files natively
- Resolves `~` and makes paths absolute so `filename`-derived ids are stable

### HuggingFace backend (`sources/huggingface.rs`)

Not yet implemented — the variant parses but `into_reader` / discovery return an error. The previous Python loader read `hf://datasets/...` directly via DuckDB's HF protocol; that path needs reimplementing in Rust before HF corpora can be loaded.

## ID space anchoring

Point IDs are `md5(bare_key + ":" + row_index)` as a UUID. The `bare_key` form must be computed identically everywhere it's used: the loader's `vf_point_id` macro (when writing to Qdrant) and any recall-eval tooling (when mapping query rows back to corpus point IDs). If the derivations drift, recall@k breaks silently — payloads still match but UUIDs don't.

The bare key is anchored at the **top-level container**: the S3 bucket. So:

- `s3://bucket/prefix/path/file.parquet` → bare key `prefix/path/file.parquet`

Two consequences fall out of this anchor choice:

**Stable across scope within a container.** Loading just `s3://b/fineweb/cc-2025-26/...` and loading the wider `s3://b/fineweb/...` produce the *same* IDs for the same physical rows. You can do incremental or partial loads, then later widen the scope, without invalidating earlier ground-truth or fragmenting the ID space.

**Reset across containers.** Migrating S3 bucket A → S3 bucket B changes the anchor and therefore the IDs. Any recall ground-truth under `eval/` must be regenerated on the new side; you cannot reuse a Qdrant collection across container migrations.

This is a deliberate trade-off, not an oversight. There is no unforced way to make a hash function span containers — they're literally different ID universes — without an external "logical dataset name" registry that someone has to set per run. The current scheme prioritises the common workflow (scoped loads within one bucket) over the rare one (cross-backend migration). The seam is the backend's `root_uri_prefix()` and `supernova.destinations.bare_key_for_uri()` — both must strip the same prefix.

## VectorStore

### Lifecycle methods (`stores/mod.rs`)

| Method | Purpose | Run by |
|--------|---------|--------|
| `ensure_collection(schema)` | Create collection if absent (schema: resolved vector specs + sizes) | `--setup-only` (controller) / single-machine |
| `defer_indexing()` | Set indexing_threshold=0 for fast bulk writes | `--setup-only` (controller) / single-machine |
| `upsert_batch(points)` | Insert a batch of points | Every worker |
| `enable_indexing()` | Restore threshold, trigger HNSW build | `--finalize` (controller) / single-machine |
| `wait_for_indexing()` | Poll until collection status is GREEN | `--finalize` (controller) / single-machine |
| `close()` | Clean up connections | Everyone |

### Deferred indexing

The key optimization for bulk loading. Without it, every upsert triggers incremental HNSW graph updates (expensive). With deferred indexing:

1. `defer_indexing()` → Qdrant stores vectors flat, no graph construction
2. Blast data in as fast as possible (parallel upserts)
3. `enable_indexing()` → Qdrant builds HNSW in one efficient batch pass
4. `wait_for_indexing()` → block until complete

Dramatically faster for bulk loads. In the distributed path these four steps straddle the fleet: the controller does steps 1 (`--setup-only`) and 3–4 (`--finalize`), workers do step 2.

### QdrantVectorStore (`stores/qdrant.rs`)

- Uses the Rust `qdrant-client` (`Qdrant::from_url`)
- `defer_indexing()` sets `indexing_threshold=0` via `update_collection`
- `enable_indexing()` sets `indexing_threshold=20000` (Qdrant default)
- `wait_for_indexing()` polls `collection_info` every 5s until status is GREEN, logging progress
- `upsert_batch` retries with exponential backoff (up to 5 attempts) before counting the batch as errored
- Applies configured collection params (shard_number, replication_factor, HNSW, optimizers) at create time

## Runner (`runner.rs`)

`run_loader()` is the core async function. Config comes from the `loader:` block (`LoaderConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 1000 | Points per upsert call |
| `prefetch_size` | batch_size × 10 | Rows per DuckDB fetch (reduces remote I/O) |
| `concurrency` | 8 | Max parallel upsert tasks (semaphore) |
| `wps` | unbounded | Target writes/sec per worker (paced; `0`/unset = max throughput) |
| `manage_indexing` | true | If false (`--no-manage-indexing`), skip collection creation + indexing lifecycle |

Two control-plane-only helpers split the lifecycle for the distributed path: `setup_collection()` (probe dims → `ensure_collection` → `defer_indexing`) backs `--setup-only`, and `finalize()` (`enable_indexing` → `wait_for_indexing`) backs `--finalize`.

### Live metrics

Workers bump an atomic per confirmed batch; a 1 Hz sampler task emits `points_loaded` (cumulative) and `wps` (velocity) to the metrics sink, plus a final `summary`. All off the upsert hot path — see [`nova-metrics`](../../crates/nova-metrics).

### Prefetch strategy

DuckDB reads `prefetch_size` rows per fetch — one remote I/O that reads full parquet row groups. Those rows are sliced into `batch_size` upsert batches locally (no network). This minimizes S3 round trips for large datasets.

```
prefetch_size=100,000  →  DuckDB reads 100k rows (one S3 request)
batch_size=1,000       →  sliced into 100 upsert batches (local)
concurrency=8          →  8 upserts running in parallel
```

## Configuration Reference

`nova load` and `nova dist load` consume the **same** config file from `configs/loader/`. The `dispatch:` and `resources:` blocks are read by `nova dist load` only and ignored by the single-machine loader.

```yaml
vectors:                      # required, at least one entry
  dense:
    type: dense               # dense | sparse | multivector
    column: dense_embedding
    distance: cosine          # dense/multivector: cosine | dot | euclid | manhattan
    # multivector only:
    # comparator: max_sim

datasource:
  type: s3                    # s3 | local  (huggingface not yet implemented)
  # S3 options
  bucket: my-bucket
  prefix: my-dataset
  # local options
  path: /data/corpus          # dir, glob, or single .parquet
  # Common options
  id_expression: "vf_point_id(filename, file_row_number)"   # DuckDB SQL; default: "row_id"
  payload_fields:             # optional payload composition
    text: text
  file_list:                  # optional explicit file list (used by dispatch workers)
    - s3://bucket/file1.parquet

vectorstore:
  type: qdrant
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}
  collection_name: my-collection
  # params: { shard_number, replication_factor, hnsw_config, optimizers_config, ... }

loader:
  batch_size: 1000            # default
  prefetch_size: 10000        # default: batch_size * 10
  concurrency: 8              # default
  # wps: 5000                 # optional: cap writes/sec per worker

# Optional metrics sink (absent → stdout):
# metrics:
#   type: postgres
#   dsn: ${SN_METRICS_DB_URL}

# Optional, only consumed by `nova dist load`:
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
- Adding a new corpus backend (e.g. GCS)
- Adding a new vector store (e.g. Weaviate)
