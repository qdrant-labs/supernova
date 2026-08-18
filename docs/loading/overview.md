# Loading Overview

supernova's loading pipeline streams pre-embedded parquet files from S3 or HuggingFace into vector stores like Qdrant. An embedding run typically produces many parquet files (one per chunk/slice) under a shared S3 prefix -- the loader reads all of them. It uses DuckDB for efficient remote parquet reads and async concurrency for parallel upserts.

![Loading Pipeline](../fig/ingestion_pipelione.svg)

## Configuration

Loader configs live in `configs/loader/`. The same file drives a single-machine run (`nova load run`) and a distributed fleet (`nova load prepare` / `load --num-jobs N --job-rank R` / `finalize`). See the [CLI reference](../reference/cli.md#nova-load) for the lifecycle.

```yaml
vectors:
  dense:
    type: dense
    column: dense_embedding
    distance: cosine
  sparse:
    type: sparse
    column: sparse_embedding
  colbert:
    type: multivector
    column: multivector_embedding
    distance: cosine
    comparator: max_sim

datasource:
  type: s3                          # s3 or huggingface
  path: s3://my-bucket/dataset/model
  id_expression: "vf_point_id(filename, file_row_number)"   # see below
  payload_fields:                   # what ends up in the vector store payload
    text: text                      # payload key: parquet column name
    source: source

vectorstore:
  type: qdrant
  collection_name: my-collection
  url: ${QDRANT_URL}                # env var substitution with ${VAR}
  api_key: ${QDRANT_API_KEY}
  # params:                         # collection-wide HNSW/quantization/optimizers — see below

loader:
  batch_size: 1000                  # points per upsert call
  concurrency: 8                    # in-flight upsert batches
  file_look_ahead: 2                # files downloaded + read ahead of the uploader
  file_retries: 3                   # per-file download+read retries before skipping the file
  upsert_retries: 3                 # per-batch upsert retries before aborting
  # max_failed_files: 50            # abort if more than N files are skipped (default: unlimited)
```

## Running

```bash
nova load configs/loader/my_dataset.yaml
```

## Datasources

### S3

Streams parquet files via DuckDB's httpfs extension. No local download.

```yaml
datasource:
  type: s3
  path: s3://my-bucket/stanford-oval--ccnews/baai_bge_large_en_v1.5
```

Reads all parquet files matching `{path}/**/*.parquet`.

### HuggingFace

Streams directly from HuggingFace Hub via DuckDB's `hf://` protocol.

```yaml
datasource:
  type: huggingface
  repo_id: CohereLabs/wikipedia-2023-11-embed-multilingual-v3
  subdir: en
```

## Point IDs (`id_expression`)

`id_expression` is a **DuckDB SQL expression** the loader evaluates per row to produce the Qdrant point ID. The default (`row_id`) is just a bare column name and works if your parquets carry a pre-baked `row_id` column. The recommended form for supernova-produced corpora is the built-in macro:

```yaml
datasource:
  id_expression: "vf_point_id(filename, file_row_number)"
```

The macro hashes `(parquet path, physical row index)` into a deterministic UUID, so recall ground truth from the eval pipeline lines up with the loaded point IDs.

`file_row_number` is critical here: it's a DuckDB virtual column that always reflects the physical row index, regardless of parallel scan order. Do **not** use `ROW_NUMBER() OVER (PARTITION BY filename)` — that reflects DuckDB's scan ordering and produces different IDs from one run to the next under concurrency. There's a regression test for this in `tests/test_loader_id_expression.py`.

The base reader auto-enables `read_parquet(..., filename=true, file_row_number=true)` whenever your `id_expression` mentions either column, so you don't have to wire that yourself.

## Vectors

The top-level `vectors:` block declares one or more named vectors. Each key becomes the vector name in Qdrant; each entry needs `type` (`dense`, `sparse`, or `multivector`) and `column` (the parquet column).

| Type | Distance | Other |
|------|----------|-------|
| `dense` | `cosine` (default), `dot`, `euclid`, `manhattan` | -- |
| `sparse` | -- | -- |
| `multivector` | same as dense | `comparator: max_sim` (default) |

A collection with multiple named vectors lets you do hybrid retrieval (e.g. dense + sparse + late-interaction multivector).

## Collection-wide params

Everything under `vectorstore.params` is optional collection-wide config — as
opposed to the per-vector `distance`/`datatype`/`on_disk` knobs in `vectors:`
above. Qdrant's own defaults apply to anything left unset.

```yaml
vectorstore:
  type: qdrant
  collection_name: my-collection
  url: ${QDRANT_URL}
  params:
    shard_number: 6
    replication_factor: 2
    write_consistency_factor: 1
    on_disk_payload: true
    recreate: false            # drop + recreate if an existing collection's structural params conflict
    hnsw:
      m: 16
      ef_construct: 100
      full_scan_threshold: 10000
      max_indexing_threads: 4
      on_disk: true
      payload_m: 8
    quantization:
      type: scalar             # scalar (default), product, binary, turbo, none
      quantile: 0.99
      always_ram: true
    optimizers:
      default_segment_number: 4
      max_segment_size_kb: 200000   # `_kb` alias for max_segment_size (both accepted)
      memmap_threshold: 50000
      indexing_threshold: 20000
      flush_interval_sec: 5
```

- `recreate: true` drops and recreates the collection if it already exists with
  conflicting structural params (shard count, per-vector size/distance, etc.).
  It's consumed by the loader itself, not part of the request sent to Qdrant.
  Default `false`: an existing collection is left as-is (its schema isn't
  diffed against your config).
- `hnsw` / `optimizers` map straight onto Qdrant's `HnswConfigDiff` /
  `OptimizersConfigDiff` — every field is optional and independently overrides
  just that one server default.
- `quantization` picks **one** collection-wide method via `type:`:

| `type` | Extra fields | Notes |
|--------|--------------|-------|
| `scalar` (default) | `quantile`, `always_ram` | int8 scalar quantization. A bare `quantization: {}` block means this. |
| `product` | `compression` (`x4`/`x8`/`x16`/`x32`/`x64`, default `x16`), `always_ram` | Smaller index the higher the ratio, at the cost of recall. |
| `binary` | `encoding` (`one_bit` default, `two_bits`, `one_and_half_bits`), `always_ram` | Most aggressive compression; `two_bits`/`one_and_half_bits` trade some of it back for accuracy. |
| `turbo` | `bits` (`1`, `1.5`, `2`, `4`), `always_ram` | Qdrant's bit-packed quantization method. |
| `none` | -- | No quantization. A no-op at creation (same as omitting `quantization:` entirely) — see [Reindexing an existing collection](#reindexing-an-existing-collection) for what it does on `reindex`. |

## Custom sharding (Qdrant)

`vectorstore.custom_sharding` creates the collection with Qdrant's
[user-defined sharding](https://qdrant.tech/documentation/guides/distributed_deployment/#user-defined-sharding)
and routes every point to a shard key computed **per row** by a DuckDB
expression — the same expression machinery as `payload_fields`, so the
expression sees the source columns, the injected `filename`, the
`file_row_number` pseudo-column, and the registered macros.

```yaml
vectorstore:
  type: qdrant
  collection_name: my-collection
  url: ${QDRANT_URL}
  custom_sharding:
    shard_key: "org_id"                          # a plain column…
    # shard_key: "strftime(created_at, '%Y-%m')" # …or time buckets
    # shard_key: "hash(user_id) % 16"            # …or a bounded hash
    # shard_key: "file_row_number % 100"         # …or perfectly balanced slices
    shards_number: 2         # optional: physical shards created per key
    replication_factor: 2    # optional: replicas per shard, per key
    pre_create: [acme, globex]  # optional: keys to create at prepare time
```

- **Key types.** A key is a string (`keyword`) or a non-negative integer
  (`number`). Anything else — `NULL`, floats, dates, negative ints — is a hard
  read error with a cast hint (e.g. `(expr)::VARCHAR`), never a silent
  stringification: a shard key is routing.
- **Keys are created lazily.** The distinct key set is *never* computed from
  the data (no `DISTINCT` scan — `prepare` never reads the corpus). Each
  worker creates a key the first time it upserts under it; racing workers are
  fine (the loser re-checks and moves on). `pre_create` is an explicit list
  for when you know the keys up front — it just creates them during
  `prepare`/`run` instead of mid-load.
- **`shard_number` changes meaning.** With custom sharding,
  `params.shard_number` (and `custom_sharding.shards_number`, which overrides
  it per key) means shards **per shard key**. Total physical shards = keys ×
  shards_number × replication_factor — keep the per-key count small for
  high-cardinality keys.
- **The expression must be deterministic.** Qdrant does not dedupe point ids
  *across* shard keys: re-loading with an expression that maps an id to a
  different key (e.g. anything `random()`-based) leaves duplicate points in
  the collection. Make the key a pure function of the row.
- **Batching.** An upsert request carries exactly one shard key, so each
  file's points are grouped by key before batching. A high-cardinality key
  interleaved *within* files fragments batches (the loader warns when this
  bites); files partitioned or sorted by the key batch perfectly.
- **Existing collections.** If the collection already exists (`recreate:
  false`), the loader verifies it was actually created with custom sharding
  and fails fast otherwise.

Custom sharding is Qdrant-only: on other backends the `custom_sharding` key
is rejected at config parse time.

## Reindexing an existing collection

`nova load reindex <config>` patches `hnsw`/`quantization`/`optimizers` on a
collection that **already has data loaded**, without touching the data itself
— useful for comparing index or quantization variants without re-loading the
whole corpus each time. It waits for the collection to finish rebuilding
before returning (polls until Qdrant reports the collection `green` and holds
there, and fails fast if Qdrant's optimizer itself reports an error).

Two things are specific to `reindex`, as opposed to `run`/`prepare` which
create the collection:

- Structural params (`shard_number`, `replication_factor`, per-vector
  `distance`/`datatype`/`size`) aren't patchable on an existing collection and
  are ignored by `reindex` — only `hnsw`/`quantization`/`optimizers` apply.
- `quantization: { type: none }` is how you explicitly **clear** quantization
  off a collection that already has it. This is different from omitting the
  `quantization:` block entirely: omitting it leaves whatever's currently
  configured untouched, while `type: none` actively turns it off.

`nova load delete <config>` drops the collection outright (a no-op if it
doesn't exist already) — handy between `reindex` sweeps if you'd rather start
from a clean collection.

## Payload composition

`payload_fields` controls what data gets stored alongside each vector:

```yaml
payload_fields:
  text: text              # store parquet "text" column as "text" in payload
  abstract: text          # ...or rename it to "abstract"
  source: source
  url: url
```

JSON-string columns that parse to a dict are automatically unpacked into the payload.

## How it works

1. **Files are prefetched** -- up to `file_look_ahead` files are downloaded + DuckDB-read ahead while the current file's batches upload, so the store connection never stalls on S3/parse time
2. **Each file's points are sliced** into `batch_size` upsert batches
3. **Async upserts** run concurrently, controlled by a semaphore (`concurrency`); each batch is retried up to `upsert_retries` times on transient store errors
4. **Deferred indexing** -- HNSW construction is disabled during load, then built in one pass
5. **Per-file resilience** -- each file's download + read is retried (`file_retries`, default 3) with exponential backoff; a file that still fails is logged and **skipped** so one bad object can't abort the whole load. `max_failed_files` caps how many skips are tolerated before aborting.

## Indexing time in the logs

After the upload each backend logs `indexing finished: index_seconds=…` (and `reindex`
logs `reindex timing: index_seconds=…`). **These numbers are not directly comparable
across backends** — each vector store accounts for indexing time differently, so treat
`index_seconds` as a within-backend signal (e.g. comparing index/quantization variants
on the same store via `reindex`), not an apples-to-apples cross-system benchmark:

- **Qdrant / Milvus** — `index_seconds` is a distinct *post-upload* index build, timed
  directly (Qdrant defers HNSW during load then builds it in one pass; Milvus builds the
  index after inserting). Milvus additionally logs a separate `load_seconds` for pulling
  the built index into memory — a step Qdrant and Elasticsearch have no equivalent of.
- **Elasticsearch** — builds the HNSW graph *inline during ingestion*, so there is no
  separate build phase to time. `index_seconds` there is Elasticsearch's own `index_time`
  stat (a fused ingest+build figure); the parenthetical `merge-settle` is only how long
  the post-load wait for background merges took (usually ~0). The ingestion cost itself
  shows up in the loader's throughput lines (`… pts/s`).

## Tuning

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `batch_size` | 1000 | Points per upsert call. Larger = fewer HTTP calls; 1000 is good for 768-1024 dim vectors. |
| `concurrency` | 8 | In-flight upsert batches. Lower if the store times out. |
| `file_look_ahead` | 2 | Files downloaded + read ahead of the uploader. Higher = more overlap, more RAM/disk. |
| `file_retries` | 3 | Per-file download+read retries (exponential backoff) before the file is skipped. |
| `upsert_retries` | 3 | Per-batch upsert retries (exponential backoff) before the run aborts. |
| `max_failed_files` | _(none)_ | Abort once more than this many files are skipped. Unset = skip every failing file and finish. |
