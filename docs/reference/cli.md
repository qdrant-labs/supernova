# CLI Reference

`nova` is a dispatcher: `nova <cmd> [args...]` finds `nova-<cmd>` on your `PATH`
and execs it, forwarding all arguments untouched. Exit codes, signals, and
stdio pass straight through.

```bash
nova --help        # list every nova-* tool found on PATH
nova --version     # dispatcher version
```

Each sub-tool owns its own argument parsing — `nova <cmd> --help` shows that
tool's real flags.

All tools read a YAML config and expand `${VAR}` / `${VAR:-default}` references
from the environment. All shard themselves with `--num-jobs` / `--job-rank`
(rank defaults to `$SKYPILOT_JOB_RANK`).

## nova embed

Embed a dataset into parquet (Python). Two subcommands; `run` is the default,
so the bare form routes to it and the original interface is unchanged.

### nova embed run (default)

```bash
nova embed <config> [--num-jobs N --job-rank R] [--dry-run]
nova embed run <config> ...        # explicit form, identical
```

| Flag | Meaning |
|------|---------|
| `<config>` | Path to the embedder YAML (or `NOVA_CONFIG_PATH`) |
| `--num-jobs` | Total parallel jobs; each rank embeds its `offset`/`limit` slice of the dataset |
| `--job-rank` | This job's rank (0-indexed); defaults to `$SKYPILOT_JOB_RANK` |
| `--dry-run` | Print the resolved plan (source, engine, storage, slice) and exit |

See [Embedding overview](../embedding/overview.md) for the config.

### nova embed predict

Predict throughput and cost for a config before committing GPUs — samples the
dataset's token distribution and prices each forward pass; no GPU needed.

```bash
nova embed predict <config> [--gpu h100 --num-gpus 8 ...]
```

See [Throughput Prediction](../embedding/throughput-prediction.md) for the
method and full flag reference.

## nova load

Load pre-embedded parquet into a vector store (Rust). Subcommands split the
lifecycle so a fleet can prepare once, load in parallel, and finalize once.

```bash
nova load run      <config>                          # single machine: all phases
nova load prepare  <config>                          # master: create collection, defer indexing
nova load load     <config> --num-jobs N --job-rank R  # worker: load this slice (no indexing mgmt)
nova load finalize <config>                          # master: re-enable + await indexing
nova load reindex  <config>                          # patch HNSW/quantization/optimizers on an existing collection
nova load delete   <config>                          # delete the collection if it exists
nova load inspect  <config> [--num-jobs N --job-rank R]  # dry inspection (config + file slice)
nova load catalog-build --input DIR --output catalog.parquet [--resume]  # build catalog for datasource.catalog
nova load catalog-merge --inputs c1.parquet c2.parquet --output catalog.parquet  # merge catalogs
```

- **`run`** is the single-machine shorthand for `prepare` + `load` + `finalize`.
- For a fleet: `prepare` once, then `load` on every worker with its rank, then
  `finalize` once after all workers exit. Files are partitioned by a deterministic
  stride, so workers need no coordination.
- `--num-jobs` / `--job-rank` apply to `load` and `inspect` only (the phases that
  operate on a slice).
- **`reindex`** patches `vectorstore.params.{hnsw,quantization,optimizers}` on an
  *already-existing* collection in place — no data is touched, and it doesn't
  create the collection first. Useful for comparing index/quantization variants
  against data you've already loaded once. See
  [Collection-wide params](../loading/overview.md#collection-wide-params) for
  the full knob list, including all quantization methods.
- **`delete`** drops the collection if it exists (a no-op otherwise) — handy for
  clearing out a variant between `reindex` sweeps.
- **`catalog-build`** creates a local parquet catalog (`relative_path`, optional
  `file_size`) so S3 datasource startup can skip object listing.
- **`catalog-merge`** combines catalog parquet files into one (fails on duplicate
  `relative_path` entries).

Files are partitioned by a deterministic stride, point ids are content-addressed
(`vf_point_id`), and HNSW indexing is deferred during the bulk load and built
once at `finalize`.

## nova bf

Brute-force exact k-NN ground truth (Python, GPU). Two subcommands: `compute`
scores queries against a slice of the corpus and writes a per-rank partial;
`merge` folds the partials into one final top-K. A single-GPU `compute` (no
`--num-jobs`) writes the final result directly — no `merge` needed.

```bash
nova bf compute <config> [--num-jobs N --job-rank R] [--io-workers N] [--io-thread-count N] [--max-files N]
nova bf merge   <config>
```

| Flag | Meaning |
|------|---------|
| `<config>` | Path to the brute-force YAML |
| `--num-jobs` | Total parallel jobs; each rank scans a stride slice of the corpus files |
| `--job-rank` | This job's rank (0-indexed); defaults to `$SKYPILOT_JOB_RANK` |
| `--io-workers` | Override `params.io_workers` — concurrent corpus-file reader threads |
| `--io-thread-count` | Override `params.io_thread_count` — pyarrow's IO pool (the real S3 fetch concurrency) |
| `--max-files` | Read only the first N corpus files of this slice; a benchmarking aid (output is **partial**) |

See [Brute-Force overview](../brute-force/overview.md) for the config, output schema, and tuning.

## nova storm

Load-test a vector store (Rust). Work is **replicated**, not partitioned — every
worker runs the same profile, so total offered load ≈ `num_workers × {concurrency
or rps} × batch_size`.

```bash
nova storm <config> [--json]                       # legacy shorthand for `run`
nova storm run <config> [--json]                   # explicit run form
nova storm report --inputs run1.json run2.json     # merge/compare summaries
nova storm report --inputs locust_stats.csv        # ingest Locust CSV too
```

The target backend is chosen by the config's `target.type` — `qdrant` (always
built in) or `milvus` / `elastic` (build with `--features elastic,milvus`).
`milvus`/`elastic` require `query.vector_name` (the vector field) and don't yet
support `query.filter`; `search_params` are validated per backend. See
`configs/storm/example.yaml`.

The config's `load` block picks the mode:

- **closed-loop** (default, `rps` unset) — hold `concurrency` requests in flight
  for `duration_s`; measures max throughput at that depth.
- **open-loop paced** (`rps > 0`) — launch a batch dispatch on a fixed `1/rps`
  schedule with `concurrency` as an in-flight cap; avoids coordinated omission.

`batch_size` (default `1`) is how many query vectors go in each dispatch (one
batched round-trip per dispatch — Qdrant `query_batch`, Milvus batched search,
or an Elasticsearch `_msearch`) — not a special case at `1`, just the default.
`rps` paces *dispatches*, not individual queries.

`load.operation_mix` optionally routes dispatches across `query`, `upsert`, and
`delete` using relative weights (default `query: 1`, `upsert: 0`, `delete: 0`).
When `upsert` or `delete` is enabled, set `query.source.id_column` so each
sampled row has a stable point id for mutation operations.

Prints a latency summary at the end: requests/errors (dispatch counts),
`batch_size`, `requests_per_sec` (dispatch rate) and `qps` (actual query
throughput, `= requests_per_sec × batch_size`), p50/p95/p99/max latency (per
dispatch), and recall stats (per query) if `ground_truth_column` is configured.
`--json` prints that same summary as a single JSON line instead of the table —
for a caller (e.g. `nova sweep`) that parses the result rather than scraping
formatted text. All logging goes to stderr, so stdout is exactly one line with
`--json` (safe to pipe straight into `jq` or a script).

`nova storm report` normalizes benchmarking artifacts into one contract:

- `nova storm --json` output files (`Summary` objects).
- Locust stats CSV exports (expects an `Aggregated` row).

It prints a comparison table (`source`, request/error counts, throughput, p50/p95/p99/max, recall when available) plus a rollup line. Use `--json` for machine-readable stdout and `--output-json <path>` to save the normalized report (`schema_version: 1`).

## nova web

Serve the `supernova-dashboard` SPA and expose async HTTP APIs for `nova load`,
`nova storm`, and `nova dist` orchestration.

```bash
nova web
```

`nova web` is provided by the `nova-web` Rust binary. In this repository, run it
with:

```bash
cargo run -p nova-web
```

Environment variables:

| Variable | Meaning |
|------|---------|
| `PORT` | HTTP listen port (default `8080`) |
| `DIST_DIR` | Directory of built frontend assets (default `web/supernova-dashboard/dist/supernova-dashboard/browser`) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Used by helper routes such as collection listing and random query |
| `NOVA_DIST_BIN` | Alternate executable for dist orchestration (default `nova`) |

Core endpoints:

| Endpoint | Purpose |
|------|---------|
| `GET /health` | Liveness check |
| `POST /api/v1/load/run` | Launch `nova load run` job |
| `POST /api/v1/load/prepare` | Launch `nova load prepare` job |
| `POST /api/v1/load/load` | Launch `nova load load` job |
| `POST /api/v1/load/finalize` | Launch `nova load finalize` job |
| `POST /api/v1/load/reindex` | Launch `nova load reindex` job |
| `POST /api/v1/load/delete` | Launch `nova load delete` job |
| `POST /api/v1/load/inspect` | Launch `nova load inspect` job |
| `POST /api/v1/storm/run` | Launch `nova storm` benchmark job |
| `POST /api/v1/storm/report` | Build storm report from input files |
| `POST /api/v1/dist/load` | Launch `nova dist load` orchestration job |
| `POST /api/v1/dist/storm` | Launch `nova dist storm` orchestration job |
| `GET /api/v1/jobs` | List job status/result metadata |
| `GET /api/v1/jobs/{job_id}` | Get one job status/result |
| `POST /api/v1/jobs/{job_id}/cancel` | Cancel a running job |

Request payload patterns:

- `load/*` and `storm/run`: pass either `config_path` or `config_yaml`.
- `storm/report`: use `inputs` array and optional `output_json`.
- `dist/load`: supports `resources`, `num_jobs`, `pool_name`, `dry_run`,
  `finalize`, and catalog build/staging flags.
- `dist/storm`: supports `resources`, `num_jobs`, `pool_name`, `dry_run`,
  `stage_query_source`, and `query_source_remote_dir`.

Example:

```bash
curl -X POST http://localhost:8080/api/v1/dist/load \
  -H 'content-type: application/json' \
  -d '{"config_path":"configs/loader/test.yaml","num_jobs":4,"dry_run":true}'
```

### Search-time tuning (`query.search_params`)

Optional, server-side (Qdrant `SearchParams`) — distinct from the `load` block's
client-side pacing knobs. Every field is optional; unset ones keep the
collection's own defaults.

```yaml
query:
  vector_name: dense
  top_k: 10
  source:
    uri: s3://my-bucket/queries.parquet
    column: query_embedding
    ground_truth_column: hit_ids
  search_params:
    hnsw_ef: 128          # beam width at query time; higher = more accurate, slower
    exact: false           # true = brute-force (bypasses HNSW *and* quantization)
    quantization:
      ignore: false        # true = search with full-precision vectors, skip the quantized index
      rescore: true         # re-score quantized top-k candidates against full-precision vectors
      oversampling: 2.0      # preselect oversampling × top_k candidates via the quantized index before rescoring
```

Use this to measure the recall/latency tradeoff of a quantized collection
(loaded via `nova load`'s `vectorstore.params.quantization` — see
[Collection-wide params](../loading/overview.md#collection-wide-params)) under
different query-time settings without reloading data.

### Filtering (`query.filter`)

Optional payload/metadata filter, shaped like `nova bf`'s own filter (see
[Brute-Force overview](../brute-force/overview.md)) — `must`/`should`/`must_not`
groups of `match`/`range`/`match_text` conditions — so a filter authored for a
`nova bf` ground-truth run and a `nova storm` load test read the same way.
Translating this into an actual request is backend-specific; today that's
Qdrant's own `Filter`.

A **static** filter applies uniformly to every query in the run:

```yaml
query:
  source:
    uri: s3://my-bucket/queries.parquet
    column: query_embedding
  filter:
    must:
      - field: category
        match: shoes
      - field: price
        range:
          gte: 10.0
          lt: 100.0
    should:
      - field: description
        match_text: "waterproof hiking"
```

A **per-query** filter pulls each condition's comparison value from a column in
the *queries* file instead of a literal — `match_from_query`,
`range_from_query`, `match_text_from_query` — so two different queries in the
same run can each be restricted to a different subset (their own tenant,
budget, or search phrase):

```yaml
query:
  source:
    uri: s3://my-bucket/queries.parquet
    column: query_embedding
  filter:
    must:
      - field: tenant_id
        match_from_query: tenant_column   # each query's own tenant, from this column
      - field: budget
        range_from_query:
          lt: max_budget                   # each query's own ceiling
```

Every column a `_from_query` condition names is read alongside that query's
vector when the queries file is loaded — a NULL in one of those columns is a
load-time error, not "no filter for this query": use a non-matching
placeholder value instead (the same convention `nova bf`'s own MS MARCO
configs use for unused per-query slots, e.g. `domain_slot_N` columns holding
`"zzznomatchzzz000"`).

**Qdrant caveat**: Qdrant's match condition has no float-equality variant (only
keyword, integer, bool, and `MatchAny` lists of integers or keywords) — a
`match`/`match_from_query` value that's a float, or a `MatchAny` list that mixes
types, is rejected with a clear error (at config-load time for a static filter,
at query-dispatch time for a per-query one, since the value there comes from
data). Use `range`/`range_from_query` with equal `gte`/`lte` bounds for a
numeric equality check instead.

## nova sweep

Orchestrate `nova-load` + `nova-storm` across a matrix of collection/index/search
configs (Python), producing one combined recall/latency/throughput report.
Ground truth is out of scope — point `queries:` at a parquet already produced
by `nova bf` (or equivalent); `nova sweep` never invokes `nova bf` itself.

```bash
nova sweep <config.yaml> [--skip-insert] [--cleanup] [--dry-run]
```

- `target.type` is **required** and selects the backend: `qdrant`, `milvus`,
  or `elastic` (same three as `nova-load`/`nova-storm`). Each carries its own
  `target:` fields and search-param vocabulary (`{hnsw_ef, exact,
  quantization}` / `{ef, nprobe}` / `{num_candidates}`); see the Sweep
  overview's "Target backends" section.
- The config declares three cartesian-grid axes — `data_layouts` (structural,
  forces a fresh `nova-load run`), `index_variants` (patched in place via
  `nova-load reindex`, no reload), `searches` (storm-only, no store-side
  change) — expanded and swept as **one collection per `data_layouts` entry,
  reused across every `index_variant`**, never a new collection per variant.
  The base collection name is now required explicitly in the config via
  `collection_name`; it is no longer inferred from the config filename.
- `index_variants` are walked in an order chosen to minimize rebuild cost
  (HNSW changes grouped together, quantization changes absorb the frequent
  transitions — small-scale empirical testing showed quantization changes
  reindex meaningfully faster), not declaration order. This cost sort keys off
  Qdrant's `hnsw.*` fields only; for Milvus/Elastic it is a no-op and variants
  run in declaration order (relevant for Elastic, where HNSW `m` may only
  increase across in-place reindexes).
- If a sweep's target collection already exists, the run **errors and exits
  immediately** rather than guessing what to do with it — pass `--skip-insert`
  to reuse it as-is (skips that data_layout's insert phase entirely), or set
  `target.recreate: always` in the config to force a fresh reload every time.
- `--cleanup` deletes only the collections *this run* inserted into — never
  one reused via `--skip-insert`.
- `--dry-run` prints the expanded slice/point counts and which data_layouts
  would need `--skip-insert`, without executing anything.
- Single machine, sequential — no `--num-jobs`/`--job-rank` yet (a single
  target has nothing to shard across; see the [Sweep overview](../sweep/overview.md#distribution)).

See the [Sweep overview](../sweep/overview.md) for the full config schema and
worked examples.

## Local dev: overriding a tool's location

`nova` resolves `nova-<cmd>` on `PATH`. For local iteration, build a tool and put
it ahead on `PATH` (e.g. `cargo build -p nova-load && export
PATH="$PWD/target/debug:$PATH"`), or install it into place with the matching
`make` target.
