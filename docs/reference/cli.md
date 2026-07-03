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

Embed a dataset into parquet (Python).

```bash
nova embed <config> [--num-jobs N --job-rank R] [--dry-run]
```

| Flag | Meaning |
|------|---------|
| `<config>` | Path to the embedder YAML (or `NOVA_CONFIG_PATH`) |
| `--num-jobs` | Total parallel jobs; each rank embeds its `offset`/`limit` slice of the dataset |
| `--job-rank` | This job's rank (0-indexed); defaults to `$SKYPILOT_JOB_RANK` |
| `--dry-run` | Print the resolved plan (source, engine, storage, slice) and exit |

See [Embedding overview](../embedding/overview.md) for the config.

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
nova storm <config> [--json]
```

The config's `load` block picks the mode:

- **closed-loop** (default, `rps` unset) — hold `concurrency` requests in flight
  for `duration_s`; measures max throughput at that depth.
- **open-loop paced** (`rps > 0`) — launch a batch dispatch on a fixed `1/rps`
  schedule with `concurrency` as an in-flight cap; avoids coordinated omission.

`batch_size` (default `1`) is how many query vectors go in each dispatch (one
Qdrant `query_batch` RPC per dispatch) — not a special case at `1`, just the
default. `rps` paces *dispatches*, not individual queries.

Prints a latency summary at the end: requests/errors (dispatch counts),
`batch_size`, `requests_per_sec` (dispatch rate) and `qps` (actual query
throughput, `= requests_per_sec × batch_size`), p50/p95/p99/max latency (per
dispatch), and recall stats (per query) if `ground_truth_column` is configured.
`--json` prints that same summary as a single JSON line instead of the table —
for a caller (e.g. a future `nova sweep`) that parses the result rather than
scraping formatted text. All logging goes to stderr, so stdout is exactly one
line with `--json` (safe to pipe straight into `jq` or a script).

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

## nova sweep

Orchestrate `nova-load` + `nova-storm` across a matrix of collection/index/search
configs (Python), producing one combined recall/latency/throughput report.
Ground truth is out of scope — point `queries:` at a parquet already produced
by `nova bf` (or equivalent); `nova sweep` never invokes `nova bf` itself.

```bash
nova sweep <config.yaml> [--skip-insert] [--cleanup] [--dry-run]
```

- The config declares three cartesian-grid axes — `data_layouts` (structural,
  forces a fresh `nova-load run`), `index_variants` (patched in place via
  `nova-load reindex`, no reload), `searches` (storm-only, no Qdrant-side
  change) — expanded and swept as **one collection per `data_layouts` entry,
  reused across every `index_variant`**, never a new collection per variant.
- `index_variants` are walked in an order chosen to minimize rebuild cost
  (HNSW changes grouped together, quantization changes absorb the frequent
  transitions — small-scale empirical testing showed quantization changes
  reindex meaningfully faster), not declaration order.
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
