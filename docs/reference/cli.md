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
nova load inspect  <config> [--num-jobs N --job-rank R]  # dry inspection (config + file slice)
```

- **`run`** is the single-machine shorthand for `prepare` + `load` + `finalize`.
- For a fleet: `prepare` once, then `load` on every worker with its rank, then
  `finalize` once after all workers exit. Files are partitioned by a deterministic
  stride, so workers need no coordination.
- `--num-jobs` / `--job-rank` apply to `load` and `inspect` only (the phases that
  operate on a slice).

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
nova storm <config>
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

## Local dev: overriding a tool's location

`nova` resolves `nova-<cmd>` on `PATH`. For local iteration, build a tool and put
it ahead on `PATH` (e.g. `cargo build -p nova-load && export
PATH="$PWD/target/debug:$PATH"`), or install it into place with the matching
`make` target.
