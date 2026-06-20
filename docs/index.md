# supernova

supernova is a toolkit for building large-scale vector search benchmarks. It does three things:

1. **Embedding generation** — take a dataset, embed it with any model (dense, sparse, or multivector), produce parquet files
2. **Vector store loading** — take pre-embedded parquet files, load them into a database
3. **Load testing** — fire query load at a vector store and measure its latency

Each is its own tool, and each shards itself for massive parallelization via `--num-jobs` / `--job-rank`. The datasets we work with are often hundreds of millions of rows and hundreds of gigabytes.

## Mental model

The pipelines are independent. You can run them separately, on different machines, at different times. The parquet files are the bridge — each embedding run produces many parquet files under a common URI prefix, which the loader then reads.

```
Source (HuggingFace) ──▶ nova embed ──▶ parquet (S3/HF/local) ──▶ nova load ──▶ Qdrant ──▶ nova storm
```

Corpora are addressed by URI. Three schemes are supported:

| Scheme | Used for |
|--------|----------|
| `s3://bucket/prefix` | S3 buckets (the typical production destination) |
| `hf://buckets/ns/name[/subdir]` | HuggingFace Storage Buckets (mutable object storage on the Hub) |
| local path | Local filesystem (handy for tests and single-machine flows) |

## One CLI, many tools

`nova` is a **git-style dispatcher**: `nova <cmd>` finds an executable named `nova-<cmd>` on your `PATH` and execs it. A command can be implemented in any language — a Rust binary (`nova-load`, `nova-storm`) or a Python console script (`nova-embed`) look identical from the outside. You install the dispatcher once and add only the sub-tools you need.

`nova --help` lists every `nova-*` tool found on your `PATH`.

| Command | Purpose | Language |
|---------|---------|----------|
| `nova embed` | Embed a dataset into parquet | Python |
| `nova load`  | Load pre-embedded parquet into a vector store | Rust |
| `nova storm` | Load-test a vector store | Rust |

Distributed runs are just N copies of a tool, each with its own `--job-rank`. Orchestration (SkyPilot, Slurm, a shell loop) is **external** — it provisions nodes and invokes `nova <tool>` on each, passing that node's rank. The tools themselves know nothing about the fleet.

## Key design principles

- **Streaming** — no pipeline loads the full dataset into memory; data is processed in chunks/batches throughout.
- **Self-sharding tools** — each tool partitions its own work from `--num-jobs` / `--job-rank` (rank defaults to `$SKYPILOT_JOB_RANK`). No central coordinator.
- **YAML-driven** — every run is defined by a YAML config; `${VAR}` / `${VAR:-default}` references are expanded from the environment.
- **Flat parquet output** — embedding output is flat columnar data (no nested JSON). Payload composition happens at load time.

See the [CLI reference](reference/cli.md) for every flag, and the [Embedding](embedding/overview.md) and [Loading](loading/overview.md) sections for each tool's config.
