# CLI Reference

Every subcommand is dispatched through the `nova` entrypoint, a click group defined in `cli/cli.py`. `nova --help` lists every command; `nova <command> --help` prints flags for one. Subcommand modules are imported lazily so `nova --help` returns in tens of milliseconds even though heavy ML libraries are involved.

Corpora and destinations are addressed by URI. The schemes supported today:

- `s3://bucket/prefix`
- `hf://buckets/namespace/name[/subdir]` — HuggingFace Storage Buckets (write + read)
- `hf://datasets/namespace/name[/subdir]` — read-only, for legacy corpora already in dataset repos (the loader's DuckDB httpfs extension only supports this form)
- `file:///abs/path`

---

## nova embed

Run the embedding pipeline locally.

```bash
nova embed <config> [options]
```

| Option | Description |
|--------|-------------|
| `--num-jobs N` | Total parallel jobs (auto-computes per-rank slice from dataset size). |
| `--job-rank N` | This job's rank (reads `$SKYPILOT_JOB_RANK` if omitted). |

The config path can also be supplied via `NOVA_CONFIG_PATH`.

## nova embed-dist

Distribute embedding across a SkyPilot GPU pool.

```bash
nova embed-dist <config> [options]
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Generate configs and print the plan, don't launch. |
| `--num-jobs N` | Number of parallel jobs (default: auto from dataset size / `chunk_size`). |
| `--chunk-size N` | Rows per job (used to auto-compute `--num-jobs`). |
| `--pool-name NAME` | SkyPilot pool name (default: auto-generated). |
| `--max-workers N` | Max pool workers for autoscaling (default: `--num-jobs`). |
| `--on-demand` | Use on-demand instead of spot — separate AWS quota, no preemption. |
| `--ramp` | Opt into SkyPilot's gradual autoscaler (`min_workers=0`). Default is burst (`min_workers=max_workers`) since EC2 provisioning is slow and the autoscaler ramps ~1 replica per 3 minutes. |

## nova partition

Run the embed pipeline with the **no-op embedder**: same I/O and sharding as `nova embed` but no GPU forward pass. Output parquets carry every column except the float vectors, so you can validate clean partitioning (`scripts/verify_no_duplicates.py`) before committing GPU time.

```bash
nova partition <config> [options]
```

| Option | Description |
|--------|-------------|
| `--num-jobs N` | Total parallel jobs (auto-computes per-rank slice from dataset size). |
| `--job-rank N` | This job's rank (defaults to `$SKYPILOT_JOB_RANK`). |
| `--list-files` | Dry-run: list matched parquet files + per-rank plan and exit. Meaningful for `source.type=huggingface` or its alias `huggingface_parquet`. |

## nova partition-dist

Distribute `nova partition` across a SkyPilot CPU pool. Same flag shape as `nova embed-dist` but cheaper resources.

```bash
nova partition-dist <config> [options]
```

Flags: `--dry-run`, `--num-jobs N`, `--chunk-size N`, `--pool-name NAME`, `--max-workers N`, `--on-demand`, `--ramp` — same semantics as `nova embed-dist`.

---

## nova load

Load pre-embedded data into a vector store.

```bash
nova load <config> [options]
```

| Option | Description |
|--------|-------------|
| `--dry-run`, `-d` | Parse config and print info without loading. |
| `--no-manage-indexing` | Skip collection creation and the indexing lifecycle (used by distributed workers; the dispatcher handles those phases). |
| `--num-jobs N` | Total parallel jobs (auto-shards files by rank). |
| `--job-rank N` | This job's rank (reads `$SKYPILOT_JOB_RANK` if omitted). |

The config path can also be supplied via `LOADER_CONFIG_PATH`. The config must include a top-level `vectors:` block.

## nova load-dist

Distribute loading across a SkyPilot CPU pool. Reads the same loader config and additionally consumes the `dispatch:` and `resources:` blocks the single-machine loader ignores.

```bash
nova load-dist <config> [options]
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Generate configs and print the plan, don't launch. |
| `--num-shards N` | Override `dispatch.num_shards`. |
| `--pool-name NAME` | SkyPilot pool name (default: auto-generated). |
| `--on-demand` | Use on-demand instead of spot. |
| `--ramp` | Gradual autoscaling instead of burst (see `nova embed-dist`). |
| `--finalize` | Skip dispatch — only enable Qdrant indexing and wait for HNSW build. Run this once all worker jobs have completed. |

---

## nova generate-queries

Sample N rows from an embedded corpus as eval query vectors. Default mode launches an in-region EC2 instance (S3 only); pass `--local` to run in-process. Output lands at `{corpus}/eval/queries_<N>.parquet` with `__source_file__` and `__source_row__` provenance columns.

```bash
nova generate-queries <corpus_uri> [options]
```

| Option | Description |
|--------|-------------|
| `-n`, `--num-queries N` | How many queries to sample (default 1000). |
| `--seed N` | RNG seed (default 42). |
| `--columns COL` | Columns to fetch (default: all). Repeat for each: `--columns dense_embedding --columns sparse_embedding`. |
| `--output FILE` | Output filename (default `queries_<n>.parquet`). |
| `--local` | Run the full pipeline in-process instead of launching EC2. |
| `--prefetch` | Download each parquet fully before reading (better for large row groups than range requests). |
| `--instance-type TYPE` | EC2 instance type (default `r5n.2xlarge` — 25 Gbps in-region). |
| `--on-demand`, `--dry-run` | As above. |

EC2 launch only works for `s3://` corpora. For `hf://` and `file://`, use `--local`.

## nova brute-force

Exhaustive nearest-neighbour search for recall evaluation. Requires `torch` (`uv sync --extra eval`).

```bash
nova brute-force <corpus_uri> [options]
```

| Option | Description |
|--------|-------------|
| `--queries FILE` | Queries parquet within `{corpus}/eval/` (default `queries_1000.parquet`). |
| `-k N` | Neighbours per query (default 1000). |
| `--metric METRIC` | `cosine` (default), `dot`, or `euclidean`. |
| `--dense-column NAME` | Embedding column to compare on (default `dense_embedding`). |
| `--output FILE` | Output filename. Default `brute_force_<queries_stem>_k<K>.parquet`. |
| `--local` | Run in-process instead of launching EC2. |
| `--num-jobs N`, `--job-rank N` | Distributed mode: each rank takes a slice of corpus files and writes a partial result. |
| `--instance-type TYPE` | EC2 instance type (default `g4dn.2xlarge` — 1× T4 GPU). |
| `--on-demand`, `--dry-run` | As above. |

EC2 launch only works for `s3://` corpora. Hits land at `{corpus}/eval/brute_force_<stem>_k<K>.parquet`. In distributed mode each worker writes `{corpus}/eval/_bf_partial_<stem>_k<K>/rankNNN.parquet`; merge with `nova brute-force-merge`.

## nova brute-force-dist

Distribute brute-force across a SkyPilot GPU pool. Each worker prefetches its assigned files to local NVMe, runs GPU similarity search, and saves a partial top-K result.

```bash
nova brute-force-dist <s3_corpus_uri> [options]
```

| Option | Description |
|--------|-------------|
| `--queries`, `-k`, `--metric`, `--dense-column` | Same as `nova brute-force`. |
| `--num-jobs N` | Number of GPU workers (default 50). |
| `--output FILE` | Final merged output filename. |
| `--instance-type TYPE` | Per-worker EC2 instance (default `g4dn.2xlarge`). |
| `--pool-name NAME`, `--on-demand`, `--dry-run` | As above. |

Today this command provisions AWS GPU instances, so it only accepts S3 corpora. For HF / local, run `nova brute-force --local`.

## nova brute-force-merge

Merge partial brute-force results from a distributed run into a single top-K parquet.

```bash
nova brute-force-merge <corpus_uri> [options]
```

| Option | Description |
|--------|-------------|
| `--queries FILE`, `-k N`, `--output FILE` | Same as `nova brute-force`. |

---

## nova analysis

Analyze a completed embedding run: schema, row count, per-rank throughput, wall clock, cost estimate.

```bash
nova analysis <config>           # derives destination from storage section
nova analysis --path s3://...    # ad-hoc, no config needed
```

| Option | Description |
|--------|-------------|
| `--path URL` | Override: direct `s3://bucket/prefix` or local dir to analyze. |
| `--cost-per-hour USD` | Per-worker hourly rate (default 0.38 = g5.xlarge A10G spot). Use ~1.01 for g5.xlarge on-demand. |
| `--check-duplicates` | Run a `source_row_id` uniqueness + coverage check across all parquets. |

## nova throughput-predict

Predict embedding throughput and cost from an embedder YAML — without running anything. Pulls dataset, model, and rendering details from the config; only GPU + simulation knobs come from the CLI.

```bash
nova throughput-predict <config> [options]
```

| Option | Description |
|--------|-------------|
| `--gpu KEY` | GPU key (default `a10g`). See `supernova/throughput.py:GPU_TABLE` for the catalogue. |
| `--gpu-scale F` | Multiplier on effective TFLOPS (e.g. `0.85` for thermal headroom). |
| `--rate USD` | $/hr override. |
| `--num-gpus N` | Parallel GPUs for wall-clock estimate. |
| `--overhead F` | Cost overhead multiplier (default 1.2). |
| `--cutoff N` | Override `dense_embedder.max_tokens`. Repeat to sweep. |
| `--model NAME`, `--hf-config NAME`, `--split NAME`, `--column NAME`, `--template STR` | Override the corresponding YAML field. |
| `--total-rows N`, `--params N` | Override row count or parameter count. |
| `--sample N` | Rows to tokenize for the empirical length distribution (default 100k). |
| `--batch-size N`, `--num-batches N` | Padding-simulation knobs. |
| `--output FILE` | Write JSON results + plot. |
| `-v`, `--verbose` | Print debug logs. |

---

## SkyPilot environment

Every dispatch command (`*-dist`, `nova brute-force`, `nova generate-queries`) calls `cli.skypilot_utils.build_env_flags()` which forwards the relevant env vars to the pool/job:

- AWS: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`, `AWS_DEFAULT_REGION`
- Per-command extras: `HF_TOKEN`, `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`

Plus SkyPilot's own `SKYPILOT_JOB_RANK` / `SKYPILOT_NUM_JOBS` which `nova embed` / `nova load` / `nova brute-force` read for slicing.
