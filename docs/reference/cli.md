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

## nova storm

Load-test a vector store from a single machine. Workers replicate the same load profile rather than partitioning work.

```bash
nova storm <config> [options]
```

| Option | Description |
|--------|-------------|
| `--qps N` | Override `load.qps` (target queries/sec per worker; 0 = max throughput). |

## nova storm-dist

Replicate the load test across a SkyPilot pool — every worker runs the same profile, so offered load scales with the worker count.

```bash
nova storm-dist <config> [options]
```

---

## nova experiment

Compose units (embed / load / storm) over a timeline against a single subject, for workload tests.

```bash
nova experiment <config> [options]
```

---

## SkyPilot environment

Every dispatch command (`*-dist`) calls `cli.skypilot_utils.build_env_flags()` which forwards the relevant env vars to the pool/job:

- AWS: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`, `AWS_DEFAULT_REGION`
- Per-command extras: `HF_TOKEN`, `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`

Plus SkyPilot's own `SKYPILOT_JOB_RANK` / `SKYPILOT_NUM_JOBS` which `nova embed` / `nova load` read for slicing.
