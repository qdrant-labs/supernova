# CLI Reference

All commands are accessed through the `vf` entrypoint.

## vf embed

Run an embedding pipeline locally.

```bash
vf embed <config> [options]
```

| Option | Description |
|--------|-------------|
| `--offset N` | Skip N rows (for manual slicing) |
| `--limit N` | Process at most N rows |
| `--num-jobs N` | Total parallel jobs (auto-computes offset/limit from dataset size) |
| `--job-rank N` | This job's rank (reads `$SKYPILOT_JOB_RANK` if omitted) |

## vf embed-dist

Distribute embedding across a SkyPilot GPU pool.

```bash
vf embed-dist <config> [options]
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Generate configs and print plan, don't launch |
| `--num-jobs N` | Number of parallel jobs (default: auto from dataset size) |
| `--chunk-size N` | Rows per job (used to auto-compute num-jobs) |
| `--pool-name NAME` | SkyPilot pool name (default: auto-generated) |
| `--max-workers N` | Max pool workers for autoscaling (default: num-jobs) |

## vf load

Load pre-embedded data into a vector store.

```bash
vf load <config> [options]
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Parse config and print info without loading |
| `--no-manage-indexing` | Skip collection creation and indexing lifecycle |
| `--num-jobs N` | Total parallel jobs (auto-shards files by rank) |
| `--job-rank N` | This job's rank (reads `$SKYPILOT_JOB_RANK` if omitted) |

## vf load-dist

Distribute loading across a SkyPilot CPU pool.

```bash
vf load-dist <config> [options]
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Generate configs and print plan, don't launch |
| `--num-shards N` | Override number of shards |
| `--pool-name NAME` | SkyPilot pool name (default: auto-generated) |
| `--finalize` | Enable Qdrant indexing (run after all jobs complete) |
