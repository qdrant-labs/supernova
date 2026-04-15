# Distributed Loading

For terabyte-scale datasets, distribute loading across SkyPilot spot instances using pools.

```bash
vf load-dist configs/dispatch/cohere200M.yaml
vf load-dist configs/dispatch/cohere200M.yaml --dry-run
vf load-dist configs/dispatch/cohere200M.yaml --num-shards 20
```

## Configuration

Dispatch configs extend the standard loader config with `dispatch` and `resources` sections:

```yaml
dispatch:
  num_shards: 10                    # number of parallel workers
  run_name: cohere200M

resources:                          # SkyPilot VM spec
  cpus: 2
  memory: 8
  cloud: aws
  use_spot: true

datasource:
  type: s3
  s3_bucket: my-bucket
  s3_prefix: cohere--wikipedia/embed-multilingual-v3
  payload_fields:
    text: text
    source: source

vectorstore:
  type: qdrant
  collection_name: cohere-wikipedia
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}

loader:
  batch_size: 1000
  prefetch_size: 100000
  concurrency: 8
```

## How it works

1. **Discover** -- lists all parquet files at the S3 prefix
2. **Setup Qdrant** -- creates the collection and defers indexing
3. **Pool** -- creates a SkyPilot pool with autoscaling CPU workers
4. **Submit** -- submits N jobs to the pool. Each job discovers files and picks its shard via `$SKYPILOT_JOB_RANK`
5. **Finalize** -- after all jobs complete, enable indexing to build the HNSW graph

```bash
# After monitoring shows all jobs succeeded:
vf load-dist configs/dispatch/cohere200M.yaml --finalize
```

## Monitoring

```bash
sky jobs pool status <pool-name>
sky jobs pool logs <pool-name>
sky jobs pool ssh <pool-name> <worker-id>
sky jobs pool down <pool-name>
```

## Generated artifacts

Each run creates a directory:

```
runs/2026-04-13T14-30_cohere200M/
  pool.yaml                  # pool config (resources, setup)
  job.yaml                   # job config (run command)
  manifest.json              # file counts, shard plan
```

## Running shards manually

Each worker runs a standard `vf load` command with `--num-jobs` and `--no-manage-indexing`. You can run shards locally without SkyPilot:

```bash
# Run shard 0 of 10
vf load configs/dispatch/cohere200M.yaml --num-jobs 10 --job-rank 0 --no-manage-indexing
```

## Prerequisites

- [SkyPilot](https://skypilot.readthedocs.io/) installed and configured (`sky check aws`)
- Environment variables: `QDRANT_URL`, `QDRANT_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
