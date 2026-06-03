# Distributed Loading

For terabyte-scale datasets, distribute loading across SkyPilot spot instances using pools.

```bash
nova load-dist configs/loader/ccnews_bge_large.yaml
nova load-dist configs/loader/ccnews_bge_large.yaml --dry-run
nova load-dist configs/loader/ccnews_bge_large.yaml --num-shards 20
```

## Configuration

Distributed runs use the same loader config as `nova load`, plus optional `dispatch` and `resources` blocks that the single-machine loader ignores:

```yaml
dispatch:
  num_shards: 10                    # number of parallel workers
  run_name: ccnews-bge-large

resources:                          # SkyPilot VM spec
  cpus: 2
  memory: 8
  cloud: aws
  use_spot: true

vectors:
  dense:
    type: dense
    column: dense_embedding
    distance: cosine

datasource:
  type: s3
  bucket: my-bucket
  prefix: stanford-oval--ccnews/baai_bge_large_en_v1.5
  payload_fields:
    text: text
    title: title
    url: requested_url

vectorstore:
  type: qdrant
  collection_name: ccnews-bge-large
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
nova load-dist configs/loader/ccnews_bge_large.yaml --finalize
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
~/.nova/runs/2026-04-13T14-30_ccnews-bge-large/
  pool.yaml                  # pool config (resources, setup)
  job.yaml                   # job config (run command)
  manifest.json              # file counts, shard plan
```

## Running shards manually

Each worker runs a standard `nova load` command with `--num-jobs` and `--no-manage-indexing`. You can run shards locally without SkyPilot:

```bash
# Run shard 0 of 10
nova load configs/loader/ccnews_bge_large.yaml --num-jobs 10 --job-rank 0 --no-manage-indexing
```

## Prerequisites

- [SkyPilot](https://skypilot.readthedocs.io/) installed and configured (`sky check aws`)
- Environment variables: `QDRANT_URL`, `QDRANT_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
