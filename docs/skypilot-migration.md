# SkyPilot Migration Plan

## Why

Modal works well for burst GPU jobs (embedding generation) but is expensive for sustained CPU workloads like data loading. SkyPilot gives us Slurm-like job management on ephemeral cloud VMs, using our existing AWS sandbox account. Spot instances are 60-90% cheaper than on-demand.

## How It Works

SkyPilot doesn't change any library code. `vectorforge/loader/` stays the same. SkyPilot YAMLs replace Modal as the orchestration layer for CPU jobs.

```
SkyPilot YAML  →  vectorforge-load CLI  →  loader YAML config
(infra/compute)    (our code)               (what to load where)
```

## Setup

```bash
pip install skypilot-nightly[aws]  # or skypilot[aws] once stable
aws sso login --profile sandbox
export AWS_PROFILE=sandbox
sky check  # verifies credentials
```

**Note:** Check AWS service quotas for your sandbox account. Request vCPU limit increases if needed (EC2 → Service Quotas → Running On-Demand/Spot instances).

## Example: Single Load Job

```yaml
# sky/load-cohere-en.yaml
resources:
  cpus: 16
  memory: 64
  cloud: aws
  use_spot: true

file_mounts:
  /app:
    source: .

envs:
  QDRANT_URL: https://your-cluster.qdrant.io
  QDRANT_API_KEY: your-key
  AWS_REGION: us-east-1

setup: |
  cd /app && pip install -e .

run: |
  cd /app && vectorforge-load configs/loader/cohere200M.yaml
```

```bash
sky launch sky/load-cohere-en.yaml
```

## Parallel Loading (Sharded)

The big win: shard the dataset across N workers, each loading a slice into Qdrant in parallel. Combined with deferred indexing, Qdrant just stores vectors flat while N workers blast data in concurrently.

### Approach

1. Pre-shard: data is already sharded by language subfolder on S3 (`en/`, `de/`, `fr/`, etc.)
2. Each SkyPilot job loads one shard
3. Fan out with `--async` so all jobs run concurrently

```bash
# launch.sh — fan out one job per language
LANGS=(en de fr es it pt ja zh ru ar)

for lang in "${LANGS[@]}"; do
  sky jobs launch sky/load-shard.yaml \
    --env SHARD=$lang \
    --name "load-$lang" \
    -y --async
done
```

```yaml
# sky/load-shard.yaml
resources:
  cpus: 8
  memory: 32
  cloud: aws
  use_spot: true

file_mounts:
  /app:
    source: .

envs:
  SHARD: ""  # overridden by --env
  QDRANT_URL: ""
  QDRANT_API_KEY: ""

setup: |
  cd /app && pip install -e .

run: |
  cd /app && vectorforge-load configs/loader/cohere_shard.yaml --shard $SHARD
```

**Note:** The `--shard` flag on `vectorforge-load` doesn't exist yet. Implementation would override the `s3_prefix` to append the shard name, e.g. `cohere--wikipedia/embed-multilingual-v3/en/`. Alternatively, generate per-shard YAML configs dynamically.

### Simpler alternative (no code changes)

Generate a config per shard with a script:

```bash
for lang in en de fr es; do
  cat > /tmp/load_${lang}.yaml <<EOF
datasource:
  type: s3
  s3_bucket: qdrant---vectorforge
  s3_prefix: cohere--wikipedia/embed-multilingual-v3/${lang}
  columns:
    id: row_id
    embedding: embedding
  payload_fields:
    text: text
    url: url
    title: title

vectorstore:
  type: qdrant
  collection_name: cohere-wikipedia
  url: \${QDRANT_URL}
  api_key: \${QDRANT_API_KEY}

loader:
  batch_size: 1000
  prefetch_size: 100000
  concurrency: 8
EOF

  sky jobs launch sky/load-shard.yaml \
    --env CONFIG=/tmp/load_${lang}.yaml \
    -y --async
done
```

## Managed Jobs (Spot Recovery)

`sky jobs launch` (vs plain `sky launch`) runs on a SkyPilot controller VM. Benefits:

- **Spot preemption recovery** — if AWS reclaims the instance, SkyPilot automatically finds a new one and restarts the job
- **Laptop can disconnect** — controller keeps running in the cloud
- **Job monitoring** — `sky jobs queue`, `sky jobs logs`

The controller is a tiny instance (~$5/mo) that stays up. Tear it down with `sky jobs controller stop` when not in use.

## Cost Estimate

For loading jobs (CPU-only, spot instances):

| Instance | vCPUs | RAM | Spot $/hr | 10 jobs × 2hr |
|----------|-------|-----|-----------|----------------|
| c6i.4xlarge | 16 | 32GB | ~$0.25 | ~$5 |
| c6i.8xlarge | 32 | 64GB | ~$0.50 | ~$10 |
| m6i.4xlarge | 16 | 64GB | ~$0.30 | ~$6 |

Loading 250M vectors sharded across 10 workers: roughly **$5-10 total**.

## What Stays on Modal

- Embedding generation (`modal_batch.py`) — needs GPUs, burst workload, Modal is a good fit
- Any future GPU-dependent jobs

## What Moves to SkyPilot

- Data import (`modal_import_cohere.py` → SkyPilot job)
- Data loading (`vectorforge-load` → SkyPilot job)
- Any sustained CPU workload