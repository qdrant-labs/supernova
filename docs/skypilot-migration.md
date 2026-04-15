# SkyPilot

vectorforge uses SkyPilot for all distributed compute -- both embedding generation (GPU) and data loading (CPU). SkyPilot gives us Slurm-like job management on ephemeral cloud VMs, using spot instances that are 60-90% cheaper than on-demand.

## How it works

SkyPilot doesn't change any library code. `vectorforge/` stays the same. SkyPilot YAMLs are the orchestration layer for cloud jobs.

```
SkyPilot YAML  ->  vf CLI           ->  YAML config
(infra/compute)    (our code)           (what to embed/load)
```

## Setup

```bash
pip install skypilot[aws]
aws sso login --profile sandbox
export AWS_PROFILE=sandbox
sky check  # verifies credentials
```

**Note:** Check AWS service quotas for your sandbox account. Request vCPU limit increases if needed (EC2 -> Service Quotas -> Running On-Demand/Spot instances).

## Distributed embedding

Uses SkyPilot [pools](https://docs.skypilot.co/en/stable/examples/pools.html) -- a set of GPU workers that auto-scale and reuse setup across jobs:

```bash
# Preview the plan
vf embed-dist configs/embedder/my_dataset.yaml --dry-run

# Run (default: A10G spot, auto-scales workers)
vf embed-dist configs/embedder/my_dataset.yaml

# Custom parallelism
vf embed-dist configs/embedder/my_dataset.yaml --num-jobs 20

# Named pool (reuse across runs)
vf embed-dist configs/embedder/my_dataset.yaml --pool-name my-gpu-pool
```

Each job gets `$SKYPILOT_JOB_RANK` and `$SKYPILOT_NUM_JOBS` from SkyPilot. The `vf` CLI auto-computes offset/limit from the dataset size and processes its slice.

Override GPU resources in your config:

```yaml
resources:
  accelerators: A10G:1
  cloud: aws
  use_spot: true
```

## Distributed loading

The `vf load-dist` command shards parquet files across CPU spot instances for parallel Qdrant loading:

```bash
vf load-dist configs/dispatch/my_dataset.yaml
```

See [Data Loading](data-loading.md) for full details.

## Managed jobs (spot recovery)

`sky jobs launch` (vs plain `sky launch`) runs on a SkyPilot controller VM. Benefits:

- **Spot preemption recovery** -- if AWS reclaims the instance, SkyPilot automatically finds a new one and restarts the job
- **Laptop can disconnect** -- controller keeps running in the cloud
- **Job monitoring** -- `sky jobs queue`, `sky jobs logs`

The controller is a tiny instance (~$5/mo) that stays up. Tear it down with `sky jobs controller stop` when not in use.

## Cost estimates

### Embedding (GPU, spot instances)

| Instance | GPU | Spot $/hr | 10 jobs x 1hr |
|----------|-----|-----------|----------------|
| g5.xlarge | A10G | ~$0.38 | ~$3.80 |
| g5.2xlarge | A10G | ~$0.45 | ~$4.50 |
| g4dn.xlarge | T4 | ~$0.16 | ~$1.60 |

### Loading (CPU, spot instances)

| Instance | vCPUs | RAM | Spot $/hr | 10 jobs x 2hr |
|----------|-------|-----|-----------|----------------|
| c6i.4xlarge | 16 | 32GB | ~$0.25 | ~$5 |
| c6i.8xlarge | 32 | 64GB | ~$0.50 | ~$10 |

Loading 250M vectors sharded across 10 workers: roughly **$5-10 total**.
