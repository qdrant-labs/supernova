# SkyPilot

supernova uses SkyPilot for all distributed compute -- both embedding generation (GPU) and data loading (CPU). SkyPilot gives us job management on ephemeral cloud VMs, using spot instances that are 60-90% cheaper than on-demand.

## Setup

```bash
pip install skypilot[aws]
aws sso login --profile sandbox
export AWS_PROFILE=sandbox
sky check
```

Check AWS service quotas for your account. Request vCPU limit increases if needed (EC2 -> Service Quotas).

## Pools

Both `nova embed-dist` and `nova load-dist` use [SkyPilot pools](https://docs.skypilot.co/en/stable/examples/pools.html). A pool is a set of workers that auto-scale and reuse setup across jobs. Workers are provisioned once (uv sync, model downloads), then jobs are submitted to the pool.

```bash
# Create a pool
sky jobs pool apply -p my-pool pool.yaml

# Submit jobs
sky jobs launch -p my-pool --num-jobs 20 job.yaml

# Monitor
sky jobs pool status my-pool
sky jobs pool status my-pool --all   # show individual workers

# Logs
sky jobs pool logs my-pool <worker-id>
sky jobs pool logs --controller my-pool

# SSH
sky jobs pool ssh my-pool <worker-id>

# Tear down
sky jobs pool down my-pool
```

## Managed jobs (spot recovery)

`sky jobs launch` runs on a SkyPilot controller VM:

- **Spot preemption recovery** -- if AWS reclaims the instance, SkyPilot finds a new one and restarts
- **Laptop can disconnect** -- controller keeps running in the cloud
- **Job monitoring** -- `sky jobs queue`, `sky jobs logs`

The controller is a tiny instance (~$5/mo). Tear it down with `sky jobs controller stop` when not in use.

## Cost estimates

### Embedding (GPU, spot)

| Instance | GPU | Spot $/hr | 10 jobs x 1hr |
|----------|-----|-----------|----------------|
| g5.xlarge | A10G | ~$0.38 | ~$3.80 |
| g4dn.xlarge | T4 | ~$0.16 | ~$1.60 |

### Loading (CPU, spot)

| Instance | vCPUs | RAM | Spot $/hr | 10 jobs x 2hr |
|----------|-------|-----|-----------|----------------|
| c6i.4xlarge | 16 | 32GB | ~$0.25 | ~$5 |
| c6i.8xlarge | 32 | 64GB | ~$0.50 | ~$10 |

Loading 250M vectors sharded across 10 workers: roughly **$5-10 total**.
