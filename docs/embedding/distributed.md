# Distributed Embedding

SkyPilot pools create a set of GPU workers and distribute embedding jobs across them. Workers are reused -- setup (uv sync, model download) happens once, not per-slice.

```bash
# Preview the plan
vf embed-dist configs/embedder/arxiv_papers.yaml --dry-run

# Run with default resources (A10G spot)
vf embed-dist configs/embedder/arxiv_papers.yaml

# Custom number of jobs
vf embed-dist configs/embedder/arxiv_papers.yaml --num-jobs 20

# Custom chunk size (smaller = more parallelism)
vf embed-dist configs/embedder/arxiv_papers.yaml --chunk-size 50000

# On-demand instead of spot (larger AWS quota, no preemption)
vf embed-dist configs/embedder/arxiv_papers.yaml --on-demand

# Burst: provision all workers in parallel instead of SkyPilot's slow one-at-a-time ramp
vf embed-dist configs/embedder/arxiv_papers.yaml --burst

# Named pool (for reuse across runs)
vf embed-dist configs/embedder/arxiv_papers.yaml --pool-name my-gpu-pool
```

For datasets too big to run in one shot (≥100M rows), use [incremental / windowed runs](../reference/config.md#incremental--windowed-runs) to split the embedding across separately-invoked increments.

## How it works

1. **Plan** (runs locally): reads config, queries the source for dataset size
2. **Pool**: creates a SkyPilot pool with autoscaling GPU workers (`min_workers: 0`, `max_workers: N`)
3. **Submit**: submits N jobs to the pool via `sky jobs launch --num-jobs N`
4. **Each job**: SkyPilot sets `$SKYPILOT_JOB_RANK` and `$SKYPILOT_NUM_JOBS`. The `vf embed` CLI uses these to compute offset/limit and process its slice. If the YAML sets `source.offset` / `source.limit`, the N ranks divide *that window* rather than the full dataset — this is how incremental runs work
5. **Autoscale**: workers scale up to handle the queue, scale back to zero when done. With `--burst` all `max_workers` are provisioned at startup so you skip the autoscaler's slow ramp

## Custom resources

Add a `resources` section to your config to override SkyPilot VM specs:

```yaml
resources:
  accelerators: A10G:1
  cloud: aws
  use_spot: true
```

Default: A10G GPU on AWS spot instances.

## Manual pool usage

You can also manage pools directly:

```bash
# Create a pool
sky jobs pool apply -p my-pool runs/<run-dir>/pool.yaml

# Submit jobs
sky jobs launch -p my-pool --num-jobs 20 runs/<run-dir>/job.yaml

# Monitor
sky jobs pool status my-pool

# View logs
sky jobs pool logs my-pool

# SSH into a worker
sky jobs pool ssh my-pool <worker-id>

# Tear down
sky jobs pool down my-pool
```

## Generated artifacts

Each run creates a directory:

```
runs/2026-04-13T14-30_vf-embed-en/
  pool.yaml                  # pool config (resources, setup)
  job.yaml                   # job config (run command)
  manifest.json              # plan metadata
```
