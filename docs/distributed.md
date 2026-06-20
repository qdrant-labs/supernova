# Distributed

Every tool shards its own work from `--num-jobs` / `--job-rank` (rank defaults to
`$SKYPILOT_JOB_RANK`). A fleet is just N copies of a tool, each with a different
rank — there's no central coordinator. You can drive that yourself on any
infrastructure, or let **`nova dist`** drive [SkyPilot](https://skypilot.co) for
you.

## DIY (no orchestrator)

If your scheduler already provisions nodes and sets a rank, call the tools
directly:

```bash
nova embed configs/embedder/test.yaml --num-jobs 50 --job-rank $RANK
nova load  load configs/loader/test.yaml --num-jobs 50 --job-rank $RANK
```

Load also needs a one-time `prepare` (create collection, defer indexing) before
the fleet and a `finalize` (build the index) after — see below.

## nova dist (SkyPilot)

```bash
make dist        # installs nova-dist (controller-side only; workers don't need it)
```

`nova dist <tool>` provisions a SkyPilot pool and submits the ranked jobs. It
runs only on your controller (laptop / dispatch box); workers just run the tool
binaries.

### Resources are separate from the workload

Compute is defined by a **SkyPilot YAML you point at** with `--resources`, kept
entirely separate from the workload config. That YAML owns the three things that
vary by environment:

- `resources:` — cloud, accelerators, spot, regions, disk, image
- `setup:` — how a worker installs the tool (Rust binary or Python package)
- `envs:` — any extra environment

`nova dist` adds only the pool wrapper, the staged workload config, and the
per-rank `run:` command. Copy a template from `configs/skypilot/` and tweak it:

```yaml
# configs/skypilot/load.yaml  (you own this)
resources:
  cloud: aws
  cpus: 8+
  use_spot: true
setup: |
  source "$HOME/.cargo/env"
  cargo install --git https://github.com/qdrant-labs/supernova nova-load
```

Secrets are **not** written to worker disk: `nova dist` forwards AWS credentials,
every `${VAR}` your workload config references, and per-tool baselines (e.g.
`HF_TOKEN`) as job env vars, and the tool resolves `${VAR}` at run time.

### embed

Fan-out: each rank embeds its slice of the dataset.

```bash
nova dist embed configs/embedder/test.yaml \
    --resources configs/skypilot/embed.yaml --num-jobs 50
```

### load

Three phases, because the collection must exist before workers write and the
index must be built after:

```bash
# 1. controller: create the collection + defer indexing
# 2. fleet:      N workers, each loading its file slice
nova dist load configs/loader/test.yaml \
    --resources configs/skypilot/load.yaml --num-jobs 50

# 3. controller, after all workers finish: build the index
nova dist load configs/loader/test.yaml --finalize
```

`nova dist load` runs phase 1 locally, then fans out phase 2. Phase 3 is a
separate command you run once the workers are done (spot workers can't be
reliably awaited from the launch call). Files are partitioned by a deterministic
stride, so workers never coordinate.

### storm

Replicated, **not** partitioned — every worker runs the same profile, so total
offered load ≈ `num_jobs × (concurrency or qps)`.

```bash
nova dist storm configs/storm/test.yaml \
    --resources configs/skypilot/storm.yaml --num-jobs 10
```

## Inspect before launching

`--dry-run` generates the pool/job YAMLs under `~/.nova/runs/<ts>_<pool>/` and
prints the plan plus the equivalent `sky jobs …` CLI commands — without
launching anything. Useful to verify the resources and the per-rank command, or
to run the launch by hand with the `sky` CLI.

```bash
nova dist load configs/loader/test.yaml \
    --resources configs/skypilot/load.yaml --num-jobs 50 --dry-run
```

## Monitor & tear down

```bash
sky jobs queue                 # job status
sky jobs logs <job-id>         # a worker's logs
sky jobs pool down <pool>      # release the pool when done
```
