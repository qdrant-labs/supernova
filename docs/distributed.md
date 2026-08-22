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

### Resources (separate from the workload)

Compute — cloud, accelerators, spot, and how a worker installs the tool — is
defined separately from the workload config. You don't *need* to define
anything: `nova dist` ships built-in defaults so a first run works out of the
box. It resolves the spec in three tiers, highest priority first:

1. **`--resources <sky.yaml>`** — an explicit override for this run
2. **`~/.nova/skypilot/<tool>.yaml`** — your standing default (override the dir
   with `$NOVA_SKYPILOT_DIR`)
3. **built-in defaults** — sensible resources + a `setup:` that installs the tool
   from git

An override is **merged by top-level key**, not wholesale: a file with only
`setup:` keeps the default `resources:` (handy for a dev build), and a file with
only `resources:` keeps the default `setup:`. The three mergeable keys:

- `resources:` — cloud, accelerators, spot, regions, disk, image
- `setup:` — how a worker installs the tool (Rust binary or Python package)
- `envs:` — extra environment

So the minimal run needs no resources file at all:

```bash
nova dist load configs/loader/test.yaml --num-jobs 50
```

To override — e.g. test a working-tree build on the fleet — drop a partial file in
`~/.nova/skypilot/load.yaml`:

```yaml
# only overrides setup; default resources are kept
setup: |
  source "$HOME/.cargo/env"
  cargo install --git https://github.com/you/supernova@my-branch nova-load
```

Or pass a full `--resources my.yaml` for a one-off. Copy-and-tweak templates live
in `configs/skypilot/`. `nova dist` prints which source it resolved
(`resources: built-in defaults` / a path).

Secrets are **not** written to worker disk: `nova dist` forwards AWS credentials,
every `${VAR}` your workload config references, and per-tool baselines (e.g.
`HF_TOKEN`) as job env vars, and the tool resolves `${VAR}` at run time.

### embed

Fan-out: each rank embeds its slice of the dataset.

```bash
nova dist embed configs/embedder/test.yaml --num-jobs 50
# (add --resources my.yaml to override the default GPU resources/setup)
```

### load

Three phases, because the collection must exist before workers write and the
index must be built after:

```bash
# 1. controller: create the collection + defer indexing
# 2. fleet:      N workers, each loading its file slice
nova dist load configs/loader/test.yaml --num-jobs 50

# 3. controller, after all workers finish: build the index
nova dist load configs/loader/test.yaml --finalize

# fleet died partway? relaunch it with --continue — every worker probes the
# store for its slice's progress and resumes where it stopped (finished ranks
# redo only their final file). Same corpus + --num-jobs required.
nova dist load configs/loader/test.yaml --num-jobs 50 --continue
```

`nova dist load` runs phase 1 locally, then fans out phase 2. Phase 3 is a
separate command you run once the workers are done (spot workers can't be
reliably awaited from the launch call). Files are partitioned by a deterministic
stride, so workers never coordinate.

### bf

Two phases, like `load` — fan out the GPU `compute` workers (each scans a corpus
stride slice), then `merge` the per-rank partials on the controller:

```bash
# 1. fleet: N GPU workers, each scoring its corpus slice
nova dist bf compute configs/brute_force/test.yaml --num-jobs 8
#    (add --resources my.yaml to override the default GPU resources/setup)

# 2. controller, after all workers finish: combine the partials
nova dist bf merge configs/brute_force/test.yaml
```

The default resources are a single-GPU box; pick the size for vCPUs (decode is
CPU-bound) rather than the GPU. See the [Brute-Force overview](brute-force/overview.md)
for config and tuning.

### storm

Replicated, **not** partitioned — every worker runs the same profile, so total
offered load ≈ `num_jobs × (concurrency or rps) × batch_size`.

```bash
nova dist storm configs/storm/test.yaml --num-jobs 10
```

## Inspect before launching

`--dry-run` generates the pool/job YAMLs under `~/.nova/runs/<ts>_<pool>/` and
prints the plan plus the equivalent `sky jobs …` CLI commands — without
launching anything. Useful to verify the resources and the per-rank command, or
to run the launch by hand with the `sky` CLI.

```bash
nova dist load configs/loader/test.yaml \
    --num-jobs 50 --dry-run
```

## Monitor & tear down

```bash
sky jobs queue                 # job status
sky jobs logs <job-id>         # a worker's logs
sky jobs pool down <pool>      # release the pool when done
```
