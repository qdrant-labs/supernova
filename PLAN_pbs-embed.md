# Plan: run `nova embed` on a PBS supercomputer

## TL;DR

`nova embed` is already scheduler-agnostic at its core — every worker shards its
own slice from `--num-jobs` / `--job-rank`, with no coordinator. What's missing
for a PBS Pro / OpenPBS machine (ALCF Polaris/Aurora, NERSC-style HPC) is
everything *around* that core, because today's assumptions are cloud-shaped:

1. **Rank detection** only knows `$SKYPILOT_JOB_RANK`. PBS sets
   `$PBS_ARRAY_INDEX` (job arrays) or nothing at all (multi-node jobs launched
   via `mpiexec`/`pbsdsh`, where rank comes from the MPI launcher).
2. **The only data source is HuggingFace Hub** — listing, footer reads, and file
   downloads all hit the network. HPC compute nodes typically have **no
   internet** (or only an HTTP proxy). We need a local-parquet source that reads
   from the shared filesystem (Lustre/GPFS).
3. **Models are downloaded at run time** from HF Hub — same no-internet problem.
   Solved by pre-staging the HF cache on shared FS + `HF_HUB_OFFLINE=1`, no code
   change needed, but it must be documented and the failure mode made loud.
4. **No resume**: a rank killed by walltime restarts its whole slice from row 0.
   On clouds you retry a spot worker; on PBS, walltime kills are *routine* and
   queue round-trips are expensive, so skip-what's-done matters much more.
5. **Multi-GPU nodes**: cloud workers are 1 GPU = 1 job. A PBS `select=8:ngpus=4`
   allocation hands you 4 GPUs per node in one job; someone has to spawn 4
   pinned processes per node with the right global ranks.
6. **Orchestration**: `nova dist` is SkyPilot-only. We should *not* port it —
   the right PBS artifact is a `qsub` script template plus a tiny rank shim,
   with an optional `nova dist pbs` generator later.

Recommended scope for a first working run: **items 1–3 + a `scripts/pbs/`
template** (phase 1 below). Resume (4) and per-node fan-out helper (5) are
phase 2. A `nova dist pbs` subcommand (6) is deferred until the manual flow has
been used in anger.

## 1. Current state (as read from the code)

- `nova-embed` CLI (`python/nova-embed/src/nova_embed/cli/run_embedder.py`):
  takes `--num-jobs N --job-rank R`, computes `offset = R * ceil(total/N)`,
  `limit = min(...)`, prefixes output files `rank{R:0w}_`. If `--job-rank` is
  omitted it falls back to `int($SKYPILOT_JOB_RANK or 0)` — the **only**
  scheduler it auto-detects (run_embedder.py:141-146).
- Sources registry contains exactly one source: `HuggingFaceSource`
  (`sources/huggingface.py`). It lists repo files via `HfApi`/`HfFileSystem`,
  reads parquet footers over the network for row counts, and prefetches files
  with `hf_hub_download`. Everything else in the pipeline is source-agnostic
  behind `DatasetSource`.
- Storage backends: `local` (writes to a directory — shared-FS friendly),
  `object_store` (S3/GCS/Azure via obstore), `huggingface` (upload). `local`
  already covers the HPC case; parquet outputs land as
  `rank{R}_<seq>.parquet` + a per-rank manifest.
- Device: `detect_device()` → `cuda` / `mps` / `cpu` only (backends/device.py).
  One process drives one device; there is no in-process multi-GPU fan-out
  (vLLM backend could tensor-parallel, but for embedding models data-parallel
  one-process-per-GPU is the right shape anyway).
- Orchestration: `nova dist embed` (`python/nova-dist`) is a SkyPilot
  controller — provisions a pool, submits N ranked jobs, forwards env/secrets.
  Docs (`docs/distributed.md`) explicitly bless the "DIY: your scheduler sets a
  rank" path, so PBS support is completing a documented contract, not a new
  design.
- Resume: none. `run_embedder` streams the slice start-to-end; nothing checks
  for existing output. (`content_addressed_files` gives deterministic *names*,
  but the pipeline never skips already-written work.)

## 2. What PBS changes (constraints, not features)

| Cloud / SkyPilot assumption | PBS reality |
| --- | --- |
| Orchestrator provisions nodes, sets `SKYPILOT_JOB_RANK` | `qsub` job arrays set `PBS_ARRAY_INDEX`; multi-node jobs give you `$PBS_NODEFILE` and you launch ranks yourself (`mpiexec`, `pbsdsh`) |
| Compute nodes have internet | Usually none, or proxy-only; network I/O happens on login/data-mover nodes |
| Worker installs itself at job start (`setup:` runs `pip install` from git) | No internet + no root: pre-built venv or Apptainer/Singularity image on shared FS, activated via the job script |
| 1 GPU per worker VM | 4–8 GPUs per node; allocation granularity is often whole nodes |
| Spot preemption is the failure mode; retry = new VM | Walltime expiry is the failure mode; retry = re-queue and wait, so partial-work reuse is valuable |
| Outputs go to S3 | Outputs go to scratch (purged!) or project FS; S3 sync, if needed, runs afterwards from a network-connected node |
| `/tmp` is fine | Node-local SSD is `$TMPDIR`/`/local/scratch`; `/tmp` may be tiny RAM-disk. `output_dir` default of `/tmp/nova_embed` (run_embedder.py:211) is a footgun |

Two launch shapes both need to work:

- **Shape A — job array** (`#PBS -J 0-49`): one array element = one rank = one
  GPU worker. Simplest mapping, closest to SkyPilot semantics. Downside: some
  centers disallow arrays on GPU queues or schedule whole nodes only, wasting
  3 of 4 GPUs per node.
- **Shape B — one multi-node job** (`#PBS -l select=13:ngpus=4`): a single job
  where the script launches `gpus_per_node` processes on each node.
  `global_rank = node_index * gpus_per_node + local_gpu`, with
  `CUDA_VISIBLE_DEVICES=local_gpu` pinning. This is the shape ALCF/OLCF actually
  want you to use.

## 3. Design

### 3.1 Rank detection: generalize, don't enumerate schedulers forever

Replace the single `SKYPILOT_JOB_RANK` lookup with an ordered probe, factored
into a small helper (e.g. `nova_embed/cli/rank.py`) so nova-load/nova-bf can
mirror the same order later:

1. `--job-rank` flag (always wins — unchanged)
2. `NOVA_JOB_RANK` — our own generic env var; any launcher (including shape B's
   per-process spawn) can set it, and it decouples us from scheduler zoo
3. `PBS_ARRAY_INDEX` (PBS Pro) / `PBS_ARRAYID` (Torque legacy)
4. `SLURM_ARRAY_TASK_ID`, `SLURM_PROCID` — near-free to add while we're here
5. `SKYPILOT_JOB_RANK` (existing)
6. default 0, with the existing "auto-detected rank R from X" log line naming
   which var matched (important for debugging mis-ranked fleets)

One subtlety: PBS arrays can start at any offset (`-J 5-20`), and it's easy to
accidentally submit `-J 1-50` (1-indexed) against 0-indexed ranks. The CLI
already errors when `offset >= dataset_total`? — it does not; slice math would
produce a negative limit. Add a hard error for `job_rank >= num_jobs` or
`job_rank < 0` with a message that mentions the 0-indexing convention.

### 3.2 Offline data: a `local_parquet` source

New `sources/local_parquet.py` registered as `type: local_parquet` (or
`local`), mirroring `HuggingFaceSource`'s contract:

- config: `path` (a directory or glob on shared FS), same `path_filter`
  glob/regex semantics, same `columns`/`exclude_columns` projection,
  provenance stamping (`source_file_name`, `source_row_number`).
- `_ensure_counts()` reads footers with `pyarrow.parquet` directly from the FS
  (threadpool still helps on Lustre metadata); `list_files()` returns sorted
  `(path, rows)` so `files_in_window` and the `--dry-run` partition preview
  work unchanged.
- **Sort order must be deterministic** (lexicographic on relative path) — the
  whole no-coordinator model rests on every rank computing the same file list.
  Document that the input directory must be frozen while a fleet runs.
- No prefetch/downloader machinery — files are already local; just open them.
  (Optional later: copy-to-`$TMPDIR` read-through for Lustre-unfriendly access
  patterns, but pyarrow's sequential row-group reads are fine to start.)

This is the one genuinely new component, and it's small — the HF source already
isolates all network I/O; everything downstream consumes `Record`s.

For HF *datasets* the user wants on the machine: stage once from a login node
with `hf download <dataset> --repo-type dataset` into project storage, then
point `local_parquet` at it. We should ship a tiny staging script
(`scripts/pbs/stage_hf.sh`) rather than teach `HuggingFaceSource` about offline
mode — its listing path (`HfApi.list_repo_files`) can't work without network
anyway.

### 3.3 Offline models: staging + loud failure

No code change to load models: `sentence-transformers`/vLLM/transformers all
honor `HF_HOME` + `HF_HUB_OFFLINE=1`. The work is:

- The staging script above also pre-downloads the model repo into the shared
  `HF_HOME` (`hf download <model>` on a login node).
- The PBS template exports `HF_HOME=<project>/hf_cache`, `HF_HUB_OFFLINE=1`,
  `HF_HUB_ENABLE_HF_TRANSFER=0` (hf_transfer buys nothing locally).
- Verify the failure is loud, not a 300-second connect-timeout hang per rank:
  with `HF_HUB_OFFLINE=1` and a missing cache entry, huggingface_hub raises
  immediately — good. Without the env var set, ranks hang then die N times in
  parallel — which is why the template sets it unconditionally.

### 3.4 Packaging: venv-on-shared-FS first, Apptainer second

- **Baseline**: build a uv venv on a login node (same arch as compute nodes on
  most machines) under project storage; the job script `source`s it after
  `module load` of the site CUDA/compiler stack. Zero new infrastructure.
- **If glibc/driver mismatch bites** (or the center pushes containers):
  an Apptainer image built from our existing Python deps. Defer until needed.
- Explicitly *not* doing SkyPilot-style per-job `setup:` — there is no per-job
  install step on HPC; environment is prepared once, out of band.

### 3.5 Launch templates: `scripts/pbs/`, not a new orchestrator

Ship two annotated templates users copy-and-edit (account, queue, walltime,
paths are site-specific and change rarely):

- `scripts/pbs/embed_array.pbs` — shape A. `#PBS -J 0-<N-1>`, each element runs
  `nova-embed $CFG --num-jobs $N` and rank auto-detects from
  `PBS_ARRAY_INDEX`.
- `scripts/pbs/embed_nodes.pbs` — shape B. Computes
  `num_jobs = nodes * gpus_per_node`, then per node × per GPU launches
  `CUDA_VISIBLE_DEVICES=$g NOVA_JOB_RANK=$((node_idx*gpn+g)) nova-embed $CFG
  --num-jobs $num_jobs &` via `pbsdsh`/`mpiexec -ppn 1` + a small node-local
  spawner (`scripts/pbs/node_spawn.sh`). The spawner also redirects each rank's
  stdout/stderr to `logs/rank<R>.log` — with 4 ranks per node multiplexed into
  one PBS output file, per-rank logs are non-negotiable for debugging.
- Both templates: `set -euo pipefail`, export the offline env block, write
  outputs to `$PROJECT/...` (never `/tmp`), and end with a one-line
  completion-count check (`ls out/*manifest* | wc -l` vs `num_jobs`) so a
  partially-failed fleet is visible from the `qstat` epilogue.

Why not `nova dist pbs` now: `nova dist`'s value on cloud is provisioning; on
PBS provisioning *is* `qsub`, and a generator that writes a `.pbs` file saves
~20 lines of copying. Revisit after real usage shows which knobs actually vary
per run (see §5).

### 3.6 Resume after walltime kill (phase 2, but design it now)

The failure mode: 50-rank fleet, walltime expires, 7 ranks were 80% done.
Today re-running redoes everything. Two options:

- **(a) File-level skip**: with deterministic output naming per rank, on
  startup a rank lists existing `rank{R}_*.parquet` in `output_dir`, sums rows
  from their footers, and fast-forwards its source iterator by that many rows.
  Requires: writer produces files in stream order (it does — sequential flush),
  and a way to distinguish a *complete* file from a torn one. The runner
  already stages then moves into place (runner.py:92), so a file that exists at
  the destination is complete — the atomic-rename property we need is already
  there for `local` storage.
- **(b) Manifest-based**: per-rank manifest records `rows_written`; resume
  reads it and skips. Cleaner contract, but manifests are written at the *end*
  of a run today, so a killed rank has no manifest — (a)'s footer-summing works
  on exactly the runs that need it.

Recommend (a), behind `--resume` (off by default; default behavior stays
"error if output files for this rank already exist" so accidental double
submission of the same rank is caught rather than silently appended). The
skip must count *source rows consumed*, not output rows — with a splitting
chunker one input row yields many output rows, so the manifest/footer count
alone is insufficient; simplest correct v1: `--resume` is supported only for
`passthrough` chunking (the fineweb/ms-marco workloads), erroring otherwise.

### 3.7 Device support (conditional)

If the target machine is NVIDIA (Polaris et al.), nothing to do. If it's
**Aurora (Intel PVC)**, `detect_device()` needs an `xpu` branch
(`torch.xpu.is_available()`), device pinning uses `ZE_AFFINITY_MASK` instead of
`CUDA_VISIBLE_DEVICES`, and backend support must be verified per embedder
(sentence-transformers on XPU works via IPEX; vLLM XPU is rough). This is a
real workstream on its own — gate it on the answer to open question #1 and
keep it out of phase 1.

## 4. Work items, phased

**Phase 1 — minimum viable PBS run** (small, independent PRs):

1. Rank-detection helper: `NOVA_JOB_RANK` > `PBS_ARRAY_INDEX`/`PBS_ARRAYID` >
   Slurm vars > `SKYPILOT_JOB_RANK`; validate `0 <= rank < num_jobs`; log the
   source. Tests: env-var precedence matrix.
2. `local_parquet` source: directory/glob, footer counts, deterministic order,
   provenance, `--dry-run` partition preview parity. Tests mirror the HF
   source's (fake parquet dir fixtures); this is the bulk of phase 1.
3. `scripts/pbs/{embed_array.pbs, embed_nodes.pbs, node_spawn.sh,
   stage_hf.sh}` + `docs/distributed.md` section "PBS / HPC" documenting the
   offline env block, staging flow, and both shapes.

**Phase 2 — quality of life once runs are real:**

4. `--resume` file-level fast-forward (passthrough chunking only, §3.6).
5. Completion tooling: a `nova-embed --check out_dir --num-jobs N` (or a
   30-line script) that reports which ranks have manifests / row counts, so
   "which ranks do I resubmit" is one command. Pairs with resume.
6. Sane default guard: refuse to run with `output_dir` under `/tmp` when
   `PBS_JOBID` is set (or just always warn) — scratch-purge and RAM-disk
   footguns.

**Phase 3 — only if manual flow proves annoying:**

7. `nova dist pbs embed <cfg> --num-jobs N [--shape array|nodes]` — renders the
   template with the config staged, prints the `qsub` line (mirroring
   `--dry-run`'s spirit: generate + show, let the user launch).
8. Apptainer image recipe, XPU support (if Aurora), S3 sync-back helper job on
   a data-mover queue.

## 5. Open questions (answers change scope)

1. **Which machine?** NVIDIA (Polaris/Perlmutter-class) → phase 1 as written.
   Aurora/Intel → add the §3.7 XPU workstream before anything else, and vLLM
   configs are likely off the table.
2. **Job arrays allowed on the GPU queue, and at what node granularity?**
   Decides whether shape A is usable or we go straight to shape B.
3. **Where must outputs ultimately live?** If S3/Qdrant, we need the sync-back
   step (data-mover node cron or follow-up job); if the analysis happens
   on-machine, `local` storage is the end state.
4. **Dataset scale vs. shared-FS quota** — fineweb-scale inputs plus outputs
   may not fit project quota; may force streaming-from-S3 (compute nodes with
   proxy?) or per-chunk stage/embed/delete choreography. Worth knowing before
   committing to "stage everything" in §3.2.
