# Development

How supernova is laid out, and how to test un-released changes on a real fleet.

## Package structure

supernova is polyglot: a pip-installable Python package exposing the `nova` CLI
(`nova = supernova.cli.cli:main`), plus a Rust workspace for the performance-critical
tools. `nova` is a thin dispatcher — it execs the Rust binaries (`storm`, `load`) and the
Python `embed` console script as separate processes, so the `nova` process itself imports
nothing heavy.

```text
supernova/                        # Python: the `nova` CLI, embed pipeline, orchestration
├── cli/                          # the `nova` CLI
│   ├── cli.py                    # entry point (LazyGroup): dispatch to binaries / Python
│   ├── run_embedder.py           # `nova embed` (Python console script `nova-embed`)
│   ├── run_experiment.py         # `nova experiment` (in-process click command)
│   ├── run_<verb>_distributed.py # `nova dist <verb>` SkyPilot dispatcher (embed/load/storm)
│   ├── config_resolve.py         # ${VAR} resolution for controller-side config reads
│   └── skypilot_utils.py         # shared dispatch helpers (worker bootstrap, pools,
│                                 #   make_run_dir, nova_home, config_mount, ...)
├── sources/        # dataset sources to embed (HuggingFace)        [ABC: DatasetSource]
├── chunkers/       # model-agnostic text splitting (issue #12)     [ABC: Chunker]
├── embedders/      # dense / sparse / multivector embedders + the [ABCs per family]
│                   #   streaming embed pipeline (buffer/runner/worker)
├── storage/        # embedding output sinks: s3 / local / hf       [ABC: StorageBackend]
├── metrics/        # swappable metrics sink (stdout / postgres)    [ABC: MetricsBackend]
├── experiment/     # compose units over a timeline (workload tests)
├── destinations.py # s3:// and hf:// URI helpers
└── models.py, utils.py

crates/                           # Rust: the performance-critical tools
├── nova-load/      # `nova load`  — load pre-embedded parquet into a vector store
├── nova-storm/     # `nova storm` — load-test a vector store
└── nova-metrics/   # shared metrics client; mirrors supernova/metrics over one schema
```

### The three-layer pattern

Every workload follows the same shape — copy it when adding a new one:

1. **Core library** — vendor-agnostic ABCs you implement per backend (a source, an
   embedder, a store). Pure, importable, no cloud.
2. **Local CLI verb** (`nova embed`, `nova load`, `nova storm`) — one worker, one process.
   `nova` execs the tool (the Rust binary for load/storm, the `nova-embed` console script
   for embed); the tool owns its arg parsing, config, and metrics. It shards itself when
   given `--num-jobs` + `--job-rank` (rank defaults to `$SKYPILOT_JOB_RANK`).
3. **`dist` wrapper** (`nova dist embed`) — provisions a SkyPilot pool and submits N copies
   of the *same* local verb, one per shard (embed/load are partitioned; storm is replicated).

Workers do **not** receive your code by file-sync. The dispatcher bootstraps each worker to
install a *pinned, named* artifact matching the controller's version, so controller and
workers always run identical code: the Python `embed` worker `pip install`s
`supernova[<extra>]==<version>` from PyPI; the Rust `storm`/`load` worker `cargo install`s
the crate at tag `v<version>` from GitHub. Python extras (`embed`, `dist`, `pg`) live in
`pyproject.toml`; the Rust crates are in the workspace `Cargo.toml`.

### Local state

Run metadata — generated pool/job YAMLs, manifests, the staged config — lives under
`~/.nova/runs/` (override with `$NOVA_HOME`). It's intentionally outside any repo so an
installed `nova` writes to a stable location no matter where it's invoked.

## Dev mode: testing un-released changes on a fleet

Workers install the *published* version by default, so local edits never reach them.
To run your working changes on a fleet you override where each worker gets the tool —
and because the tools are polyglot, the knob depends on the language:
`NOVA_WORKER_INSTALL_SPEC` for the Python `embed` worker, `NOVA_RUST_INSTALL_SPEC` for the
Rust `storm` / `load` workers. For a single-machine run, `NOVA_<TOOL>_BIN` points straight
at a built binary (see [Local binary override](#local-binary-override-nova_tool_bin)).

**Always iterate locally first** (no cloud, instant):

```bash
nova embed configs/embedder/test.yaml      # Python
nova storm configs/storm/test.yaml         # Rust — or NOVA_STORM_BIN, see below
```

### Python tools (`embed`)

Override the worker install source with the `NOVA_WORKER_INSTALL_SPEC` env var — a PEP 508
spec in which `{extra}` is substituted per command.

**Distributed, off a pushed git commit:**

```bash
git checkout -b my-feature
git commit -am "wip" && git push          # workers clone the SHA from GitHub
export NOVA_WORKER_INSTALL_SPEC='supernova[{extra}] @ git+https://github.com/qdrant-labs/supernova@'$(git rev-parse HEAD)
nova embed-dist configs/embedder/test.yaml --num-jobs 3
```

A helper to cut the per-iteration friction:

```bash
nova-dev() {
  git push -q &&
  export NOVA_WORKER_INSTALL_SPEC='supernova[{extra}] @ git+https://github.com/qdrant-labs/supernova@'$(git rev-parse HEAD) &&
  echo "workers → $(git rev-parse --short HEAD)"
}
# loop:  git commit -am wip && nova-dev && nova embed-dist ...
```

**Gotchas**

- **Push is required.** Workers clone the commit from GitHub — unpushed/uncommitted
  work is invisible to them. Develop on a feature branch.
- **Pools persist; `setup:` runs once per worker provision.** Re-launching into the
  *same* pool keeps the old install. Tear down between iterations:
  `sky jobs pool down <pool-name>`.
- **Re-point after each push** — `$(git rev-parse HEAD)` is captured when you `export`.
  Or pin to the branch (`@my-feature`) instead of a SHA, so the env var stays put and
  a `git push` is enough (fresh workers pull the branch's latest HEAD).
- **Back to normal:** `unset NOVA_WORKER_INSTALL_SPEC` → workers resume auto-pinning the
  controller's released version.

> Trade-off: the git-SHA loop requires a commit+push, so it can't test *uncommitted*
> changes. A variant that builds a wheel from your working tree and ships it via S3
> avoids that, at the cost of a build+upload per run.

### Rust tools (`storm` / `load`)

`storm` and `load` are Rust binaries, so they have their own knob. The worker `setup` runs
`cargo install --git <repo> --tag v<version> <crate>` by default (pinned to the controller's
release). Point it at un-released code with `NOVA_RUST_INSTALL_SPEC` — the literal
`cargo install` args, with `{binary}` substituted per crate (`nova-storm` / `nova-load`):

```bash
git checkout -b my-feature
git commit -am "wip" && git push          # cargo clones the ref from GitHub
# pin a branch — a plain `git push` is then enough, the env var stays put:
export NOVA_RUST_INSTALL_SPEC='--git https://github.com/qdrant-labs/supernova --branch my-feature {binary}'
# …or pin a sha (re-export after each push, like the Python loop):
export NOVA_RUST_INSTALL_SPEC="--git https://github.com/qdrant-labs/supernova --rev $(git rev-parse HEAD) {binary}"
nova dist storm configs/storm/test.yaml
```

Same gotchas as the Python path: **push is required** (cargo clones from GitHub —
uncommitted work is invisible to workers), and **pools persist** so `setup:` won't re-run —
`sky jobs pool down <pool>` between iterations. `unset NOVA_RUST_INSTALL_SPEC` to resume
auto-pinning the release tag.

> Each fresh worker compiles from source (a full Rust toolchain + build). Swapping the
> on-worker compile for a prebuilt-binary download is tracked in #16; until then there's no
> uncommitted-tree path for the Rust tools either.

### Local binary override (`NOVA_<TOOL>_BIN`)

For single-machine runs (`nova storm`, `nova load`), skip the install dance entirely and
point the dispatcher at a binary you just built. `nova <tool>` checks `NOVA_<TOOL>_BIN`
before falling back to `PATH`:

```bash
cargo build --release -p nova-storm -p nova-load
export NOVA_STORM_BIN="$(pwd)/target/release/nova-storm"
export NOVA_LOAD_BIN="$(pwd)/target/release/nova-load"
nova storm configs/storm/test.yaml      # runs your working-tree binary, no cargo install
```

This is the tightest loop for iterating on the Rust code itself — no commit, no push. It
only affects *this* machine; remote workers still come from `NOVA_RUST_INSTALL_SPEC`.

## Releasing

Publishing is tag-driven via `.github/workflows/release.yml` (PyPI trusted publishing,
no tokens). Bump `version` in `pyproject.toml`, then:

```bash
git tag v0.1.2 && git push origin v0.1.2
```

Once the new version is on PyPI, `-dist` workers auto-install it (no override needed).