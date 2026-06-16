# Development

How supernova is laid out, and how to test un-released changes on a real fleet.

## Package structure

`supernova` is a pip-installable package exposing the `nova` CLI
(`nova = supernova.cli.cli:main`):

```text
supernova/
├── cli/                          # the `nova` CLI
│   ├── cli.py                    # entry point (LazyGroup) + subcommand map
│   ├── run_<verb>.py             # one per command: embed, load, storm, experiment
│   ├── run_<verb>_distributed.py # the `-dist` SkyPilot dispatcher for each verb
│   └── skypilot_utils.py         # shared dispatch helpers (worker bootstrap, pools,
│                                 #   make_run_dir, nova_home, config_mount, ...)
├── sources/        # dataset sources to embed (HuggingFace)        [ABC: DatasetSource]
├── embedders/      # dense / sparse / multivector embedders + the [ABCs per family]
│                   #   streaming embed pipeline (buffer/runner/worker)
├── storage/        # embedding output sinks: s3 / local / hf       [ABC: StorageBackend]
├── destinations.py # s3:// and hf:// URI helpers
├── loader/         # load pre-embedded parquet into vector stores
│   ├── datasource/ #   read parquet from s3 / hf                   [ABC: DataReader]
│   └── vectorstore/#   write to a vector DB (Qdrant)               [ABC: VectorStore]
├── storm/          # load-test a vector store (the `nova storm` workload)
├── experiment/     # compose units over a timeline (workload tests)
└── models.py, utils.py
```

### The three-layer pattern

Every workload follows the same shape — copy it when adding a new one:

1. **Core library** — vendor-agnostic ABCs you implement per backend (a source, an
   embedder, a store). Pure, importable, no cloud.
2. **Local CLI verb** (`nova embed`, `nova load`) — runs a single worker in-process.
   This is the unit of work; it shards itself when given `--num-jobs` + `--job-rank`
   (rank defaults to `$SKYPILOT_JOB_RANK`).
3. **`-dist` wrapper** (`nova embed-dist`) — provisions a SkyPilot pool and submits N
   copies of the *same* local verb, one per shard.

Workers do **not** receive your code by file-sync. The `-dist` wrapper makes each
worker `pip install "supernova[<extra>]==<the controller's version>"` from PyPI, so
the controller and its workers always run identical code. Dependency extras
(`embed`, `load`, `storm`, `dist`) are declared in `pyproject.toml`.

### Local state

Run metadata — generated pool/job YAMLs, manifests, the staged config — lives under
`~/.nova/runs/` (override with `$NOVA_HOME`). It's intentionally outside any repo so an
installed `nova` writes to a stable location no matter where it's invoked.

## Dev mode: testing un-released changes on a fleet

Workers install the *published* version by default, so local edits never reach them.
To run your working changes distributed, override the worker install source with the
`NOVA_WORKER_INSTALL_SPEC` env var — a PEP 508 spec in which `{extra}` is substituted
per command.

**Always iterate locally first** (no cloud, instant):

```bash
nova embed configs/embedder/test.yaml
```

**Then distributed, off a pushed git commit:**

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

## Releasing

Publishing is tag-driven via `.github/workflows/release.yml` (PyPI trusted publishing,
no tokens). Bump `version` in `pyproject.toml`, then:

```bash
git tag v0.1.2 && git push origin v0.1.2
```

Once the new version is on PyPI, `-dist` workers auto-install it (no override needed).