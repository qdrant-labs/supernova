# supernova — commands

Every user-facing `nova-*` command lives here, **organized by command, not by
language**. A command may be a Rust binary or a Python console script; the
git-style `nova` dispatcher treats them identically. Commands are not
code-coupled across the language boundary — they meet only at runtime through
the dispatcher and shared data contracts (the YAML config, parquet/point
formats). Backend implementations behind the load/storm command contracts live
one level up, in `../backends`.

## The dispatcher model

`nova <cmd> [args...]` finds an executable named `nova-<cmd>` on `PATH` and
replaces itself with it (`os.execv`). A command can be implemented in any
language:

- **Rust** — a binary like `nova-load`, installed by `cargo install`.
- **Python** — a console script like `nova-embed`, installed by `pip`/`uv`.

The dispatcher doesn't know or care which. To add a command, just put a
`nova-<name>` executable on `PATH`.

## Commands

| Command        | Language | Provides                       | Install                                        |
|----------------|----------|--------------------------------|------------------------------------------------|
| `nova` (root)  | Python   | the dispatcher                 | `uv pip install -e .` (repo root)              |
| `nova-load`    | Rust     | shim → `nova-load-qdrant`      | `cargo install --path commands/nova-load`      |
| `nova-storm`   | Rust     | shim → `nova-storm-qdrant`     | `cargo install --path commands/nova-storm`     |
| `nova-contract`| Rust     | backend conformance checker    | `cargo install --path commands/nova-contract`  |
| `nova-inspect` | Rust     | parquet schema / vector count  | `cargo install --path commands/nova-inspect`   |
| `nova-embed`   | Python   | embedding pipeline (heavy ML)  | `uv pip install -e 'commands/nova-embed[embed]'` |
| `nova-bf`      | Python   | brute-force ground truth       | `uv pip install -e 'commands/nova-bf[compute]'`  |
| `nova-opt`     | Python   | cost/recall BO tuner (WIP)     | `uv pip install -e commands/nova-opt`          |
| `nova-sweep`   | Python   | parameter-sweep orchestrator   | `uv pip install -e commands/nova-sweep`        |
| `nova-dist`    | Python   | SkyPilot fleet orchestration   | `uv pip install -e commands/nova-dist`         |

The **dispatcher lives at the repo root** (`pyproject.toml` + `src/cli/`), so
`uv pip install -e .` from the root installs `nova`. It's the project's front
door and the spine of the polyglot tool, so it sits at the top rather than
buried as just-another-command. It's deliberately dependency-free. Every other
command is installed only where needed — like `git-*` subcommands. (`nova-opt`
is a work-in-progress tracked on another branch; it may be absent from a given
checkout.)

Each `Makefile` target installs one command (`make embed`, `make bf`, …);
`make load`/`make storm` install both the shim and its Qdrant backend. See the
top-level `AGENTS.md` for the command/backend/contract architecture.

## nova-embed

Embedding generation (chunkers → embedders → storage), streamed from a dataset
source and written as parquet. Honors the same `--num-jobs` / `--job-rank`
distributed contract as `nova-load`: each rank computes its own `offset`/`limit`
slice of the dataset (from `--job-rank`, or `$SKYPILOT_JOB_RANK`).

Config is validated with **pydantic** (`nova_embed.config`): `pipeline` knobs are
typed with defaults in one place, while `source`/`*_embedder`/`storage` carry a
`type` plus flexible backend-specific kwargs. `${VAR}` / `${VAR:-default}`
references are env-expanded, matching the Rust crates.

The base package is light (pydantic, pyarrow, …); the actual ML stack (torch,
sentence-transformers, …) is the `embed` extra:

```sh
uv pip install -e 'commands/nova-embed[embed]'
nova embed configs/embedder/test.yaml --num-jobs 50 --job-rank $SKYPILOT_JOB_RANK
nova embed configs/embedder/test.yaml --dry-run
```

## Dev setup

```sh
uv pip install -e .                              # the `nova` dispatcher (from repo root)
uv pip install -e commands/nova-embed            # embedding command (heavy)
cargo install --path commands/nova-load          # the `nova-load` shim
cargo install --path backends/nova-load/qdrant   # the `nova-load-qdrant` backend

nova --help        # lists discovered nova-* commands
nova load inspect configs/loader/test.yaml
nova embed ...
```

> Ensure your Python user-scripts dir (e.g. `~/.local/bin` or
> `~/Library/Python/X.Y/bin`) and `~/.cargo/bin` are on `PATH`.
