# Installation

`nova` is a git-style dispatcher: `nova <cmd>` runs a `nova-<cmd>` executable
from your `PATH`. Install the dispatcher once, then only the tools a machine
needs.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- [Rust / cargo](https://rustup.rs/) for `nova load`, `nova storm`,
  `nova inspect`, and the native extension `nova bf` builds
- A CUDA GPU for `nova bf compute` and for embedding at any real scale
  (`nova embed` falls back to MPS or CPU)
- `protoc`, only if you build the optional Milvus backend

## Install everything

```bash
git clone <repo-url> supernova && cd supernova
uv venv && source .venv/bin/activate   # the Python tools install into this venv
make all
export PATH="$HOME/.cargo/bin:$PATH"   # Rust tools land here
nova --help                            # lists every nova-* tool it can find
```

The Python tools install into the active virtualenv, so activate it in every
shell you use `nova` from.

## Install one tool at a time

| Target | Command | Language | Notes |
|---|---|---|---|
| `make cli` | `nova` | Python | The dispatcher. No dependencies. |
| `make embed` | `nova embed` | Python | Pulls torch, sentence-transformers, and more. |
| `make load` | `nova load` | Rust | Installs to `~/.cargo/bin`. |
| `make storm` | `nova storm` | Rust | Installs to `~/.cargo/bin`. |
| `make inspect` | `nova inspect` | Rust | Dev tool: vector count and parquet schema. |
| `make bf` | `nova bf` | Python + Rust | Pulls torch and builds `nova-textscan` and `rust_fetch` (the native reader behind `NOVA_BF_NATIVE_DENSE` / `NOVA_BF_NATIVE_FETCH`). |
| `make sweep` | `nova sweep` | Python | Runs on the controller; needs `nova load` and `nova storm` on `PATH`. |
| `make dist` | `nova dist` | Python | Runs on the controller; pulls SkyPilot. |

A machine that only runs `nova bf merge` doesn't need torch or the native
reader:

```bash
uv pip install -e python/nova-bf ./crates/nova-textscan
```

### Other vector stores

`nova load` and `nova storm` build with Qdrant support only by default. To add
the other backends:

```bash
make load  LOAD_FEATURES=elastic,opensearch,milvus    # milvus needs protoc
make storm STORM_FEATURES=elastic,opensearch,milvus
```

## Fleet runs

There's nothing extra to install on workers. `nova dist` installs the tools on
each node it launches, so only the controller needs `make dist`. See
[Distributed](../distributed.md).

## Environment variables

Configs read these through `${VAR}` or `${VAR:-default}`. Set only the ones
your run uses.

| Variable | Used for |
|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | The Qdrant cluster (`nova load`, `nova storm`, `nova sweep`) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Reading or writing S3 |
| `AWS_SESSION_TOKEN` | S3 with temporary credentials |
| `AWS_REGION` | S3 region (defaults to `us-east-1`) |
| `HF_TOKEN` | Private Hugging Face datasets, writing to `hf://` |
| `OPENAI_API_KEY` | The OpenAI embedder |

## Verify

```bash
nova --help          # every nova-* tool found on PATH
nova load --help     # run, prepare, load, finalize, reindex, delete, inspect
nova bf --help       # compute, merge
```

## Build these docs

```bash
make docs            # live preview at http://localhost:8000 (needs uv)
make docs-build      # static site in site/
```
