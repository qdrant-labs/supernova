# Installation

supernova is polyglot. The `nova` CLI is a git-style dispatcher — `nova <cmd>`
execs a `nova-<cmd>` executable on your `PATH` — so you install the dispatcher
once, then add only the sub-tools you need.

## Requirements

- [uv](https://docs.astral.sh/uv/) — for the Python pieces (`nova`, `nova embed`)
- [Rust / cargo](https://rustup.rs/) — for the Rust tools (`nova load`, `nova storm`)
- Python 3.11+

## The fast path

```bash
git clone <repo-url> supernova && cd supernova
make all
```

`make all` installs the `nova` dispatcher plus all sub-tools. Then put the
install dirs on your `PATH` so `nova` can find the sub-tools:

```bash
export PATH="$HOME/.cargo/bin:$PATH"     # Rust binaries: nova-load, nova-storm
export PATH="$HOME/.local/bin:$PATH"     # uv/pip user scripts: nova, nova-embed
```

Verify:

```bash
nova --help        # lists every nova-* tool found on PATH
```

## Installing piece by piece

Each `make` target maps to one tool. Install only what a given machine needs.

| Target | Installs | Command it provides |
|--------|----------|---------------------|
| `make cli`   | the dispatcher (zero deps, instant) | `nova` |
| `make embed` | Python ML stack (torch, sentence-transformers, …) | `nova embed` |
| `make load`  | Rust binary → `~/.cargo/bin` | `nova load` |
| `make storm` | Rust binary → `~/.cargo/bin` | `nova storm` |

## Environment variables

Set the variables relevant to your workflow. Configs reference them with
`${VAR}` (or `${VAR:-default}`), expanded at load time. Some examples of what each tool might need:

### Embedding (`nova embed`)

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI embedder |
| `HF_TOKEN` | Private HF source datasets; writing to `hf://buckets/...` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 storage backend |

### Loading (`nova load`) & storm (`nova storm`)

| Variable | Required for |
|----------|-------------|
| `QDRANT_URL` / `QDRANT_API_KEY` | The Qdrant cluster |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Reading corpora from S3 |
| `AWS_SESSION_TOKEN` | S3 via AWS SSO (temporary credentials) |
| `AWS_REGION` | S3 region (defaults to `us-east-1`) |

## Distributed

There's nothing extra to install for distributed runs. Each tool shards itself
from `--num-jobs` / `--job-rank` (rank defaults to `$SKYPILOT_JOB_RANK`). Your
orchestrator provisions the nodes and invokes `nova <tool>` on each — see the
[Quickstart](quickstart.md#distributed) for the pattern.

## Verify

```bash
nova --help
nova embed --help
nova load --help     # subcommands: run / prepare / load / finalize / inspect
nova storm --help
```
