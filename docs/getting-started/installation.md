# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
git clone <repo-url>
cd supernova
uv sync
```

`uv sync` alone installs the **base** deps (pyarrow, numpy, click, boto3, huggingface_hub, etc.) — enough to run `nova --help`, parse configs, and use the `supernova.destinations` / URI helpers.

The actual pipelines live under optional extras so embed workers don't pull in qdrant-client and load workers don't pull in torch:

| Extra | Installs | Use when |
|-------|----------|----------|
| `embed` | sentence-transformers, torch, transformers, FlagEmbedding, fastembed, openai, tiktoken, aiobotocore | Running `nova embed` workers locally |
| `load` | duckdb, qdrant-client | Running `nova load` (loader workers) |
| `storm` | qdrant-client, duckdb | Running `nova storm` load-test workers |
| `dist` | skypilot[aws] | Dispatching distributed jobs from your laptop / Hetzner box |

Pick the extras that match your role. Common combinations:

```bash
# Local laptop dispatching distributed embed + load
uv sync --extra dist --extra load

# A single embed worker
uv sync --extra embed

# A single load worker
uv sync --extra load

# Everything (heavy)
uv sync --all-extras
```

The dispatch CLIs (`nova embed-dist`, `nova load-dist`, `nova storm-dist`) bake the right `uv sync --extra ...` into the pool's `setup:` script, so workers install the right slice automatically.

## Environment variables

Set the variables relevant to your workflow.

### Embedding pipeline

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI embedder |
| `AWS_ACCESS_KEY_ID` | S3 storage backend |
| `AWS_SECRET_ACCESS_KEY` | S3 storage backend |
| `HF_TOKEN` | HuggingFace Storage Buckets (write) and reads from `hf://buckets/...` or `hf://datasets/...` (private source datasets / legacy corpora) |

### Loading pipeline

| Variable | Required for |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Reading from S3 |
| `AWS_SECRET_ACCESS_KEY` | Reading from S3 |
| `AWS_SESSION_TOKEN` | S3 with AWS SSO (see [AWS SSO setup](../reference/aws-sso.md)) |
| `AWS_REGION` | S3 region (defaults to `us-east-1`) |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `HF_TOKEN` | Reading private corpora from `hf://buckets/...` (new) or `hf://datasets/...` (legacy) |

## SkyPilot setup

SkyPilot is used to parallelize embedding generation (GPU instances), loading (CPU spot instances), and load testing (`nova storm-dist`). It's only needed if you'll be **dispatching** distributed jobs — workers themselves don't need it.

```bash
uv sync --extra dist
sky check aws
```

SkyPilot requires IAM permissions to launch EC2 instances. See the [SkyPilot AWS permissions docs](https://docs.skypilot.co/en/stable/cloud-setup/cloud-permissions/aws.html#minimal-permissions).

## Verify installation

```bash
nova --help                # lists every subcommand; should run in well under a second
nova embed --help
nova load --help
nova storm --help

uv run pytest tests/ -v
```

The CLI is a click group with **lazy subcommand loading** (`cli/cli.py:LazyGroup`) — `nova --help` doesn't import torch or qdrant-client, only the relevant module is loaded when you actually run a subcommand.
