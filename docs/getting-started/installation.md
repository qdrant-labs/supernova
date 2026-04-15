# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
git clone <repo-url>
cd vectorforge
uv sync
```

This installs everything needed for both embedding generation and data loading.

## Environment variables

Set the variables relevant to your workflow:

### Embedding pipeline

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI embedder |
| `AWS_ACCESS_KEY_ID` | S3 storage backend |
| `AWS_SECRET_ACCESS_KEY` | S3 storage backend |
| `HF_TOKEN` | HuggingFace Hub storage backend |

### Loading pipeline

| Variable | Required for |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Reading from S3 |
| `AWS_SECRET_ACCESS_KEY` | Reading from S3 |
| `AWS_SESSION_TOKEN` | S3 with AWS SSO (see [AWS SSO setup](../reference/aws-sso.md)) |
| `AWS_REGION` | S3 region (defaults to `us-east-1`) |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |

## SkyPilot setup

SkyPilot is used to parallelize both embedding generation (GPU instances) and vector store loading (CPU spot instances).

```bash
# SkyPilot is included in vectorforge dependencies
sky check aws
```

SkyPilot requires IAM permissions to launch EC2 instances. See the [SkyPilot AWS permissions docs](https://docs.skypilot.co/en/stable/cloud-setup/cloud-permissions/aws.html#minimal-permissions).

## Verify installation

```bash
vf embed --help
vf embed-dist --help
vf load --help
vf load-dist --help

uv run pytest tests/ -v
```
