# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Basic setup

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
| `AWS_SESSION_TOKEN` | S3 with AWS SSO (see [AWS SSO setup](aws-sso-setup.md)) |
| `AWS_REGION` | S3 region (defaults to `us-east-1`) |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |

## Modal setup (for distributed embedding)

Modal is used to parallelize embedding generation across GPUs/CPUs in the cloud.

```bash
pip install modal
modal setup
```

Create a secret group with your credentials:

```bash
modal secret create vectorforge-secrets \
  OPENAI_API_KEY=$OPENAI_API_KEY \
  AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  AWS_DEFAULT_REGION=us-east-1 \
  HF_TOKEN=$HF_TOKEN
```

If using AWS SSO, refresh secrets before each run with:

```bash
eval "$(aws configure export-credentials --profile your-profile --format env)"
modal secret create vectorforge-secrets \
  AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
  AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1} \
  OPENAI_API_KEY=${OPENAI_API_KEY:-} \
  HF_TOKEN=${HF_TOKEN:-} \
  --force
```

## SkyPilot setup (for distributed loading)

SkyPilot is used to parallelize vector store loading across EC2 spot instances.

```bash
# SkyPilot is included in vectorforge dependencies
sky check aws
```

SkyPilot requires IAM permissions to launch EC2 instances. See the [SkyPilot AWS permissions docs](https://docs.skypilot.co/en/stable/cloud-setup/cloud-permissions/aws.html#minimal-permissions) or ask your AWS admin to set up the `skypilot-v1` instance profile.

## Verify installation

```bash
# Check CLI tools are available
vectorforge --help
vectorforge-load --help
vectorforge-load-distributed --help

# Run tests
uv run pytest tests/ -v
```
