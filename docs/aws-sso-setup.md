# AWS SSO Setup for vectorforge

## Overview

vectorforge uses AWS credentials in two contexts:

1. **Local** — running `vectorforge-load`, `query_example.py`, etc. from your machine
2. **Modal** — running `modal_batch.py`, `modal_import_cohere.py` in cloud containers

AWS SSO gives you temporary credentials tied to your company identity. The challenge is that Modal containers can't run `aws sso login` interactively — they need explicit credentials passed as secrets.

## 1. Configure AWS SSO Profile

One-time setup. Run:

```bash
aws configure sso
```

You'll be prompted for:
- **SSO start URL**: your company's SSO URL (e.g. `https://yourcompany.awsapps.com/start`)
- **SSO region**: region of SSO (e.g. `us-east-1`)
- **Account**: select your sandbox account
- **Role**: select your role (e.g. `AdministratorAccess`, `PowerUserAccess`)
- **Profile name**: pick something short, e.g. `sandbox`

This creates a profile in `~/.aws/config`.

## 2. Local Usage

```bash
# Log in (opens browser, lasts 8-12 hours)
aws sso login --profile sandbox

# Option A: Set the profile
export AWS_PROFILE=sandbox

# Option B: Export temporary credentials as env vars
eval "$(aws configure export-credentials --profile sandbox --format env)"
```

**Option A** is simpler for local work — boto3/aiobotocore/DuckDB all pick up the profile automatically.

**Option B** exports `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` as env vars. Useful when a tool doesn't support profiles (e.g. DuckDB httpfs).

### DuckDB / vectorforge-load

DuckDB's httpfs reads `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from env vars. It also needs `AWS_SESSION_TOKEN` for SSO temporary credentials. Update the S3 reader to pass the session token if present.

After that, the workflow is:

```bash
aws sso login --profile sandbox
eval "$(aws configure export-credentials --profile sandbox --format env)"
vectorforge-load configs/loader/cohere200M.yaml
```

## 3. Modal Usage

Modal containers need credentials as environment variables. Since SSO tokens expire every 8-12 hours, you need to refresh the Modal secret before each run.

### Refresh Script

Create a helper script that logs in, exports creds, and updates the Modal secret:

```bash
#!/bin/bash
# scripts/refresh-modal-secrets.sh

set -e

PROFILE="${AWS_PROFILE:-sandbox}"

# Ensure SSO session is active
aws sso login --profile "$PROFILE"

# Export temporary credentials
eval "$(aws configure export-credentials --profile "$PROFILE" --format env)"

# Update Modal secret with fresh creds
modal secret create vectorforge-secrets \
  AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
  AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
  OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  QDRANT_URL="${QDRANT_URL:-}" \
  QDRANT_API_KEY="${QDRANT_API_KEY:-}" \
  --force

echo "Modal secret updated."
```

### Workflow

```bash
# Refresh creds + Modal secret
./scripts/refresh-modal-secrets.sh

# Run Modal jobs (within the 8-12hr window)
modal run modal_import_cohere.py --configs en
modal run modal_batch.py --config configs/embedder/arxiv.yaml
```

**Important:** The `--force` flag overwrites the existing secret. The temporary credentials last 8-12 hours, so refresh before long-running jobs.

## 4. Code Changes Required

### DuckDB S3 reader needs AWS_SESSION_TOKEN

SSO temporary credentials include a session token. DuckDB needs it set explicitly.

In `vectorforge/loader/datasource/s3.py`, the `_configure_connection` method needs:

```python
session_token = os.environ.get("AWS_SESSION_TOKEN", "")
if session_token:
    conn.execute(f"SET s3_session_token = '{session_token}';")
```

Same for `scripts/query_example.py`.

### aiobotocore (S3Backend) — no changes needed

aiobotocore automatically reads `AWS_SESSION_TOKEN` from the environment, so `vectorforge/storage/s3.py` works as-is.

## 5. Quick Reference

| Context | How credentials flow |
|---------|---------------------|
| Local (boto3/aiobotocore) | `AWS_PROFILE=sandbox` or exported env vars |
| Local (DuckDB httpfs) | Exported env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) |
| Modal | `vectorforge-secrets` secret → env vars in container |

| Command | What it does |
|---------|-------------|
| `aws sso login --profile sandbox` | Authenticate (8-12hr session) |
| `aws configure export-credentials --profile sandbox --format env` | Get temp creds as env vars |
| `./scripts/refresh-modal-secrets.sh` | Refresh Modal secret with fresh SSO creds |