# AWS SSO Setup for vectorforge

## Overview

vectorforge uses AWS credentials for S3 storage and SkyPilot instance provisioning. AWS SSO gives you temporary credentials tied to your company identity.

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

**Option A** is simpler for local work -- boto3/aiobotocore/DuckDB all pick up the profile automatically.

**Option B** exports `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` as env vars. Useful when a tool doesn't support profiles (e.g. DuckDB httpfs).

### DuckDB / vf load

DuckDB's httpfs reads `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from env vars. It also needs `AWS_SESSION_TOKEN` for SSO temporary credentials. Update the S3 reader to pass the session token if present.

After that, the workflow is:

```bash
aws sso login --profile sandbox
eval "$(aws configure export-credentials --profile sandbox --format env)"
vf load configs/loader/ccnews_bge_large.yaml
```

## 3. SkyPilot Usage

SkyPilot jobs receive credentials via `--env` flags at launch time (never written to disk). The `vf embed-dist` and `vf load-dist` commands handle this automatically by forwarding relevant env vars from your shell.

Before launching distributed jobs:

```bash
aws sso login --profile sandbox
eval "$(aws configure export-credentials --profile sandbox --format env)"
vf embed-dist configs/embedder/my_dataset.yaml
```

**Important:** SSO temporary credentials last 8-12 hours. Refresh before long-running jobs.

## 4. Code Changes Required

### DuckDB S3 reader needs AWS_SESSION_TOKEN

SSO temporary credentials include a session token. DuckDB needs it set explicitly.

In `vectorforge/loader/datasource/s3.py`, the `_configure_connection` method needs:

```python
session_token = os.environ.get("AWS_SESSION_TOKEN", "")
if session_token:
    conn.execute(f"SET s3_session_token = '{session_token}';")
```

### aiobotocore (S3Backend) -- no changes needed

aiobotocore automatically reads `AWS_SESSION_TOKEN` from the environment, so `vectorforge/storage/s3.py` works as-is.

## 5. Quick Reference

| Context | How credentials flow |
|---------|---------------------|
| Local (boto3/aiobotocore) | `AWS_PROFILE=sandbox` or exported env vars |
| Local (DuckDB httpfs) | Exported env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) |
| SkyPilot | Forwarded via `--env` flags at job launch |

| Command | What it does |
|---------|-------------|
| `aws sso login --profile sandbox` | Authenticate (8-12hr session) |
| `aws configure export-credentials --profile sandbox --format env` | Get temp creds as env vars |
