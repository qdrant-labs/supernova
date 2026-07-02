# AWS SSO Setup for supernova

## Overview

supernova uses AWS credentials for S3 storage and SkyPilot instance provisioning. AWS SSO gives you temporary credentials tied to your company identity.

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

### DuckDB / nova load

DuckDB's httpfs reads `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from env vars. It also needs `AWS_SESSION_TOKEN` for SSO temporary credentials. Update the S3 reader to pass the session token if present.

After that, the workflow is:

```bash
aws sso login --profile sandbox
eval "$(aws configure export-credentials --profile sandbox --format env)"
nova load configs/loader/ccnews_bge_large.yaml
```

## 3. SkyPilot Usage

SkyPilot jobs receive credentials via `--env` flags at launch time (never written to disk). The `nova embed-dist` and `nova load-dist` commands handle this automatically by forwarding relevant env vars from your shell.

Before launching distributed jobs:

```bash
aws sso login --profile sandbox
eval "$(aws configure export-credentials --profile sandbox --format env)"
nova embed-dist configs/embedder/my_dataset.yaml
```

**Important:** SSO temporary credentials last 8-12 hours. Refresh before long-running jobs.

## 4. How credentials reach each component

### DuckDB S3 reader

DuckDB's httpfs reads `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` directly from environment variables. SSO temporary credentials add `AWS_SESSION_TOKEN`, which DuckDB needs set explicitly — `S3DataReader._configure_connection` handles this:

```python
# supernova/loader/datasource/s3.py
session_token = os.environ.get("AWS_SESSION_TOKEN", "")
if session_token:
    conn.execute(f"SET s3_session_token = '{session_token}';")
```

So as long as you `eval "$(aws configure export-credentials --profile sandbox --format env)"` before running `nova load`, DuckDB picks everything up.

### object store (S3 storage backend)

The embed pipeline's `object_store` backend writes S3 via `obstore`, using
`obstore`'s boto3 credential provider — so it reads `AWS_SESSION_TOKEN` (and the
rest) from the environment through boto3's standard credential chain, same as
`cli/run_push_hf.py`. No special handling needed.

### SkyPilot dispatch

`cli/skypilot_utils.build_env_flags()` forwards `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`, `AWS_DEFAULT_REGION` from your shell to every job/pool launched. Credentials are passed via `--env` flags at launch time and never written to disk.

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
