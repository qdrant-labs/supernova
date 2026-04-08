#!/bin/bash
# scripts/refresh_modal_secrets.sh

set -e

PROFILE="${AWS_PROFILE:-qdrant-sandbox}"

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