#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/submit_batch.sh [config1.yaml config2.yaml ...]
# If no args, submits all configs/*.yaml

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PROJECT="vectorforge"

# Get ECR URL from terraform
ECR_URL=$(cd terraform && terraform output -raw ecr_repository_url)
QUEUE_ARN=$(cd terraform && terraform output -raw job_queue_arn)

echo "Building Docker image..."
docker build -t "$PROJECT" .

echo "Pushing to ECR"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URL"
docker tag "$PROJECT:latest" "$ECR_URL:latest"
docker push "$ECR_URL:latest"

# Determine which configs to submit
if [ $# -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=(configs/*.yaml)
fi

echo "Submitting ${#CONFIGS[@]} jobs"
for config in "${CONFIGS[@]}"; do
    job_name="${PROJECT}-$(basename "$config" .yaml)"
    echo "Submitting: $job_name ($config)"

    aws batch submit-job \
        --job-name "$job_name" \
        --job-queue "$QUEUE_ARN" \
        --job-definition "$PROJECT" \
        --container-overrides "{\"command\": [\"$config\"]}" \
        --region "$REGION"
done

echo "All jobs submitted"
echo "Monitor at: https://${REGION}.console.aws.amazon.com/batch/home?region=${REGION}#jobs"
