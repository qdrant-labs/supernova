#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/submit_batch.sh [config1.yaml config2.yaml ...]
#   ./scripts/submit_batch.sh --gpu configs/local_model.yaml
# If no configs specified, submits all configs/*.yaml

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PROJECT="vectorforge"
GPU=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Get ECR URL from terraform
ECR_URL=$(cd terraform && terraform output -raw ecr_repository_url)

if [ "$GPU" = true ]; then
    QUEUE_ARN=$(cd terraform && terraform output -raw gpu_job_queue_arn)
    JOB_DEF="${PROJECT}-gpu"
    TAG="latest-gpu"
else
    QUEUE_ARN=$(cd terraform && terraform output -raw job_queue_arn)
    JOB_DEF="${PROJECT}"
    TAG="latest"
fi

echo "Building Docker image..."
docker build -t "$PROJECT:$TAG" .

echo "Pushing to ECR"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URL"
docker tag "$PROJECT:$TAG" "$ECR_URL:$TAG"
docker push "$ECR_URL:$TAG"

# Determine which configs to submit
if [ $# -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=(configs/*.yaml)
fi

echo "Submitting ${#CONFIGS[@]} jobs (gpu=$GPU)"
for config in "${CONFIGS[@]}"; do
    job_name="${PROJECT}-$(basename "$config" .yaml)"
    echo "Submitting: $job_name ($config)"

    aws batch submit-job \
        --job-name "$job_name" \
        --job-queue "$QUEUE_ARN" \
        --job-definition "$JOB_DEF" \
        --container-overrides "{\"command\": [\"$config\"]}" \
        --region "$REGION"
done

echo "All jobs submitted"
echo "Monitor at: https://${REGION}.console.aws.amazon.com/batch/home?region=${REGION}#jobs"
