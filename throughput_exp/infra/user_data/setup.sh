#!/bin/bash
set -euo pipefail
exec > /var/log/gpu-setup.log 2>&1

echo "=== Setup starting (DLAMI — drivers + PyTorch already installed) ==="

# setup uv
# curl -LsSf https://astral.sh/uv/install.sh | sh

# venv
# uv venv

# uv pip install \
#     "datasets" \
#     "pyarrow" \
#     "aiobotocore" \
#     "pyyaml" \
#     "tiktoken>=0.12.0" \
#     "tqdm" \
#     "huggingface_hub" \
#     "duckdb>=1.5.1" \
#     "sentence-transformers" \
#     "torch" \
#     "transformers<5" \
#     "boto3"

# Signal completion
touch /home/ec2-user/.setup-complete
echo "=== Setup complete $(date) ==="