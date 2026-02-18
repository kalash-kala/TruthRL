#!/bin/bash
set -e

# Configuration
# For Gemma 27B: Use 'google/gemma-3-27b-it' and '--tensor-parallel-size 1'
export MODEL_ID="google/gemma-3-27b-it" 
export TP_SIZE=1

# Ensure HF Token is provided
# Try to get token from argument, or fall back to stored token file
if [ -n "$1" ]; then
    HF_TOKEN=$1
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
    echo "Using stored Hugging Face token from cache."
else
    echo "Error: Hugging Face token not found."
    echo "Usage: $0 <huggingface_token>"
    echo "Or login first using: hf auth login"
    exit 1
fi

echo "Deploying $MODEL_ID with vllm..."

# Run the docker container
sudo docker run -d --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    --env HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model $MODEL_ID \
    --tensor-parallel-size $TP_SIZE \
    --dtype auto \
    --api-key token-abc123

echo "Deployment started. The model will be available at http://localhost:8000/v1/models once loaded."
echo "You can check the logs with: sudo docker logs <container_id>"
