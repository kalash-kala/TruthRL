#!/bin/bash
#################################################################################
# TRUTHRL - Verifier Deployment (A100)
# Purpose: Launches a vLLM server (Gemma-27B) on a single A100 GPU node.
# Output: Provides an internal IP/URL for training scripts to connect to.
#################################################################################

#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=verifier_gemma_a100
#SBATCH --output=slurm_logs/logs/%x_%j.out
#SBATCH --error=slurm_logs/errors/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Configuration
MODEL_PATH="/home/sriramg/kalashabhayk/models/gemma-3-27b-it"
unset ROCR_VISIBLE_DEVICES

# Get the Internal IP of this node
INTERNAL_IP=$(hostname -I | awk '{print $1}')
echo "----------------------------------------------------------------"
echo "VERIFIER STARTING ON A100 NODE: $(hostname)"
echo "INTERNAL IP: $INTERNAL_IP"
echo "URL FOR TRAINER SCRIPT: http://$INTERNAL_IP:8000/v1"
echo "----------------------------------------------------------------"

echo "DEBUG: Checking GPUs..."
nvidia-smi

# Start vLLM server
# TP=1 fits on a single A100 (40GB or 80GB)
# Note: If it's a 40GB A100, we might need --quantization awq if it OOMs
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name google/gemma-3-27b-it \
    --tensor-parallel-size 1 \
    --port 8000 \
    --api-key token-abc123 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --host 0.0.0.0
