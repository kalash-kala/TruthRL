#!/bin/bash
#################################################################################
# TRUTHRL - Verifier Deployment (ADA6000)
# Purpose: Launches a vLLM server (Gemma-27B) split across 2 ADA6000 GPUs.
# Features: Uses Tensor Parallel (TP=2) to fit weights on 48GB VRAM cards.
#################################################################################

#SBATCH --partition=ada
#SBATCH --gres=gpu:ADA6000:2
#SBATCH --job-name=verifier_gemma
#SBATCH --output=slurm_logs/logs/%x_%j.out
#SBATCH --error=slurm_logs/errors/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=90G
#SBATCH --cpus-per-task=16

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Configuration
MODEL_PATH="/home/sriramg/kalashabhayk/models/gemma-3-27b-it"
unset ROCR_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1  <-- Removed to let Slurm manage device visibility

# Get the Internal IP of this node
INTERNAL_IP=$(hostname -I | awk '{print $1}')
echo "----------------------------------------------------------------"
echo "VERIFIER STARTING ON NODE: $(hostname)"
echo "INTERNAL IP: $INTERNAL_IP"
echo "URL FOR H200 SCRIPT: http://$INTERNAL_IP:8000/v1"
echo "----------------------------------------------------------------"

# Start vLLM server
# TP=2 allows the 27B model to fit across two 48GB ADA6000 cards
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name google/gemma-3-27b-it \
    --tensor-parallel-size 2 \
    --port 8000 \
    --api-key token-abc123 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9
