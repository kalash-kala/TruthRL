#!/bin/bash

#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=truthrl_joint_h200
#SBATCH --output=slurm_logs/logs/%x_%j.out
#SBATCH --error=slurm_logs/errors/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Get the directory where this script is located (ultimate path-agnostic way)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Ensure log directories exist locally
mkdir -p slurm_logs/logs slurm_logs/errors

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Configuration
MODEL_PATH="/home/sriramg/kalashabhayk/models/gemma-3-27b-it"
# Training script expects this variable or defaults to localhost
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="token-abc123"
unset ROCR_VISIBLE_DEVICES

echo "----------------------------------------------------------------"
echo "STARTING JOINT JOB ON H200 NODE: $(hostname)"
echo "SCRIPT LOCATION: $SCRIPT_DIR"
echo "----------------------------------------------------------------"

echo "DEBUG: Checking GPUs..."
nvidia-smi

# 1. Start vLLM server on GPU 0
echo "Starting vLLM Verifier on GPU 0..."
# Use absolute log path for background process to be safe
VLLM_LOG="$SCRIPT_DIR/slurm_logs/logs/verifier_joint_$SLURM_JOB_ID.log"
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name google/gemma-3-27b-it \
    --tensor-parallel-size 1 \
    --port 8000 \
    --api-key $OPENAI_API_KEY \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 > "$VLLM_LOG" 2>&1 &

VERIFIER_PID=$!
echo "Verifier PID: $VERIFIER_PID"
echo "Verifier log: $VLLM_LOG"

# 2. Wait for Verifier to be ready
echo "Waiting for Verifier to be ready..."
# Giving it plenty of time (60 mins) due to slow IO
MAX_RETRIES=360
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Try connecting to the models endpoint
    if curl -s -o /dev/null -H "Authorization: Bearer $OPENAI_API_KEY" http://localhost:8000/v1/models; then
        echo "Verifier is ready!"
        break
    else
        echo "Waiting for verifier... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Timeout waiting for verifier. Check logs at $VLLM_LOG"
    echo "Tail of verifier log:"
    tail -n 20 "$VLLM_LOG"
    kill $VERIFIER_PID
    exit 1
fi

# 3. Start Training on GPU 1
echo "Starting Training on GPU 1..."
# Explicitly setting CUDA_VISIBLE_DEVICES to 1 so the training script sees only the second GPU.
CUDA_VISIBLE_DEVICES=1 bash train_grpo_h200_client.sh

EXIT_CODE=$?

# 4. Cleanup
echo "Training finished with exit code $EXIT_CODE. Stopping verifier."
kill $VERIFIER_PID
wait $VERIFIER_PID

exit $EXIT_CODE

