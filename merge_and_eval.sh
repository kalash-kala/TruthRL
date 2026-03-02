#!/bin/bash

# Script to merge sharded model parameters and run evaluation
# Usage: ./merge_and_eval.sh <checkpoint_dir> [run_name]

CHECKPOINT_DIR=$1
RUN_NAME=$2

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_dir> [run_name]"
    echo "Example: $0 /root/Desktop/kalashkala/TruthRL/checkpoints/global_step_230/actor my_eval"
    exit 1
fi

# 1. Setup Paths
ACTOR_DIR="${CHECKPOINT_DIR}"
TARGET_DIR="${CHECKPOINT_DIR}_merged"
BASE_MODEL="/root/Desktop/kalashkala/Models/Qwen2.5-VL-3B-Instruct" # Adjust this to your base model path

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="eval_$(basename $CHECKPOINT_DIR)_$(date +%Y%m%d_%H%M%S)"
fi

echo "=================================================="
echo "Step 1: Merging sharded model..."
echo "Source: $ACTOR_DIR"
echo "Target: $TARGET_DIR"
echo "=================================================="

# Run the merge command
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Model merging failed."
    exit 1
fi

echo ""
echo "=================================================="
echo "Step 2: Starting evaluation..."
echo "Model: $TARGET_DIR"
echo "Processor: $BASE_MODEL"
echo "Name: $RUN_NAME"
echo "=================================================="

# Run the evaluation script (reusing your existing run_eval_vsr.sh logic)
./run_eval_vsr.sh -m "$TARGET_DIR" -p "$BASE_MODEL" -n "$RUN_NAME"
