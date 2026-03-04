#!/bin/bash

# Script to merge sharded model parameters and run evaluation
# Usage: ./merge_and_eval.sh -c <checkpoint_dir> [-n <run_name>] [-o <output_dir>] [-t]

# Example with nohup
# nohup ./merge_and_eval.sh -c /root/Desktop/kalashkala/TruthRL/checkpoints/global_step_230/actor -n eval_trained_v1 -o results/vsr_eval -t > eval_trained_v1.log 2>&1 &

CHECKPOINT_DIR=""
RUN_NAME=""
OUTPUT_DIR="results/vsr_eval"
DISABLE_TIMESTAMP=false

usage() {
    echo "Usage: $0 -c <checkpoint_dir> [-n <run_name>] [-o <output_dir>] [-t]"
    echo "  -c: Path to the checkpoint directory (REQUIRED)"
    echo "  -n: Name for this evaluation run (default: eval_$(basename CHECKPOINT_DIR))"
    echo "  -o: Directory to save results (default: results/vsr_eval)"
    echo "  -t: Disable timestamp in output sub-directory name"
    exit 1
}

while getopts "c:n:o:t" opt; do
    case ${opt} in
        c ) CHECKPOINT_DIR=$OPTARG ;;
        n ) RUN_NAME=$OPTARG ;;
        o ) OUTPUT_DIR=$OPTARG ;;
        t ) DISABLE_TIMESTAMP=true ;;
        \? ) usage ;;
    esac
done

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory (-c) is required."
    usage
fi

# 1. Setup Paths
ACTOR_DIR="${CHECKPOINT_DIR}"
TARGET_DIR="${CHECKPOINT_DIR}_merged"
BASE_MODEL="/root/Desktop/kalashkala/Models/Qwen2.5-VL-3B-Instruct" # Adjust this to your base model path

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="eval_$(basename $CHECKPOINT_DIR)_$(date +%Y%m%d_%H%M%S)"
fi

# Note: In run_eval_vsr.sh, timestamp is added by evaluate_vsr.py unless --no_timestamp is passed.
# Here we want to respect the -t flag.

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
echo "Output Dir: $OUTPUT_DIR"
echo "=================================================="

# Run the evaluation script
EVAL_CMD="./run_eval_vsr.sh -m $TARGET_DIR -p $BASE_MODEL -n $RUN_NAME -o $OUTPUT_DIR"
if [ "$DISABLE_TIMESTAMP" = true ]; then
    EVAL_CMD="$EVAL_CMD -t"
fi

$EVAL_CMD
