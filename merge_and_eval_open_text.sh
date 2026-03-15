#!/bin/bash

# Script to merge sharded model parameters and run open-text evaluation
# with LLM-as-a-Judge. Works with any VERL-format parquet (VQA, VSR open-text, etc.)
# This script handles the full pipeline: merge → start LLM judge → evaluate → cleanup
#
# Usage: ./merge_and_eval_open_text.sh -c <checkpoint_dir> -d <data_path> [-n <run_name>] [-o <output_dir>] [-j <judge_model>] [-t]
#
# Examples:
#   1. VQA evaluation:
#      ./merge_and_eval_open_text.sh \
#        -c /data/kalashkala/checkpoints/TruthRL_VQA/.../global_step_250/actor \
#        -d /root/Desktop/kalashkala/Datasets/VQAv2/processed_for_verl/validation_vqa.parquet \
#        -n eval_vqa_trained_v1
#
#   2. VSR open-text evaluation:
#      ./merge_and_eval_open_text.sh \
#        -c /data/kalashkala/checkpoints/TruthRL_VSR/.../global_step_230/actor \
#        -d /home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test_open_text.parquet \
#        -n eval_vsr_open_text_v1
#
#   3. With nohup in background:
#      nohup ./merge_and_eval_open_text.sh -c /path/to/actor -d /path/to/data.parquet -n eval_run > eval_run.log 2>&1 &

# ============================================================================
# Default Values
# ============================================================================
CHECKPOINT_DIR=""
RUN_NAME=""
OUTPUT_DIR="results/open_text_eval"
DISABLE_TIMESTAMP=false
DATA_PATH=""
BASE_MODEL="/root/Desktop/kalashkala/Models/Qwen2.5-VL-3B-Instruct"
JUDGE_MODEL="/root/Desktop/kalashkala/Models/Meta-Llama-3.1-8B-Instruct"
JUDGE_PORT=8000
JUDGE_GPU_UTIL=0.85

# ============================================================================
# Usage
# ============================================================================
usage() {
    echo "Usage: $0 -c <checkpoint_dir> -d <data_path> [-n <run_name>] [-o <output_dir>] [-j <judge_model>] [-t]"
    echo "  -c: Path to the checkpoint directory (REQUIRED)"
    echo "  -d: Path to the VERL-format evaluation parquet file (REQUIRED)"
    echo "  -n: Name for this evaluation run (default: eval_open_text_\$(basename CHECKPOINT_DIR)_TIMESTAMP)"
    echo "  -o: Directory to save results (default: results/open_text_eval)"
    echo "  -j: Path to the LLM judge model (default: $JUDGE_MODEL)"
    echo "  -t: Disable timestamp in output sub-directory name"
    exit 1
}

# ============================================================================
# Parse Arguments
# ============================================================================
while getopts "c:n:o:d:j:t" opt; do
    case ${opt} in
        c ) CHECKPOINT_DIR=$OPTARG ;;
        n ) RUN_NAME=$OPTARG ;;
        o ) OUTPUT_DIR=$OPTARG ;;
        d ) DATA_PATH=$OPTARG ;;
        j ) JUDGE_MODEL=$OPTARG ;;
        t ) DISABLE_TIMESTAMP=true ;;
        \? ) usage ;;
    esac
done

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory (-c) is required."
    usage
fi

if [ -z "$DATA_PATH" ]; then
    echo "Error: Data path (-d) is required."
    usage
fi

# ============================================================================
# Setup Paths
# ============================================================================
ACTOR_DIR="${CHECKPOINT_DIR}"
TARGET_DIR="${CHECKPOINT_DIR}_merged"

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="eval_open_text_$(basename $CHECKPOINT_DIR)_$(date +%Y%m%d_%H%M%S)"
fi

# ============================================================================
# Cleanup handler — kills the judge server when script exits
# ============================================================================
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "Cleaning up vLLM judge server (PID $VLLM_PID)..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null
        echo "Judge server terminated."
    fi
}
trap cleanup INT TERM EXIT

# ============================================================================
# Step 1: Merge Sharded Model
# ============================================================================
echo "=================================================="
echo "Step 1: Merging sharded model..."
echo "Source: $ACTOR_DIR"
echo "Target: $TARGET_DIR"
echo "=================================================="

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Model merging failed."
    exit 1
fi

echo ""
echo "Model merged successfully."

# ============================================================================
# Step 2: Start LLM Judge Server
# ============================================================================
echo ""
echo "=================================================="
echo "Step 2: Starting vLLM judge server..."
echo "Judge Model: $JUDGE_MODEL"
echo "Port: $JUDGE_PORT"
echo "=================================================="

# Kill any existing vLLM server on the judge port
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
fuser -k ${JUDGE_PORT}/tcp 2>/dev/null || true
sleep 2

python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --dtype auto \
    --port $JUDGE_PORT \
    --gpu-memory-utilization $JUDGE_GPU_UTIL > vllm_judge_eval.log 2>&1 &

VLLM_PID=$!
echo "vLLM judge server started with PID $VLLM_PID"
echo "Waiting for judge server to load model weights..."

# Wait for the judge server to be ready (poll the health endpoint)
MAX_WAIT=120
WAITED=0
INTERVAL=5
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${JUDGE_PORT}/v1/models > /dev/null 2>&1; then
        echo "Judge server is ready! (waited ${WAITED}s)"
        break
    fi
    # Check if the process is still alive
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Error: vLLM judge server process died unexpectedly."
        echo "Check vllm_judge_eval.log for details."
        exit 1
    fi
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "Error: Judge server did not become ready within ${MAX_WAIT}s."
    echo "Check vllm_judge_eval.log for details."
    exit 1
fi

# ============================================================================
# Step 3: Run Open-Text Evaluation (LLM-as-a-Judge)
# ============================================================================
echo ""
echo "=================================================="
echo "Step 3: Starting open-text evaluation (LLM-as-a-Judge)..."
echo "Model:       $TARGET_DIR"
echo "Processor:   $BASE_MODEL"
echo "Data:        $DATA_PATH"
echo "Judge:       $JUDGE_MODEL"
echo "Name:        $RUN_NAME"
echo "Output Dir:  $OUTPUT_DIR"
echo "=================================================="

EVAL_CMD="python3 evaluation/evaluate_vsr_llm_verifier.py \
    --model_path $TARGET_DIR \
    --processor_path $BASE_MODEL \
    --data_path $DATA_PATH \
    --judge_model $JUDGE_MODEL \
    --judge_api_base http://localhost:${JUDGE_PORT}/v1 \
    --name $RUN_NAME \
    --output_dir $OUTPUT_DIR"

if [ "$DISABLE_TIMESTAMP" = true ]; then
    EVAL_CMD="$EVAL_CMD --no_timestamp"
fi

echo ""
echo "Running: $EVAL_CMD"
echo ""

$EVAL_CMD
EVAL_EXIT=$?

# ============================================================================
# Done
# ============================================================================
echo ""
if [ $EVAL_EXIT -eq 0 ]; then
    echo "=================================================="
    echo "Merge + Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/$RUN_NAME"
    echo "=================================================="
else
    echo "=================================================="
    echo "Evaluation finished with exit code $EVAL_EXIT"
    echo "=================================================="
fi

# The trap handler will clean up the judge server on exit
exit $EVAL_EXIT
