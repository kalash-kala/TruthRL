#!/bin/bash
#################################################################################
# TRUTHRL - Evaluation Runner
# Purpose: Wrapper script to run model evaluation with or without LoRA adapters.
# Usage: ./run_eval.sh -n <name> [-l <adapter_path>] [-b]
#################################################################################


# Default values
# Examples:
#   1. Baseline (Fresh Model) in background:
#      ./run_eval.sh -n baseline_fresh -b
#
#   2. Trained Model (Checkpoint) in background:
#      ./run_eval.sh -n trained_v1 -l checkpoints/TruthRL/TruthRL-meta-llama/Llama-3.1-8B-Instruct_bsz_8_lr_1e-6_kl_loss_coef_0.001/global_step_19/actor/lora_adapter -b
#
#   3. Foreground run:
#      ./run_eval.sh -n test_run

LOG_NAME="eval_$(date +%Y%m%d_%H%M%S)"
LORA_PATH=""
BACKGROUND=false

usage() {
    echo "Usage: $0 [-n log_name] [-l lora_path] [-b]"
    echo "  -n: Name of the log directory (default: eval_timestamp)"
    echo "  -l: Path to the LoRA adapter (if omitted, fresh model is used)"
    echo "  -b: Run in background using nohup"
    exit 1
}

while getopts "n:l:b" opt; do
    case ${opt} in
        n ) LOG_NAME=$OPTARG ;;
        l ) LORA_PATH=$OPTARG ;;
        b ) BACKGROUND=true ;;
        \? ) usage ;;
    esac
done

export TRUTHRL_LOG_NAME="$LOG_NAME"
export LORA_PATH="$LORA_PATH"

# Ensure the conda environment is active if needed
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate truthrl-eval

CMD="python3 evaluation/evaluate.py"
LOG_FILE="evaluation_${LOG_NAME}.log"

if [ "$BACKGROUND" = true ]; then
    echo "Running evaluation in background..."
    echo "Log name: $LOG_NAME"
    echo "LoRA: ${LORA_PATH:-FRESH MODEL}"
    echo "Output redirected to: $LOG_FILE"
    nohup $CMD > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "To monitor progress: tail -f $LOG_FILE"
else
    echo "Running evaluation in foreground..."
    echo "Log name: $LOG_NAME"
    echo "LoRA: ${LORA_PATH:-FRESH MODEL}"
    $CMD
fi
