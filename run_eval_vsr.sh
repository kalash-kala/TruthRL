#!/bin/bash
# TRUTHRL - VSR Evaluation Runner (Qwen2.5-VL)
# Purpose: Evaluate performance on the VSR task using local parquet data.
#
# Examples:
#   1. Evaluate Vanilla Qwen2.5-VL-3B (Baseline):
#      ./run_eval_vsr.sh -m /data/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 -n eval_vanilla -b
#
#   2. Evaluate Trained Checkpoint:
#      ./run_eval_vsr.sh -m /root/kalashkala/TruthRL/checkpoints/TruthRL_VSR/vsr_qwen2_5_vl_3b_4gpu_optimized/global_step_210/actor -n eval_trained -b
#      ./run_eval_vsr.sh -m /root/kalashkala/TruthRL/checkpoints/TruthRL_VSR/vsr_qwen2_5_vl_3b_4gpu_optimized/global_step_210/actor_hf -n eval_trained -b
# Usage: ./run_eval_vsr.sh -m <model_path> [-n <run_name>] [-b]

# Default Values
MODEL_PATH=""
RUN_NAME="eval_vsr_$(date +%Y%m%d_%H%M%S)"
BACKGROUND=false
DATA_PATH="/data/visual-spatial-reasoning-final/truthrl-sample/parquet/test.parquet"

usage() {
    echo "Usage: $0 -m <model_path> [-n <run_name>] [-b]"
    echo "  -m: Path to the model checkpoint or huggingface model (REQUIRED)"
    echo "  -n: Name for this evaluation run (default: eval_vsr_TIMESTAMP)"
    echo "  -b: Run in background using nohup"
    exit 1
}

while getopts "m:n:b" opt; do
    case ${opt} in
        m ) MODEL_PATH=$OPTARG ;;
        n ) RUN_NAME=$OPTARG ;;
        b ) BACKGROUND=true ;;
        \? ) usage ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path (-m) is required."
    usage
fi

CMD="python3 evaluation/evaluate_vsr.py --model_path $MODEL_PATH --name $RUN_NAME --data_path $DATA_PATH"
LOG_FILE="${RUN_NAME}.log"

echo "=================================================="
echo "Starting VSR Evaluation: $RUN_NAME"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "=================================================="

if [ "$BACKGROUND" = true ]; then
    echo "Running evaluation in background..."
    echo "Output redirected to: $LOG_FILE"
    nohup $CMD > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "To monitor progress: tail -f $LOG_FILE"
else
    echo "Running evaluation in foreground..."
    $CMD
fi
