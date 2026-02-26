#!/bin/bash
# TRUTHRL - VSR Evaluation Runner (Qwen2.5-VL)
# Purpose: Evaluate performance on the VSR task using local parquet data.


# Command to merge sharded model parameters
# python3 -m verl.model_merger merge --backend fsdp --local_dir /home/debarpanb1/kalashkala/TruthRL/58-Cluster-scripts/checkpoints/TruthRL_VSR/vsr_qwen2_5_vl_3b_8gpu_optimized/global_step_230/actor --target_dir /home/debarpanb1/kalashkala/TruthRL/58-Cluster-scripts/checkpoints/TruthRL_VSR/vsr_qwen2_5_vl_3b_8gpu_optimized/global_step_230/actor_merged


# Examples:
#   1. Evaluate Vanilla Qwen2.5-VL-3B (Baseline):
#      ./run_eval_vsr.sh -m /home/debarpanb1/models/Qwen2.5-VL-3B-Instruct -n eval_vanilla -b
#
#   2. Evaluate Trained VeRL Checkpoint (processor loaded from base model):
#      ./run_eval_vsr.sh -m /home/debarpanb1/kalashkala/TruthRL/58-Cluster-scripts/checkpoints/TruthRL_VSR/vsr_qwen2_5_vl_3b_8gpu_optimized/global_step_230/actor -p /home/debarpanb1/models/Qwen2.5-VL-3B-Instruct -n eval_trained -b
#      ./run_eval_vsr.sh -m /home/debarpanb1/kalashkala/TruthRL/58-Cluster-scripts/checkpoints/TruthRL_VSR/vsr_qwen2_5_vl_3b_8gpu_optimized/global_step_230/actor_hf -p /home/debarpanb1/models/Qwen2.5-VL-3B-Instruct -n eval_trained -b

#   3. Sharded model evaluation
#      ./run_eval_vsr.sh -m /home/debarpanb1/kalashkala/TruthRL/58-Cluster-scripts/checkpoints/TruthRL_VSR/vsr_qwen2_5_vl_3b_8gpu_optimized/global_step_230/actor_merged -n eval_trained_v1 -b

# Usage: ./run_eval_vsr.sh -m <model_path> [-n <run_name>] [-b]

# Default Values
MODEL_PATH="/home/debarpanb1/models/Qwen2.5-VL-3B-Instruct"
PROCESSOR_PATH=""  # If empty, defaults to MODEL_PATH inside evaluate_vsr.py
RUN_NAME="eval_vsr_$(date +%Y%m%d_%H%M%S)"
BACKGROUND=false
DATA_PATH="/home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test_with_idk.parquet"

usage() {
    echo "Usage: $0 -m <model_path> [-p <processor_path>] [-n <run_name>] [-b]"
    echo "  -m: Path to the model checkpoint or huggingface model (REQUIRED)"
    echo "  -p: Path to processor/tokenizer (optional; use base model path for VeRL checkpoints)"
    echo "  -n: Name for this evaluation run (default: eval_vsr_TIMESTAMP)"
    echo "  -b: Run in background using nohup"
    exit 1
}

while getopts "m:p:n:b" opt; do
    case ${opt} in
        m ) MODEL_PATH=$OPTARG ;;
        p ) PROCESSOR_PATH=$OPTARG ;;
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
if [ -n "$PROCESSOR_PATH" ]; then
    CMD="$CMD --processor_path $PROCESSOR_PATH"
fi
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
