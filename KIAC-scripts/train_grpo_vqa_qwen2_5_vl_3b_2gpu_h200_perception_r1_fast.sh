#!/bin/bash
# ============================================================================
# VQA + Qwen2.5-VL-3B FULL PARAMETER (Perception-R1 Logic) — 2× H200 (KIAC)
# Uses TruthRL verl engine (main_ppo) with Perception-R1 reward function
# Tailored for KIAC H200 infrastructure based on TruthRL environment.
# ============================================================================
#SBATCH --partition=h200
#SBATCH --account=sriramg
#SBATCH --qos=h200_qos
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=vqa_qwen2_5_vl_3b_full_perception_r1_h200
#SBATCH --output=/home/sriramg/kalashabhayk/TruthRL/slurm_logs/logs/%x_%j.out
#SBATCH --error=/home/sriramg/kalashabhayk/TruthRL/slurm_logs/errors/%x_%j.err
#SBATCH --chdir=/home/sriramg/kalashabhayk/TruthRL
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "/home/sriramg/kalashabhayk/TruthRL"

# Ensure log directories exist
mkdir -p slurm_logs/logs slurm_logs/errors

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Clean up old Ray sessions and temp files before starting
ray stop
rm -rf /tmp/ray/*

# Kill any existing vLLM server launched by us just in case
pkill -u $USER -f "vllm.entrypoints.openai.api_server" || true

# Wait for old processes to fully release GPU memory
sleep 10

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

# Network config
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'
export PYTHONUNBUFFERED=1

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.95

# 2× H200 GPUs (KIAC layout)
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# JUDGE MODEL CONFIG (Qwen2.5-32B-Instruct-AWQ)
# ============================================================================
export JUDGE_MODEL='/home/sriramg/kalashabhayk/models/Qwen2.5-32B-Instruct-AWQ'
export JUDGE_PORT=$(( 8000 + RANDOM % 1000 ))
export OPENAI_API_BASE="http://localhost:${JUDGE_PORT}/v1"
export VQA_JUDGE_MODEL="${JUDGE_MODEL}"

# ============================================================================
# START LOCAL vLLM JUDGE SERVER (BACKGROUND)
# ============================================================================
echo "Starting local vLLM judge server on GPUs 0,1 in the background on port $JUDGE_PORT..."
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model "${JUDGE_MODEL}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 2048 \
    --max-num-seqs 128 \
    --enforce-eager \
    --dtype float16 \
    --port "${JUDGE_PORT}" \
    > "vllm_judge_server_perception_r1_${JUDGE_PORT}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID. Waiting for judge server to be ready..."

WAITED=0
INTERVAL=10
while true; do
    if curl -s http://localhost:${JUDGE_PORT}/v1/models > /dev/null 2>&1; then
        echo "Judge server is ready! (waited ${WAITED}s)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Error: vLLM judge server process died. Check vllm_judge_server_perception_r1_${JUDGE_PORT}.log"
        exit 1
    fi
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    echo "Still waiting for judge server... (${WAITED}s elapsed)"
done

# Ensure we cleanup vLLM process on script exit
trap "echo 'Cleaning up vLLM server (PID $VLLM_PID)...'; kill $VLLM_PID; exit" INT TERM EXIT

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR=/home/sriramg/kalashabhayk/visual-question-answering/processed_for_verl
MODEL_PATH=/home/sriramg/kalashabhayk/models/Qwen2.5-VL-3B-Instruct
REWARD_FN_PATH=/home/sriramg/kalashabhayk/TruthRL/training/verl/verl/utils/reward_score/vqa_perception_r1_fast.py

# ============================================================================
# Hyperparameters
# ============================================================================
# Adjusted for 2x H200 to match effective batch size 32
LR=1e-6
BSZ=128
GROUP_SIZE=16
EPOCHS=3

# ============================================================================
# Launch Training using TruthRL main_ppo
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train_perturbed_vqa_updated.parquet \
    data.val_files=$DATA_DIR/val_perturbed_vqa_updated.parquet \
    data.reward_fn_key=ability \
    data.image_key=images \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.val_before_train=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.project_name="Perception-R1" \
    trainer.experiment_name="vqa_qwen2_5_vl_3b_2gpu_h200_perception_r1" \
    trainer.default_local_dir="/storage/users/sriramg/kalashabhayk/checkpoints/Perception-R1/vqa_qwen2_5_vl_3b_2gpu_h200_perception_r1" \
    trainer.logger="['console']" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=50 \
    trainer.total_epochs=$EPOCHS "$@"

echo "Training complete."
