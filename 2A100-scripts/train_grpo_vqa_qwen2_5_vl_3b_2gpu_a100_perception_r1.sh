#!/bin/bash
# ============================================================================
# VQA + Qwen2.5-VL-3B FULL PARAMETER (Perception-R1 Logic) — 2× A100 (TruthRL)
# Mimics H200 script but optimized for 2x A100 (80GB) memory constraints.
# ============================================================================

# nohup bash train_grpo_vqa_qwen2_5_vl_3b_2gpu_a100_perception_r1.sh > train_vqa_2gpu_a100_epoch3_ft_bsz16_lr1e6_gs4_perception_r1.log 2>&1 &

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "/home/kalashkala/TruthRL"

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

# 2× A100 GPUs
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# JUDGE MODEL CONFIG (Qwen2.5-32B-Instruct-AWQ)
# ============================================================================
export JUDGE_MODEL='/home/kalashkala/Models/Qwen2.5-32B-Instruct-AWQ'
export JUDGE_PORT=8000
export OPENAI_API_BASE="http://localhost:${JUDGE_PORT}/v1"
export VQA_JUDGE_MODEL="${JUDGE_MODEL}"

# ============================================================================
# START LOCAL vLLM JUDGE SERVER (BACKGROUND)
# ============================================================================
echo "Starting local vLLM judge server on GPUs 0,1 in the background on port $JUDGE_PORT..."
# Using tensor-parallel-size 2 and low gpu-memory-utilization to leave room for training
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
    --model "${JUDGE_MODEL}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.30 \
    --max-model-len 2048 \
    --max-num-seqs 128 \
    --enforce-eager \
    --dtype float16 \
    --port 8000 \
    > "vllm_judge_server_perception_r1_a100_${JUDGE_PORT}.log" 2>&1 &

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
        echo "Error: vLLM judge server process died. Check vllm_judge_server_perception_r1_a100_${JUDGE_PORT}.log"
        exit 1
    fi
    sleep $INTERVAL
    WAITED=$((WAITED + INTERVAL))
    echo "Still waiting for judge server... (${WAITED}s elapsed)"
done

# Ensure we cleanup vLLM process on script exit
trap 'echo "Cleaning up vLLM server (PID $VLLM_PID)..."; kill $VLLM_PID; exit' INT TERM EXIT

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR=/home/kalashkala/Datasets/VQAv2/processed_for_verl
MODEL_PATH=/home/kalashkala/Models/Qwen2.5-VL-3B-Instruct
# REWARD_FN_PATH=/home/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vqa_perception_r1.py
REWARD_FN_PATH=/home/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vqa_perception_r1_fast.py


# ============================================================================
# Hyperparameters
# ============================================================================
LR=1e-6
BSZ=16
GROUP_SIZE=4
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.val_before_train=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.project_name="Perception-R1" \
    trainer.experiment_name="vqa_qwen2_5_vl_3b_2gpu_a100_perception_r1" \
    trainer.logger="['console']" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=210 \
    trainer.test_freq=50 \
    trainer.total_epochs=$EPOCHS "$@"

echo "Training complete."
