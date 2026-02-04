#!/bin/bash
#################################################################################
# TRUTHRL - Joint Local Training (H200)
# Purpose: Single-script solution that launches a local vLLM Verifier (Gemma-27B) 
#          and the GRPO Trainer (Llama-8B) on the same node.
# Resource Management: Automatically calculates memory splits to fit both on 1 GPU.
#################################################################################

#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=TruthRL_H200_Local
#SBATCH --output=slurm_logs/logs/TruthRL_H200_Local_%j.out
#SBATCH --error=slurm_logs/errors/TruthRL_H200_Local_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Ensure we are in the right directory
cd /home/sriramg/kalashabhayk/TruthRL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

set -x

# =================================================================
# 1. LAUNCH LOCAL VERIFIER (Gemma-27B)
# =================================================================

# Unique port based on Job ID to avoid collisions
VERIFIER_PORT=$((10000 + (${SLURM_JOBID:-$$} % 10000)))
VERIFIER_MODEL=/home/sriramg/kalashabhayk/models/gemma-3-27b-it
VERIFIER_LOG=slurm_logs/logs/verifier_local_${SLURM_JOBID}.log

echo "=================================================="
echo "LAUNCHING LOCAL VERIFIER (Gemma-27B)"
echo "Port: $VERIFIER_PORT"
echo "Log: $VERIFIER_LOG"
echo "=================================================="

# Ensure ROCR/CUDA vars don't conflict
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

# Start vLLM in background
# MEMORY CONFIG:
# H200 = 141GB VRAM
# Gemma-27B needs ~54GB (Weights)
# We limit it to 45% (approx 63.5GB). Sufficient for weights + cache.
# We ensure Flash Attention is ENABLED for the verifier (unset DISABLE_FLASH_ATTN).
# Note: We do NOT use a subshell ( ... ) so we can capture the PID correctly.

# Temporarily unset for verifier
unset DISABLE_FLASH_ATTN

mkdir -p slurm_logs/logs

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $VERIFIER_MODEL \
    --served-model-name google/gemma-3-27b-it \
    --tensor-parallel-size 1 \
    --port $VERIFIER_PORT \
    --api-key token-abc123 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 8192 \
    --disable-log-requests > $VERIFIER_LOG 2>&1 &

VERIFIER_PID=$!
echo "Verifier PID: $VERIFIER_PID"

# Wait for Verifier to start
echo "Waiting for Verifier startup (timeout: 45 mins)..."
MAX_RETRIES=540
COUNT=0
URL="http://localhost:$VERIFIER_PORT/v1/models"
READY=0
while [ $COUNT -lt $MAX_RETRIES ]; do
    sleep 5
    # Check if process is alive
    if ! kill -0 $VERIFIER_PID 2>/dev/null; then
        echo "Verifier process died!"
        tail -n 50 $VERIFIER_LOG
        exit 1
    fi
    # Check if port is listening/responding
    if curl -s $URL > /dev/null; then
        echo "Verifier is responding to requests!"
        READY=1
        break
    else
        echo "Waiting for verifier... ($COUNT/$MAX_RETRIES)"
    fi
    COUNT=$((COUNT+1))
done

if [ $READY -eq 0 ]; then
    echo "Timed out waiting for verifier."
    kill $VERIFIER_PID
    exit 1
fi

# Cleanup trap to kill verifier when script exits
trap "kill $VERIFIER_PID" EXIT

# =================================================================
# 2. START TRAINING (Llama-8B)
# =================================================================

echo "=================================================="
echo "STARTING TRAINING (Llama-8B)"
echo "=================================================="

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_METRICS_EXPORT_DISABLE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Point to our local verifier
export OPENAI_API_BASE=http://localhost:$VERIFIER_PORT/v1
export OPENAI_API_KEY=token-abc123

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_PROJECT="TruthRL"
export WANDB_MODE=offline

# Explicitly disable Flash Attention for Trainer if needed (based on previous issues)
export DISABLE_FLASH_ATTN=1

DATA_DIR=../truthrl_data
N_GPUS=1
ROLLOUT_TP_SIZE=1
MODEL_NAME=/home/sriramg/kalashabhayk/models/Llama-3.1-8B-Instruct
LR=1e-6
KL_LOSS_COEF=0.001
BSZ=32

# Training Command
# MEMORY CONFIG:
# We limit the Trainer's vLLM Rollout engine to 0.45 (63.5GB).
# Total GPU usage: 0.45 (Verifier) + 0.45 (Trainer) = 0.9. Safe.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name='TruthRL-'$MODEL_NAME'_H200_LocalVerifier_bsz_'$BSZ \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.resume_mode=auto \
    trainer.val_before_train=false \
    trainer.total_epochs=10 $@
