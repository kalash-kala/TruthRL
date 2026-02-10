#!/bin/bash
#################################################################################
# TRUTHRL - GRPO Training on H200 (External Verifier)
# Purpose: Run GRPO training for Llama-8B on an H200 GPU.
# Requires: An external verifier (vLLM) already running on another node.
# Config: Update OPENAI_API_BASE to point to your running verifier.
#################################################################################

#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=TruthRL_H200
#SBATCH --output=slurm_logs/logs/TruthRL_H200_%j.out
#SBATCH --error=slurm_logs/errors/TruthRL_H200_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Ensure we are in the right directory
cd /home/sriramg/kalashabhayk/TruthRL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

set -x

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_METRICS_EXPORT_DISABLE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# These are for the local verifier/judge model (not OpenAI's API)
# The verifier model must be hosted with an OpenAI-compatible API (e.g., vLLM)
export OPENAI_API_BASE=http://192.168.2.6:8000/v1 # A100 (Internal IP CN6)
export OPENAI_API_KEY=token-abc123

# NCCL configuration for single-node training
export NCCL_IB_DISABLE=1
# Using a more generic interface pattern to avoid "source not found" or "interface not found" errors
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN

# CUDA and tokenizer configuration
export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Disable Flash Attention 2.0 to avoid segfault
export DISABLE_FLASH_ATTN=1

export WANDB_PROJECT="TruthRL"
export WANDB_MODE=offline

DATA_DIR=../truthrl_data

N_GPUS=1
ROLLOUT_TP_SIZE=1

MODEL_NAME=/home/sriramg/kalashabhayk/models/Llama-3.1-8B-Instruct
LR=1e-6
KL_LOSS_COEF=0.001
BSZ=32

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name='TruthRL-'$MODEL_NAME'_H200_bsz_'$BSZ'_epochs_100' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.resume_mode=auto \
    trainer.val_before_train=false \
    trainer.total_epochs=10 $@
