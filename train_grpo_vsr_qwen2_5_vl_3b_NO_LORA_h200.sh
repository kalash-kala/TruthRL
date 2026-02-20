#!/bin/bash
#################################################################################
# TRUTHRL - VSR + Qwen2.5-VL-3B Training on H200 (NO LoRA)
# SLURM settings from train_grpo_h200.sh
# Code from train_grpo_vsr_qwen2_5_vl_3b_NO_LORA.sh
# Command sbatch train_grpo_vsr_qwen2_5_vl_3b_NO_LORA_h200.sh
#################################################################################

#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=TruthRL_VSR_Qwen2_5_VL_3B
#SBATCH --output=slurm_logs/logs/TruthRL_VSR_H200_%j.out
#SBATCH --error=slurm_logs/errors/TruthRL_VSR_H200_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Ensure we are in the right directory
cd /home/sriramg/kalashabhayk/TruthRL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

# NCCL configuration for single-node training (Adapted from train_grpo_h200.sh)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=WARN

# Select GPU(s). For single GPU test, keep as 0. For 4 GPUs, use 0,1,2,3
export CUDA_VISIBLE_DEVICES=0   
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Disable Flash Attention 2.0 to avoid segfault (from train_grpo_h200.sh)
# export DISABLE_FLASH_ATTN=1

# Local paths - Updated based on current environment search
DATA_DIR=/data/visual-spatial-reasoning-final/truthrl-sample/parquet
MODEL_PATH=/home/sriramg/kalashabhayk/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
REWARD_FN_PATH=/home/sriramg/kalashabhayk/TruthRL/training/verl/verl/utils/reward_score/vsr_lexical.py

# Hyperparameters
LR=1e-6
BSZ=16
GROUP_SIZE=2
ROLLOUT_TP_SIZE=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=512 \
    data.max_response_length=32 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=ability \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[Qwen2VLDecoderLayer]' \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    algorithm.kl_ctrl.type=adaptive \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_ctrl.target_kl=0.1 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    'trainer.logger=["console"]' \
    trainer.project_name="TruthRL_VSR" \
    trainer.experiment_name="vsr_qwen2_5_vl_3b_NO_LORA" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=46 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.total_epochs=5 "$@"
