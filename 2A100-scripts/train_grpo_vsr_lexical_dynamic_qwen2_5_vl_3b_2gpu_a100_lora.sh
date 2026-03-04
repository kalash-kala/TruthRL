#!/bin/bash
# ============================================================================
# VSR (LEXICAL DYNAMIC) + Qwen2.5-VL-3B WITH LoRA — 2× A100 Server
# ============================================================================
# Optimized for: 2 GPUs (A100 80GB), ~210GB System RAM
# RAM-per-GPU: ~105GB (vs 31GB on Node 1) — LoRA safe
#
# Run with:
#   nohup bash train_grpo_vsr_lexical_dynamic_qwen2_5_vl_3b_2gpu_a100_lora.sh > train_2gpu_a100_lora_dynamic.log 2>&1 &
# ============================================================================

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

# Network config — adjust if this server has InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.95

# 2× A100 GPUs
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# PATHS — UPDATE THESE FOR THE A100 SERVER
# ============================================================================
DATA_DIR=/root/Desktop/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet
MODEL_PATH=/root/Desktop/kalashkala/Models/Qwen2.5-VL-3B-Instruct
REWARD_FN_PATH=/root/Desktop/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vsr_lexical_dynamic.py

# ============================================================================
# Hyperparameters
# ============================================================================
# Learning Rate: Higher LR is safe with LoRA (only adapter weights updated)
LR=1e-5

# Batch & Rollout Config
# P = BSZ = 8 prompts per step
# G = GROUP_SIZE = 8 rollouts per prompt
# Total sequences per step: P × G = 64
# With 2 GPUs: each GPU handles 64/2 = 32 sequences
#
# Normalization (done by verl internally):
#   normalized_mini_batch = ppo_mini_batch_size × G / world_size
#                         = 8 × 8 / 2 = 32
#   gradient_accumulation_steps = 32 / ppo_micro_batch_size_per_gpu
#                               = 32 / 4 = 8 updates
BSZ=8
GROUP_SIZE=4
ROLLOUT_TP_SIZE=1
EPOCHS=1

# LoRA configuration
LORA_RANK=64
LORA_ALPHA=128

# Avoid CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Launch Training
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train_with_idk.parquet \
    data.val_files=$DATA_DIR/validation_with_idk.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=ability \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.val_before_train=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=adaptive \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="TruthRL_VSR" \
    trainer.experiment_name="vsr_lexical_dynamic_qwen2_5_vl_3b_2gpu_a100_lora_bsz8_lr1e5_gs4_r64" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=$EPOCHS \
    trainer.max_critic_ckpt_to_keep=$EPOCHS \
    trainer.total_epochs=$EPOCHS "$@"
