#!/bin/bash
# VSR + Qwen2.5-VL-3B WITHOUT LoRA - Dynamic Reward
# Using 4 GPUs (4,5,6,7) and optimized parameters for Qwen2.5-VL-3B
# Run with: nohup bash train_grpo_vsr_qwen2_5_vl_3b_4gpu_dynamic.sh > train_qwen2_5_vl_3b_4gpu_dynamic.log 2>&1 &

# Clean up old Ray sessions and temp files before starting
ray stop
rm -rf /tmp/ray/*

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

# Using free GPUs: 4, 5, 6, 7
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Local paths
DATA_DIR=/home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet
MODEL_PATH=/home/debarpanb1/models/Qwen2.5-VL-3B-Instruct
REWARD_FN_PATH=/home/debarpanb1/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vsr_lexical_dynamic.py

# Hyperparameters matched to gemma_4b script
LR=1e-6
BSZ=8
GROUP_SIZE=4
ROLLOUT_TP_SIZE=1

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
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.val_before_train=False \
    algorithm.kl_ctrl.type=adaptive \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name="TruthRL_VSR" \
    trainer.experiment_name="vsr_qwen2_5_vl_3b_4gpu_dynamic_bsz8_lr1e6_groupsize4_epoch1" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.total_epochs=1 "$@"
