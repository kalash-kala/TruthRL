#!/bin/bash
# Test vLLM inference WITHOUT LoRA to verify multimodal works

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp0s6
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES=0   
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATA_DIR=/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet
MODEL_NAME=google/gemma-3-4b-it
REWARD_FN_PATH=/home/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vsr_lexical.py

# Run WITHOUT LoRA - just validation
/home/kalashkala/miniconda3/envs/truthrl-verl/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=16 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=ability \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    'trainer.logger=["console"]' \
    trainer.project_name="TruthRL_VSR" \
    trainer.experiment_name="test_no_lora" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=True \
    trainer.val_only=True "$@"
