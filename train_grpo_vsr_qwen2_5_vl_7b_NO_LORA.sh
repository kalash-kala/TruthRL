#!/bin/bash
# VSR + Qwen2.5-VL-7B Example

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'

export CUDA_VISIBLE_DEVICES=0   
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATA_DIR=/data/visual-spatial-reasoning-final/truthrl-sample/parquet
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
REWARD_FN_PATH=/root/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vsr_lexical.py

LR=1e-6
BSZ=16
GROUP_SIZE=2
ROLLOUT_TP_SIZE=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=ability \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    trainer.experiment_name="vsr_qwen2_5_vl_7b_NO_LORA" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=46 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.total_epochs=5 "$@"
