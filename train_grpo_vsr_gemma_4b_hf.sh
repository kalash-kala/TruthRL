# VSR + Gemma 3 4B using HF rollout (bypasses vLLM 256/512 placeholder mismatch)
# Use this when vLLM rollout fails with "Attempted to assign 256 = 256 multimodal tokens to 512 placeholders"
#
# To revert to vLLM: use train_grpo_vsr_gemma_4b.sh instead, or change rollout.name=vllm

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

LR=1e-5
BSZ=16
GROUP_SIZE=4

/home/kalashkala/miniconda3/envs/truthrl-verl/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key=ability \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    +actor_rollout_ref.model.exclude_modules='.*vision_tower.*' \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['Gemma3DecoderLayer','SiglipEncoderLayer'] \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    +actor_rollout_ref.rollout.micro_batch_size=2 \
    actor_rollout_ref.rollout.hf_rollout_omit_position_ids=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="TruthRL_VSR" \
    trainer.experiment_name="vsr_gemma_3_4b_grpo_hf" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.total_epochs=5 "$@"
