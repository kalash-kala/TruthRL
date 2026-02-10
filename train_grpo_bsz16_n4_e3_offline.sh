# nohup bash train_grpo_bsz16_n4_e3_offline.sh > train_grpo_bsz16_n4_e3_offline.log 2>&1 &
set -x
export WANDB_MODE=disabled

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0
# These are for the local verifier/judge model (not OpenAI's API)
# The verifier model must be hosted with an OpenAI-compatible API (e.g., vLLM)
# Update OPENAI_API_BASE if your verifier is hosted elsewhere
# export OPENAI_API_BASE=http://10.148.0.18:8080/v1 # H100 2 new
export OPENAI_API_BASE=http://10.148.0.19:8000/v1 # A100
export OPENAI_API_KEY=token-abc123

# NCCL configuration for single-node training (disable InfiniBand requirement)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp0s6
export NCCL_DEBUG=WARN

# CUDA and tokenizer configuration
export CUDA_VISIBLE_DEVICES=0   
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Disable Flash Attention 2.0 to avoid segfault (use standard attention instead)
export DISABLE_FLASH_ATTN=1

export WANDB_PROJECT="TruthRL"

DATA_DIR=/home/kalashkala/truthrl_data # refer to HF data repo: weizhepei/TruthRL-CRAG

N_GPUS=1
ROLLOUT_TP_SIZE=1

MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
LR=1e-6
KL_LOSS_COEF=0.001
BSZ=16  # Increased to 16 for better efficiency
GROUP_SIZE=4 # Increased GRPO group size for better reward estimation

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
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name='TruthRL-'$MODEL_NAME'_bsz_'$BSZ'_n_'$GROUP_SIZE'_lr_'$LR'' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.resume_mode=auto \
    trainer.total_epochs=5 $@
