# nohup bash train_grpo_4gpu_aggressive.sh > train_grpo_4gpu_aggressive.log 2>&1 &
set -x
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1

export HYDRA_FULL_ERROR=1

export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
# Prevent Ray from clearing CUDA_VISIBLE_DEVICES in worker processes
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_RUNTIME_ENV_VARS='{"CUDA_VISIBLE_DEVICES":"0,1,2,3"}'

# Update OPENAI_API_BASE if your verifier is hosted elsewhere
export OPENAI_API_BASE=http://35.198.251.55:8000/v1 # A100 GCP
export OPENAI_API_KEY=token-abc123

# NCCL configuration
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# CUDA configuration - 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# FLASH ATTENTION ENABLED (Optimized for A100)
# We removed DISABLE_FLASH_ATTN=1 to use the installed flash-attn 2.7.4

export WANDB_PROJECT="TruthRL"
DATA_DIR=$(pwd)/../truthrl_data

# AGGRESSIVE 4-GPU CONFIGURATION - Maximum Utilization
N_GPUS=4
ROLLOUT_TP_SIZE=2

MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
LR=1e-5  # INCREASED from 1e-6
KL_LOSS_COEF=0.001
BSZ=40  # Aggressive batch size - 2.5x from 1 GPU
GROUP_SIZE=8  # Maximum rollouts matching original 8-GPU setup

PYTHON_BIN=/root/anaconda3/envs/truthrl-verl/bin/python3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$PYTHON_BIN - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=14000 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.lora_rank=256 \
    actor_rollout_ref.model.lora_alpha=512 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name='TruthRL-'$MODEL_NAME'_4GPU_Aggressive_LoRA256' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=1000 \
    trainer.resume_mode=auto \
    trainer.total_epochs=1 $@
