#!/bin/bash
# ============================================================================
# VSR (DYNAMIC CONSISTENCY) + Qwen2.5-VL-3B FULL PARAMETER — 2× H200 Server (KIAC)
# ============================================================================
#SBATCH --partition=h200
#SBATCH --account=sriramg
#SBATCH --qos=h200_qos
#SBATCH --gres=gpu:h200:2
#SBATCH --job-name=vsr_consistency_qwen_full_2gpu_h200
#SBATCH --output=/home/sriramg/kalashabhayk/TruthRL/slurm_logs/logs/%x_%j.out
#SBATCH --error=/home/sriramg/kalashabhayk/TruthRL/slurm_logs/errors/%x_%j.err
#SBATCH --chdir=/home/sriramg/kalashabhayk/TruthRL
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "/home/sriramg/kalashabhayk/TruthRL"

# Ensure log directories exist
mkdir -p slurm_logs/logs slurm_logs/errors

source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Clean up old Ray sessions
ray stop
rm -rf /tmp/ray/*

# Kill any existing vLLM server on port 8000 just in case
pkill -f "vllm.entrypoints.openai.api_server" || true
fuser -k 8000/tcp || true

set -x
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_DASHBOARD_ENABLED=0
export RAY_USAGE_STATS_ENABLED=0

# Network config
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'
export PYTHONUNBUFFERED=1

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.95

# 2× H200 GPUs
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# START LOCAL vLLM JUDGE SERVER (BACKGROUND)
# ============================================================================
echo "Starting local vLLM judge server on GPUs 0,1 in the background..."
# Using tensor-parallel-size 2 and low gpu-memory-utilization on H200s
CUDA_VISIBLE_DEVICES=0,1 vllm serve /home/sriramg/kalashabhayk/models/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.1 \
    --enforce-eager \
    --port 8000 > vllm_judge_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID. Waiting 45 seconds to load..."
sleep 45
echo "Proceeding with training..."

# Ensure we cleanup vLLM process on script exit
trap "echo 'Cleaning up vLLM server (PID $VLLM_PID)...'; kill $VLLM_PID; exit" INT TERM EXIT

# ============================================================================
# PATHS (KIAC SPECIFIC)
# ============================================================================
DATA_DIR=/home/sriramg/kalashabhayk/visual-spatial-reasoning/truthrl-sample/parquet
MODEL_PATH=/home/sriramg/kalashabhayk/models/Qwen2.5-VL-3B-Instruct
REWARD_FN_PATH=/home/sriramg/kalashabhayk/TruthRL/training/verl/verl/utils/reward_score/vsr_dynamic_consistency.py

# ============================================================================
# Hyperparameters - Matched to Full Finetuning
# ============================================================================
EPOCHS=2
LR=1e-6
BSZ=8
GROUP_SIZE=4
ROLLOUT_TP_SIZE=1

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
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=adaptive \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="TruthRL_VSR" \
    trainer.experiment_name="vsr_consistency_qwen2_5_vl_3b_2gpu_h200_full_bsz8_lr1e6_gs4" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    trainer.default_local_dir='/home/sriramg/kalashabhayk/TruthRL/checkpoints/${trainer.project_name}/${trainer.experiment_name}' \
    trainer.max_actor_ckpt_to_keep=$EPOCHS \
    trainer.max_critic_ckpt_to_keep=$EPOCHS \
    trainer.total_epochs=$EPOCHS "$@"

echo "Training complete."
