#!/bin/bash
# ============================================================================
# VQAv2 + Qwen2.5-VL-3B WITHOUT LoRA — 58-Cluster (4× GPU)
# ============================================================================
# Adapted from:
#   - train_grpo_vqa_qwen2_5_vl_3b_8gpu_lora.sh  (VQA + LoRA + vLLM judge)
#   - train_grpo_vsr_qwen2_5_vl_3b_8gpu.sh        (no-LoRA conventions)
#
# Uses LLM-as-Judge reward function (vqa_reward.py) with local vLLM judge
#
# Run with:
#   nohup bash train_grpo_vqa_qwen2_5_vl_3b_4gpu.sh > train_vqa_4gpu.log 2>&1 &
# ============================================================================

# Clean up old Ray sessions and temp files before starting
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

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.95

# 8× GPUs (58-Cluster)
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# JUDGE MODEL CONFIG
# ============================================================================
# Symlink to HF cache: ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
JUDGE_MODEL_PATH=/home/debarpanb1/kalashkala/models/Meta-Llama-3.1-8B-Instruct

# ============================================================================
# START LOCAL vLLM JUDGE SERVER (BACKGROUND)
# ============================================================================
echo "Starting local vLLM judge server in the background..."
# Use GPUs 0,1 with tensor-parallel-size 2 and low gpu-memory-utilization
# to leave room for VERL training which uses all 8 GPUs.
CUDA_VISIBLE_DEVICES=4,5 vllm serve "$JUDGE_MODEL_PATH" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.1 \
    --enforce-eager \
    --port 8000 > vllm_judge_server.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID. Waiting 45 seconds for model weights to load into VRAM..."
sleep 45
echo "Proceeding with training..."

# Ensure we cleanup vLLM process on script exit
trap "echo 'Cleaning up vLLM server (PID $VLLM_PID)...'; kill $VLLM_PID; exit" INT TERM EXIT

# ============================================================================
# PATHS (58-Cluster)
# ============================================================================
DATA_DIR=/home/debarpanb1/kalashkala/visual-question-answering/processed_for_verl
MODEL_PATH=/home/debarpanb1/kalashkala/models/Qwen2.5-VL-3B-Instruct
REWARD_FN_PATH=/home/debarpanb1/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vqa_reward.py

# Export judge model path so vqa_reward.py can pick it up
export VQA_JUDGE_MODEL="$JUDGE_MODEL_PATH"

# ============================================================================
# Hyperparameters (full fine-tuning, no LoRA)
# ============================================================================
LR=1e-6
BSZ=8
GROUP_SIZE=4
ROLLOUT_TP_SIZE=1
EPOCHS=3

# ============================================================================
# Launch Training
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train_vqa.parquet \
    data.val_files=$DATA_DIR/validation_vqa.parquet \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.val_before_train=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=adaptive \
    reward_model.enable=False \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="TruthRL_VQA" \
    trainer.experiment_name="vqa_qwen2_5_vl_3b_4gpu_full_ft_bsz8_lr1e6_gs4_epoch3" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    trainer.max_actor_ckpt_to_keep=$EPOCHS \
    trainer.max_critic_ckpt_to_keep=$EPOCHS \
    trainer.total_epochs=$EPOCHS "$@"

echo "Training complete."
