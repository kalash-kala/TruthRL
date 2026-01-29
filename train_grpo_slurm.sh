#!/bin/bash
#SBATCH --job-name=TruthRL-Llama-3.3-70B-Instruct
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

source ~/.bashrc
conda activate truthrl-verl

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false

export RAY_DEDUP_LOGS=0
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY="token-abc123"

export WANDB_PROJECT="TruthRL"

# ----- User-configurable paths -------------------------------------------------
DATA_DIR=<path_to_data_dir> # refer to HF data repo: weizhepei/TruthRL-CRAG

MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct

LR=1e-6
KL_LOSS_COEF=0.001
BSZ=64

verl_workdir="$(dirname "${BASH_SOURCE[0]}")/training/verl"
train_files=$DATA_DIR/train.parquet
val_files=$DATA_DIR/test.parquet

# No container image needed when using a local Conda environment
# ------------------------------------------------------------------------------

# ----- Hyperparameters & experiment naming -------------------------------------

EXPERIMENT_NAME='TruthRL-'$MODEL_NAME'_bsz_'$BSZ'_lr_'$LR'_kl_loss_coef_'$KL_LOSS_COEF'_trinary'

# ------------------------------------------------------------------------------

# If running under Slurm, update the job name dynamically to match EXPERIMENT_NAME
if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "Setting Slurm job name to: $EXPERIMENT_NAME"
  scontrol update JobId=$SLURM_JOB_ID Name=$EXPERIMENT_NAME || true
fi


# -------------------------- Cluster-specific setup -----------------------------
# List hostnames allocated to this job
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
# Grab the IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Convert to IPv4 if the cluster returns IPv6 + IPv4 together
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPv6 address detected. Using IPv4 $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head

echo "IP Head: $ip_head"

# Print env for debugging
printenv

# ------------------------------- Start Ray ------------------------------------

echo "Starting RAY HEAD on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# Give the head a moment to start before launching workers
sleep 10

# Number of worker nodes (all nodes except head)
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting RAY WORKER $i on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

# ----------------------------- Launch training (with retry) --------------------

export VERL_AUTO_PADDING=1

run_training() {
  PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --chdir=$verl_workdir \
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=$train_files \
      data.val_files=$val_files \
      data.train_batch_size=$BSZ \
      data.max_prompt_length=16384 \
      data.max_response_length=2048 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      actor_rollout_ref.model.path=$MODEL_NAME \
      actor_rollout_ref.actor.optim.lr=$LR \
      actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=16 \
      actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
      actor_rollout_ref.rollout.n=8 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.use_kl_in_reward=False \
      trainer.critic_warmup=0 \
      trainer.logger='["console","wandb"]' \
      trainer.project_name=$WANDB_PROJECT \
      trainer.experiment_name=$EXPERIMENT_NAME \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=8 \
      trainer.save_freq=20 \
      trainer.test_freq=5 \
      trainer.resume_mode=auto \
      trainer.total_epochs=100 "$@"
}

MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_DELAY_SEC=${RETRY_DELAY_SEC:-60}

attempt=1
training_succeeded=0
while true; do
  echo "Starting training attempt $attempt/$MAX_RETRIES"
  set +e
  run_training "$@"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "Training attempt $attempt succeeded"
    training_succeeded=1
    break
  fi
  echo "Training attempt $attempt failed with exit code $status"
  if [[ $attempt -ge $MAX_RETRIES ]]; then
    echo "Exceeded maximum retries ($MAX_RETRIES). Cleaning up Ray and proceeding to evaluation."
    # Clean up Ray before exiting to avoid leaving blocked steps
    cleanup_ray() {
      echo "Stopping Ray on allocated nodes..."
      srun --overlap --nodes=1 --ntasks=1 -w "$head_node" ray stop || true
      for ((i = 1; i <= worker_num; i++)); do
          node_i=${nodes_array[$i]}
          srun --overlap --nodes=1 --ntasks=1 -w "$node_i" ray stop || true
      done
    }
    cleanup_ray
    break
  fi
  echo "Sleeping ${RETRY_DELAY_SEC}s before retry..."
  sleep "$RETRY_DELAY_SEC"
  attempt=$((attempt + 1))
done
