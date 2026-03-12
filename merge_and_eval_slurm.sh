#!/bin/bash
# ============================================================================
# SLURM Wrapper for Merge and Evaluation — H200 Server (KIAC)
# ============================================================================
#SBATCH --partition=h200
#SBATCH --account=sriramg
#SBATCH --qos=h200_qos
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=merge_eval_vsr
#SBATCH --output=/home/sriramg/kalashabhayk/TruthRL/slurm_logs/logs/%x_%j.out
#SBATCH --error=/home/sriramg/kalashabhayk/TruthRL/slurm_logs/errors/%x_%j.err
#SBATCH --chdir=/home/sriramg/kalashabhayk/TruthRL
#SBATCH --time=04:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Ensure log directories exist
mkdir -p slurm_logs/logs slurm_logs/errors

# Setup Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl

# Environment Variables for Performance/Stability
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,virbr,br-,veth'
export PYTHONUNBUFFERED=1
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

echo "=================================================="
echo "Starting SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Arguments: $@"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Run the merge and eval script
# Pass all arguments through to the original script
chmod +x ./merge_and_eval.sh
./merge_and_eval.sh "$@"

if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Job Completed Successfully"
    echo "=================================================="
else
    echo "=================================================="
    echo "Job Failed with exit code $?"
    echo "=================================================="
    exit 1
fi
