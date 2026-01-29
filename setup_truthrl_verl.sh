#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="truthrl-verl"
PYTHON_VERSION="3.10"

echo ">>> Initializing conda..."
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda command not found. Please ensure Conda is installed and on your PATH."
  exit 1
fi

echo ">>> Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION} (if it does not already exist)..."
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "Environment '${ENV_NAME}' already exists. Skipping creation."
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo ">>> Activating Conda environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo ">>> Installing CUDA toolkit (nvidia/label/cuda-12.4.0::cuda-toolkit)..."
conda install -n "${ENV_NAME}" -c nvidia/label/cuda-12.4.0 cuda-toolkit -y

# REPO_ROOT is the directory where this script is located
REPO_ROOT="$(dirname "${BASH_SOURCE[0]}")"
echo ">>> Changing directory to training/verl under ${REPO_ROOT}..."
cd "${REPO_ROOT}/training/verl"

echo ">>> Running vLLM / sglang / mcore install script with USE_MEGATRON=0..."
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

echo ">>> Installing Python packages via pip..."
pip install \
  numpy==1.26.1 \
  opentelemetry-sdk==1.26.0 \
  click==8.2.1 \
  tensordict==0.8.1

echo ">>> Installing current project in editable mode (no deps)..."
pip install --no-deps -e .

echo
echo ">>> Setup steps complete."
echo "You likely still need to authenticate with Hugging Face and Weights & Biases."
echo "Run the following commands (they are interactive):"
echo "  huggingface-cli login"
echo "  wandb login"
echo
echo "Make sure to run them from an activated '${ENV_NAME}' environment."

