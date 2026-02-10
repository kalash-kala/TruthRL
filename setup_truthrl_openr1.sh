#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="truthrl-openr1"
PYTHON_VERSION="3.11"

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

echo ">>> Installing CUDA toolkit..."
conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y

echo ">>> Installing vllm..."
pip install vllm==0.9.2

echo ">>> Installing setuptools..."
pip install setuptools

echo ">>> Installing flash-attn (no build isolation)..."
pip install flash-attn --no-build-isolation

# REPO_ROOT is the directory where this script is located
REPO_ROOT="$(dirname "${BASH_SOURCE[0]}")"
echo ">>> Changing directory to open-r1 under ${REPO_ROOT}/training..."
cd "${REPO_ROOT}/training/open-r1"

echo ">>> Installing open-r1 in editable mode with dev extras..."
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"

echo
echo ">>> Open-R1 environment setup complete for '${ENV_NAME}'."
