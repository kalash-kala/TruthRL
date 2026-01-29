#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="truthrl-eval"
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

REPO_ROOT="/home/kalashkala/TruthRL"
echo ">>> Changing directory to evaluation under ${REPO_ROOT}..."
cd "${REPO_ROOT}/evaluation"

echo ">>> Installing evaluation requirements from requirements.txt..."
pip install -r requirements.txt

echo
echo ">>> Evaluation environment setup complete for '${ENV_NAME}'."

