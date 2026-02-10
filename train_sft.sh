#!/bin/bash
#################################################################################
# TRUTHRL - SFT Training
# Purpose: Supervised Fine-Tuning template using the Open-R1 framework.
# Config: Points to a specific config_sft.yaml file for training parameters.
#################################################################################
source ~/.bashrc


export WANDB_PROJECT="TruthRL"

conda activate truthrl-openr1

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file training/open-r1/recipes/accelerate_configs/zero3.yaml training/open-r1/src/open_r1/sft.py \
    --config training/open-r1/recipes/Llama3.1-8B-Instruct/sft/config_sft.yaml