#!/bin/bash
#################################################################################
# TRUTHRL - DPO Training
# Purpose: Run Direct Preference Optimization on pre-collected paired data.
# Features: Uses the Open-R1 framework and deepspeed Zero3 for optimization.
#################################################################################
source ~/.bashrc


export WANDB_PROJECT="TruthRL"

conda activate truthrl-openr1

LR=3e-6
N_GPU=8
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE / N_GPU))

iter_idx=0

##### Use this when iter_idx = 0
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
REF_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

##### Use this when iter_idx > 0
# MODEL_NAME="TruthRL/training/checkpoints/dpo/CRAG_Llama-3.1-8B-Instruct_dpo_lr_3e-6_bsz_32_iter_$((iter_idx-1))"
# REF_MODEL_NAME="TruthRL/training/checkpoints/dpo/CRAG_Llama-3.1-8B-Instruct_dpo_lr_3e-6_bsz_32_iter_$((iter_idx-1))"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
TRAIN_DATA_DIR="${SCRIPT_DIR}/data/CRAG/DPO/iter_dpo_CRAG_data/iter_${iter_idx}/best_of_32/train_data_reward_paired.json"
RUN_NAME="CRAG_Llama-3.1-8B-Instruct_dpo_lr_${LR}_bsz_${BATCH_SIZE}_iter_${iter_idx}"
OUTPUT_DIR="${SCRIPT_DIR}/training/checkpoints/dpo/${RUN_NAME}"

echo "Running DPO with learning rate ${LR} for iteration ${iter_idx}"

accelerate launch --config_file training/open-r1/recipes/accelerate_configs/zero3.yaml training/open-r1/src/open_r1/dpo.py \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_NAME} \
    --ref_model ${REF_MODEL_NAME} \
    --learning_rate ${LR} \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --do_train true \
    --do_eval true \
    --eval_strategy steps \
    --eval_steps 6 \
    --choose_type max_min \
    --train_dir ${TRAIN_DATA_DIR} \
    --eval_dir ${TRAIN_DATA_DIR} \
    --loss_type sigmoid \
    --lr_scheduler_type cosine \
    --max_length 16384 \
    --max_prompt_length 14336 \
    --bf16 true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --report_to wandb \
    --label_smoothing 0.1 \
    --use_liger_kernel true