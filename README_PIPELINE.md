# TruthRL Pipeline Setup Guide: VQA & Open-Text VSR

This document outlines the end-to-end process for setting up the Visual Question Answering (VQAv2) and Open-Text Visual Spatial Reasoning (VSR) training pipelines on a new server.

## Overview
Because the `TruthRL` repository structure remains consistent across all servers, the main steps when migrating to a new server involve:
1. Downloading and preprocessing the datasets.
2. Updating the absolute paths for **Data** and **Models** in the training & evaluation scripts.

---

## 1. Data Preparation

### A. VQAv2 Pipeline
1. **Download Raw Data:** Download the VQAv2 parquet files (train and validation) from Hugging Face to your server's data directory.
2. **Convert to VERL Format:** Run the conversion script, which extracts the binary image bytes into `.jpg` files and restructures the parquet to match VERL's expected schema (including the strict prompt and reward model info).
   ```bash
   python3 scripts/convert_vqa_for_verl.py \
     --input_parquet /path/to/raw/vqa_train.parquet \
     --output_dir /path/to/processed_vqav2_dir \
     --output_name train_vqa.parquet
   ```
   *(Repeat for the validation split as needed).*

### B. Open-Text VSR Pipeline
1. **Generate Open-Text CSV:** If you haven't already, run `scripts/convert_vsr_to_open_text.py` to use an LLM to generate natural questions out of the raw VSR captions. This outputs a CSV file (e.g., `vsr_open_text_train.csv`).
2. **Convert to VERL Format:** Convert the newly generated CSV into a VERL-compatible parquet.
   ```bash
   python3 scripts/convert_vsr_open_text_for_verl.py \
     --input_csv /path/to/vsr_open_text_train.csv \
     --output_path /path/to/processed_vsr_dir/train_open_text.parquet
   ```

---

## 2. Server Configuration (Path Updates)

Whenever you move code to a new server, you must update the hardcoded paths in the training scripts located in `2A100-scripts/` (e.g., `train_grpo_vqa_qwen2_5_vl_3b_2gpu_a100_lora.sh` and `train_grpo_vsr_open_text_qwen2_5_vl_3b_2gpu_a100_lora.sh`).

Open the relevant script and locate the `PATHS` section. Update the following variables specific to your server's folder locations:

```bash
# ============================================================================
# PATHS
# ============================================================================
# 1. Update this to where you saved the converted parquet datasets
DATA_DIR=/your/server/path/Datasets/VQAv2/processed_for_verl

# 2. Update this to where your Qwen2.5-VL model weights are stored
MODEL_PATH=/your/server/path/Models/Qwen2.5-VL-3B-Instruct

# 3. Update this to where the TruthRL repo is located on this server
REWARD_FN_PATH=/your/server/path/TruthRL/training/verl/verl/utils/reward_score/vqa_reward.py
```

### vLLM Judge Server Path Update
Our scripts automatically start a local `vLLM` server in the background for the LLM-as-a-judge reward function. You **must** update the judge model path in the startup command inside the training script:

```bash
# Update the path to the Meta-Llama-3.1-8B-Instruct model
vllm serve /your/server/path/Models/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    ...
```

---

## 3. Training Execution

Once paths are updated, training can be started directly. The scripts handle starting the judge server, waiting for it to load into VRAM, and spinning up the VERL PPO pipeline.

```bash
cd /your/server/path/TruthRL
nohup bash 2A100-scripts/train_grpo_vqa_qwen2_5_vl_3b_2gpu_a100_lora.sh > train_vqa.log 2>&1 &
```
*Note: The script automatically kills the background vLLM judge server when training finishes or if the script is terminated.*

---

## 4. Evaluation 

To evaluate your trained checkpoints, you will use the script: `evaluation/evaluate_vsr_llm_verifier.py`.

Like training, this requires an LLM judge. Look at the arguments passed to the script to update paths as needed:

1. **Start the Judge Server** in a separate terminal:
   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
     --model /your/server/path/Models/Meta-Llama-3.1-8B-Instruct \
     --dtype auto --port 8000 --gpu-memory-utilization 0.85
   ```

2. **Run the Evaluation Script**:
   ```bash
   python3 evaluation/evaluate_vsr_llm_verifier.py \
     --model_path /your/server/path/Models/Qwen2.5-VL-3B-Instruct \
     --data_path /your/server/path/Datasets/Processed/test_open_text.parquet \
     --judge_model /your/server/path/Models/Meta-Llama-3.1-8B-Instruct \
     --name open_text_evaluation
   ```

The script will produce a `summary_metrics.json` and a detailed `judge_reward_detail.jsonl` breakdown in the `results/` folder.
