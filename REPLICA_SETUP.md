# TruthRL Replicated Setup Guide

This guide ensures a fast, error-free setup of the TruthRL environment on a new server with a separate large data partition (e.g., `/data`).

---

## 1. System & Disk Initialization

If you have a large secondary disk (1.9TB) mounted at `/data`, use it for models and datasets to prevent the root partition from filling up.

### Set up Directory Structure
```bash
# Create folders on the large disk
mkdir -p /data/truthrl_data
mkdir -p /data/huggingface_cache
mkdir -p /data/visual-spatial-reasoning-final

# Create symbolic links in your home folder for convenience
ln -s /data/truthrl_data ~/kalashkala/truthrl_data
ln -s /data/visual-spatial-reasoning-final ~/kalashkala/visual-spatial-reasoning-final
```

### Configure Hugging Face Cache
Add this to your `~/.bashrc` to ensure all models are downloaded to the large disk:
```bash
echo 'export HF_HOME=/data/huggingface_cache' >> ~/.bashrc
source ~/.bashrc
```

---

## 2. Environment Setup

The project uses a specific `verl` fork and specific package versions.

```bash
# Clone the repository (if not already done)
git clone <your-repo-url> ~/kalashkala/TruthRL
cd ~/kalashkala/TruthRL

# Run the automated setup script
# This creates a conda env 'truthrl-verl' and installs vLLM/Flash-Attn
bash setup_truthrl_verl.sh

# Activate the environment
conda activate truthrl-verl
```

---

## 3. Dataset Acquisition (VSR Example)

To replicate the Visual Spatial Reasoning setup:

### A. Download raw metadata/blobs
```bash
# Use huggingface-cli for efficient download
huggingface-cli download juletxara/visual-spatial-reasoning --local-dir ~/kalashkala/vsr-dataset
```

### B. Extract Images and JSONL
Run the extraction script which handles checksums and places data in the `/data` area:
```bash
# Modify TruthRL/scripts/extract_vsr_dataset.py if paths differ (it currently uses absolute paths)
python3 TruthRL/scripts/extract_vsr_dataset.py
```

### C. Preprocess for Training (Parquet)
Convert the raw JSONL to Verl-compatible parquets:
```bash
python3 TruthRL/scripts/preprocess_vsr.py \
    --train_path ~/kalashkala/visual-spatial-reasoning-final/random/train.jsonl \
    --test_path ~/kalashkala/visual-spatial-reasoning-final/random/test.jsonl \
    --image_dir ~/kalashkala/visual-spatial-reasoning-final/images \
    --output_dir /data/visual-spatial-reasoning-final/truthrl-sample/parquet
```

---

## 4. Model Deployment (Verifier/Judge)

For GRPO training, you often need an external verifier endpoint (e.g., Gemma-3-27B).

```bash
# On the GPU node intended for the verifier:
bash deploy_verifier_a100.sh
```
*Note: Note the `INTERNAL_IP` printed by this script. You will need it for the training configuration.*

---

## 5. Running Training

Ensure your training script points to the correct `DATA_DIR` and `OPENAI_API_BASE` (the verifier IP).

```bash
# Example for VSR training
bash train_grpo_vsr_gemma_4b_NO_LORA.sh
```

---

## Troubleshooting Checklist
1. **Flash Attention**: If you get segfaults on H100/A100, try `export DISABLE_FLASH_ATTN=1`.
2. **Ray Logs**: Check `/tmp/ray/session_latest/logs/` for distributed training errors.
3. **Internal IP**: Verify that `OPENAI_API_BASE` in your training script matches the current IP of your verifier node.
4. **Permissions**: Ensure your user has write access to `/data`.
