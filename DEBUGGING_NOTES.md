# TruthRL Training Debugging Notes

## Overview
This document contains debugging notes and solutions for common issues encountered when running the TruthRL GRPO training script.

---

## Issue 1: KeyError: 'embed_tokens.weight'

### Problem
When running `train_grpo.sh`, the script failed with:
```
KeyError: 'embed_tokens.weight'
```

This error occurred during vLLM weight loading when synchronizing FSDP actor model parameters to the vLLM inference engine.

### Root Cause
There was a mismatch in parameter key naming between the FSDP-wrapped actor model and the vLLM inference engine:

1. **FSDP provides**: `model.embed_tokens.weight`
2. **vLLM expects**: `model.embed_tokens.base_layer.weight`

The vLLM model in this configuration (with LoRA) wraps `embed_tokens` in a base layer, requiring the `.base_layer.` suffix.

### Solution
Modified `/home/kalashkala/TruthRL/training/verl/verl/workers/sharding_manager/fsdp_vllm.py` in the `replace_lora_wrapper` function to explicitly map the key:

```python
# Explicitly handle embed_tokens if it requires base_layer
if k == "model.embed_tokens.weight":
    return "model.embed_tokens.base_layer.weight"
```

**File**: [`fsdp_vllm.py:311-313`](file:///home/kalashkala/TruthRL/training/verl/verl/workers/sharding_manager/fsdp_vllm.py#L311-L313)

### Verification
- The training script successfully loads weights without the KeyError
- The process continues past the initialization phase

---

## Issue 2: Verifier Model 404 Error

### Problem
After fixing the KeyError, training continued but failed with:
```
ERROR: The model `meta-llama/Llama-3.3-70B-Instruct` does not exist.
```

The training code was requesting `meta-llama/Llama-3.3-70B-Instruct` from the verifier API, but the deployed vLLM server had `google/gemma-3-27b-it` loaded.

### Root Cause
The verifier model name was hardcoded in the reward scoring function and didn't match the deployed model.

### Solution
Modified `/home/kalashkala/TruthRL/training/verl/verl/utils/reward_score/truthrl_qa.py` to use the correct model:

```python
response = client.chat.completions.create(
    model="google/gemma-3-27b-it",  # Changed from meta-llama/Llama-3.3-70B-Instruct
    messages=messages,
    temperature=0,
    top_p=0.9,
    max_tokens=512,
)
```

**File**: [`truthrl_qa.py:412-418`](file:///home/kalashkala/TruthRL/training/verl/verl/utils/reward_score/truthrl_qa.py#L412-L418)

---

## Running the Training

### Prerequisites
1. **Conda Environment**: `truthrl-verl` (contains all dependencies)
2. **Verifier Model Server**: vLLM server running at `http://10.128.0.30:8000/v1` with `google/gemma-3-27b-it` loaded
3. **GPU**: NVIDIA H100 80GB (or similar) with sufficient memory

### Commands

#### Run Training
```bash
conda run -n truthrl-verl env PYTHONPATH=$PYTHONPATH:$(pwd)/training/verl bash ./train_grpo.sh
```

Or from within the conda environment:
```bash
conda activate truthrl-verl
export PYTHONPATH=$PYTHONPATH:$(pwd)/training/verl
bash ./train_grpo.sh
```

#### Check GPU Usage
```bash
nvidia-smi
```

#### Kill Stuck GPU Processes
```bash
# Find the PID from nvidia-smi output
kill <PID>

# Force kill if needed
kill -9 <PID>
```

---

## Configuration Details

### Environment Variables (set in `train_grpo.sh`)
- `OPENAI_API_BASE`: `http://10.128.0.30:8000/v1` (verifier model endpoint)
- `OPENAI_API_KEY`: `token-abc123`
- `CUDA_VISIBLE_DEVICES`: `0`
- `DISABLE_FLASH_ATTN`: `1`

### Model Configuration
- **Actor Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Verifier Model**: `google/gemma-3-27b-it` (deployed separately on vLLM server)
- **LoRA Config**: rank=16, alpha=32, target_modules=all-linear

### Training Parameters
- Batch size: 64
- Learning rate: 1e-6
- KL loss coefficient: 0.001
- Max prompt length: 8192
- Max response length: 2048

---

## Common Issues

### OutOfMemoryError
**Symptom**: `torch.OutOfMemoryError: CUDA out of memory`

**Cause**: Previous training process still holding GPU memory

**Solution**:
1. Check GPU processes: `nvidia-smi`
2. Kill the process: `kill <PID>`
3. Re-run training

### Module Not Found
**Symptom**: `ModuleNotFoundError: No module named 'verl'` or `'numpy'`

**Cause**: Running script without proper conda environment or PYTHONPATH

**Solution**: Use the full conda run command with PYTHONPATH set

---

## Files Modified

1. **`/home/kalashkala/TruthRL/training/verl/verl/workers/sharding_manager/fsdp_vllm.py`**
   - Added `embed_tokens` to `base_layer` mapping

2. **`/home/kalashkala/TruthRL/training/verl/verl/utils/reward_score/truthrl_qa.py`**
   - Changed verifier model name to `google/gemma-3-27b-it`

3. **`/home/kalashkala/TruthRL/train_grpo.sh`**
   - Updated `OPENAI_API_BASE` to point to verifier server IP
   - Added `HYDRA_FULL_ERROR=1` for better error messages

---

## Debugging Tips

### Enable Full Error Traces
```bash
export HYDRA_FULL_ERROR=1
```

### Check Ray Logs
```bash
# Find the latest Ray session
ls -lt /tmp/ray/session_latest/logs/

# Search for specific errors
grep -r "ERROR" /tmp/ray/session_latest/logs/
grep -r "KeyError" /tmp/ray/session_latest/logs/
```

### Add Debug Prints
When debugging weight loading issues, you can add print statements in `fsdp_vllm.py`:
```python
import sys
print(f"DEBUG: Updated params keys: {list(updated_params.keys())[:20]}", file=sys.stdout)
sys.stdout.flush()
```

---

## Additional Resources

- **vLLM Documentation**: https://docs.vllm.ai/
- **VERL Repository**: https://github.com/volcengine/verl
- **Training Logs**: `./wandb/` directory
- **WandB Project**: TruthRL

---

**Last Updated**: 2026-01-18
