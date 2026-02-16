# Gemma 3 Multimodal Training Investigation Summary

## Problem
Model outputs gibberish during training with vLLM, suggesting it cannot "see" the images.

## Root Cause Discovery Process

### Step 1: Dataset Format Issue
- **Initial hypothesis**: Dataset had wrong image token
- **Finding**: Dataset was using Gemma 3's special character `🖼️` instead of generic `<image>` tag
- **Fix**: Created `fix_parquet_use_image_tag.py` to convert dataset
- **Result**: Dataset now has `<image>` tags, but model still blind

### Step 2: Token Collapsing Issue  
- **Hypothesis**: vLLM wasn't receiving the BOI placeholder token
- **Finding**: `_collapse_multimodal_tokens` was skipping soft tokens but not ensuring BOI presence
- **Fix**: Updated function to always insert BOI token (255999) when collapsing
- **Result**: Verification script confirms BOI token present, but model still blind during training

### Step 3: verl Processing Pipeline
- **Hypothesis**: verl's `_build_messages` wasn't processing `<image>` tags correctly
- **Finding**: verl specifically looks for literal `<image>` tags to create structured format
- **Verification**: Confirmed dataset has `<image>` tags and verl processes them correctly
- **Debug output**: `multi_modal_data` IS being passed to vLLM with image data
- **Result**: Pipeline is correct, but model still blind

### Step 4: LoRA + Multimodal Incompatibility (CURRENT HYPOTHESIS)
- **User insight**: Training uses LoRA, but verification script doesn't
- **Finding**: vLLM has warning: *"Regarding multimodal models, vLLM currently only supports adding LoRA to language model"*
- **Code analysis**: vLLM wraps entire model with LoRA manager even for multimodal models
- **Hypothesis**: LoRA wrapping breaks multimodal processing pipeline in vLLM
- **Test in progress**: Running training WITHOUT LoRA to confirm

## Technical Details

### Data Flow (Verified Working)
1. Dataset: `<image>` tag in prompt text ✓
2. verl's `_build_messages()`: Converts to `[{"type": "image"}, {"type": "text", ...}]` ✓
3. `processor.apply_chat_template()`: Inserts Gemma 3 image token (`🖼️` / ID 255999) ✓
4. `processor()`: Expands to 256 soft tokens (262144) + EOI (256000) ✓
5. `_collapse_multimodal_tokens()`: Collapses back to single BOI (255999) ✓
6. vLLM receives: `prompt_token_ids` with BOI + `multi_modal_data` with image ✓

### vLLM LoRA Implementation
```python
# From vllm/worker/model_runner.py
if self.lora_config:
    if supports_multimodal(self.model):
        logger.warning(
            "Regarding multimodal models, vLLM currently "
            "only supports adding LoRA to language model.")
    
    # Uses text config for multimodal models
    text_config = self.model_config.hf_config.get_text_config()
    
    self.lora_manager = LRUCacheWorkerLoRAManager(...)
    self.model = self.lora_manager.create_lora_manager(self.model)  # Wraps model
```

## Files Modified
1. `/home/kalashkala/TruthRL/training/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
   - Updated `_collapse_multimodal_tokens()` function
   - Added debug logging

2. `/home/kalashkala/TruthRL/vllm_dry_run.py`
   - Updated `_collapse_multimodal_tokens()` to match training version

3. `/home/kalashkala/scripts/fix_parquet_use_image_tag.py`
   - Script to convert dataset from `🖼️` to `<image>` tags

4. `/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet`
   - Updated with `<image>` tags

5. `/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test.parquet`
   - Updated with `<image>` tags

## Test Scripts Created
1. `verify_full_flow.py` - Comprehensive flow validation (PASSES)
2. `test_no_lora_validation.sh` - Training without LoRA (IN PROGRESS)

## Next Steps
1. **Wait for no-LoRA test results**
   - If model can "see" without LoRA → Confirms LoRA is the issue
   - If model still blind → investigate vLLM multimodal implementation

2. **If LoRA is confirmed as the issue:**
   - Option A: Train without LoRA (full fine-tuning)
   - Option B: Report bug to vLLM team
   - Option C: Find workaround (e.g., different LoRA config)

3. **If LoRA is NOT the issue:**
   - Deep dive into vLLM's multimodal processing
   - Check if vision tower is being called correctly
   - Verify image data format matches vLLM expectations
