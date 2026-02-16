# TruthRL MLLM Debugging: Gemma 3 4B VSR

This document summarizes the issues resolved during the setup and debugging of multimodal training for **Gemma 3 4B Instruct** using the **verl** library and **GRPO** algorithm.

## 1. FSDP Configuration Error
- **Issue**: `TypeError: FSDPEngineConfig.__init__() got an unexpected keyword argument 'transformer_layer_cls_to_wrap'`
- **Cause**: The Hydra configuration path for FSDP wrapping was incorrectly structured.
- **Fix**: Corrected the override path to:
  `+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer,SiglipEncoderLayer]`

## 2. vLLM Weight Loading Failure
- **Issue**: `KeyError: 'vision_model.encoder.layers.0.self_attn.qkv_proj.base_layer.weight'`
- **Cause**: vLLM currently only supports LoRA for the language model. Applying LoRA to the vision tower causes a weight mapping mismatch during weight syncing between the Actor and Rollout engine.
- **Fix**: Explicitly excluded the vision tower from LoRA application using a global regex:
  `+actor_rollout_ref.model.exclude_modules='.*vision_tower.*'`

## 3. Hydra Parsing Grammar Error
- **Issue**: `hydra.errors.OverrideParseException: mismatched input '(' expecting <EOF>`
- **Cause**: Complex regex strings containing parentheses and pipes `(a|b)` in command-line arguments confused the Hydra parser.
- **Fix**: Reverted `target_modules` to a standard list format `[q_proj, k_proj, ...]` which Hydra handles natively.

## 4. Reward Function Signature Mismatch
- **Issue**: `TypeError: compute_score() got an unexpected keyword argument 'data_source'`
- **Cause**: The `verl` trainer passes additional metadata (like `data_source`) to the custom reward function, which the original signature did not accept.
- **Fix**: Added `**kwargs` to the `compute_score` function in `vsr_lexical.py` to gracefully handle extra metadata.

## 5. Background Environment Issues
- **Issue**: `ModuleNotFoundError: No module named 'verl'` when running via `nohup`.
- **Cause**: The script defaulted to the system `python3` instead of the specific conda environment, losing access to installed libraries.
- **Fix**: Updated the training shell script to use the absolute path of the environment's Python:
  `/home/kalashkala/miniconda3/envs/truthrl-verl/bin/python`

## 6. Shell Script Syntax
- **Issue**: Intermittent failures when launching scripts.
- **Cause**: Escaped newlines and quotes were improperly placed in the `.sh` file.
- **Fix**: Cleaned up the multi-line command formatting to ensure robust execution in background sessions.

## 7. Image Token Mismatch (Model Blindness)
- **Issue**: The model was outputting long strings of gibberish and the `<image>` tag appeared missing in training logs.
- **Cause**: Gemma 3 does not use the literal text `"<image>"` as its special vision token. It requires a specific special token `🖼️` (ID `255999`). The literal string `"<image>"` was being tokenized as three separate text tokens, causing the model to ignore the image pixels entirely.
- **Fix**: Ran a data correction script to replace all occurrences of `"<image>"` with the actual special character `processor.image_token` (`🖼️`) in `train.parquet` and `test.parquet`. Verified fix with `scripts/dry_run_mllm.py`, confirming the model now "sees" the image and generates valid VSR responses.

## 8. Multimodal Placeholder Mismatch (vLLM Tensor Error)
- **Issue**: `ValueError: Attempted to assign 256 = 256 multimodal tokens to 512 placeholders`
- **Cause**: **Double Expansion**. `verl`'s HF processor expanded the image token into 256 soft tokens. `vLLM`'s internal processor received this *already expanded* prompt and expanded the placeholder *again*, resulting in 512 tokens for 256 vision features.
- **Fix**: ✅ **Token Collapse Strategy**. Implemented `_collapse_multimodal_tokens` in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`. This utility strips the 256 soft tokens back to a single `Gemma3` placeholder (ID 255999) *just before* sending the request to vLLM.
- **Result**:
    - **Actor**: Training on full 256-token sequence (Correct).
    - **vLLM**: Receives 1 placeholder, internally expands to 256 (Correct).
    - **Crash**: Resolved.
    - **Patches**: All library hot-patches (forcing single crop, auto-alignment) have been **REVERTED**. The codebase is now clean.

## 9. Model Blindness (Gibberish/Out-of-Vocab Output)
- **Issue**: `vLLM Response: ''` or repeating out-of-vocab IDs (e.g., `262207`).
- **Resolution**: Identified two secondary factors corrupting the vision-to-language alignment: **Pan & Scan** and **Dtype Precision**.

## 10. vLLM Configuration: Pan and Scan Alignment
- **Issue**: Even with token IDs aligned, vLLM's default Gemma 3 behavior is to "Pan and Scan" (crop high-res images), which expects **512+ tokens** (1 base + 1 crop).
- **Cause**: The `verl` dataset processor only emits **256 tokens** (single-crop). This caused a mismatch between the number of vision features (256) and the number of placeholder slots vLLM created (512).
- **Fix**: ✅ **Explicit Config**. Passed `mm_processor_kwargs={"do_pan_and_scan": False}` to the vLLM engine initialization. This forces vLLM to use single-crop behavior, matching the training data exactly.

## 11. Tooling: Dtype Precision (bfloat16)
- **Issue**: Using `float16` for inference caused the vision encoder outputs to overflow/underflow, resulting in garbage "out-of-vocabulary" tokens.
- **Fix**: Verified that the model must run in **`bfloat16`** (its native training precision) for stable vision features.
- **Status**: ✅ **Confirmed**. The dry run script now correctly describes visual details (e.g., *"a fluffy white poodle wearing a yellow bow"*) using the collapsed token-ID path.

---
### **Final Stable Architecture**
1. **No Library Patches**: `vllm/model_executor/models/gemma3_mm.py` is fully restored to original code.
2. **verl Rollout Fix**: The worker now intelligently "collapses" tokens before inference.
3. **vLLM Engine Fix**: Configured to disable Pan & Scan for 1:1 token alignment.
4. **Ready for Training**: Use `bash train_grpo_vsr_gemma_4b.sh`.
