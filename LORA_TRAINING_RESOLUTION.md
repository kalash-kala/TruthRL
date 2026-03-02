# LoRA Training: vLLM Weight Loading & Memory Optimization

This document outlines the resolution for the `KeyError: 'layers.0.self_attn.qkv_proj.weight'` encountered during Qwen2.5-VL LoRA training and explains the internal mechanism of weight synchronization between FSDP and vLLM.

## 1. The Issue: `KeyError: 'qkv_proj.weight'`

### Symptom
During the transition from the training phase to the rollout phase, the process would crash with:
```python
KeyError: 'layers.0.self_attn.qkv_proj.weight'
```

### Root Cause
The training script was using the default `load_format=dummy_hf` (or `dummy_dtensor`). This configuration triggers a **Manual Weight Injection** mechanism:

1.  **Manual Sync**: The `FSDPVLLMShardingManager` attempts to collect the *entire* base model state dict from the FSDP training module.
2.  **Key Mismatch**: The training-side state dict contains split weights: `q_proj.weight`, `k_proj.weight`, and `v_proj.weight`.
3.  **vLLM Expectation**: Internally, vLLM's Qwen2 implementation expects a fused `qkv_proj.weight` key when `load_weights()` is called manually.
4.  **Failure**: Since the injected dictionary lacks the fused key, vLLM throws a `KeyError`.

## 2. The Solution: `safetensors` + `layered_summon`

The resolution involves changing the weight loading strategy in the training launch script:

```bash
actor_rollout_ref.rollout.load_format=safetensors
actor_rollout_ref.rollout.layered_summon=True
```

### Why this works:
*   **`load_format=safetensors`**: This tells vLLM to load the base model weights **directly from the disk** using its own optimized loader. vLLM handles the Q/K/V fusion internally during the disk load.
*   **Decoupling**: By loading from disk, we avoid the "Manual Weight Injection" path for base weights. The training process no longer needs to push 7B+ parameters through the network/memory bus on the first step.
*   **LoRA Only Sync**: Once the base weights are loaded from disk, `verl` switches to syncing only the LoRA deltas (adapters) via `add_lora()`. These deltas match the expected format perfectly.

## 3. Internal Mechanism: How LoRA Weights are Loaded

The integration uses the `FSDPVLLMShardingManager` to bridge the gap between FSDP (Training) and vLLM (Inference).

### The Synchronization Flow:

1.  **Initialization**:
    *   If `load_format=safetensors`, `self.base_sync_done` is set to `True`.
    *   vLLM initializes the base model using the local checkpoint.

2.  **The Generation Phase (`__enter__`)**:
    *   Before rollouts begin, the manager calls `__collect_lora_params()`.
    *   With **`layered_summon=True`**, it uses a memory-efficient strategy: it iterates through the FSDP model layer-by-layer, fetching only the LoRA parameters (`lora_A`, `lora_B`).
    *   This is much faster and uses significantly less VRAM/System RAM than a full model `summon_full_params()`.

3.  **Weight Injection (`update_params`)**:
    *   The collected LoRA tensors are wrapped in a `TensorLoRARequest`.
    *   This request is sent to the vLLM engine via `self.inference_engine.llm_engine.add_lora(lora_request)`.
    *   vLLM applies these adapter weights on top of the base weights already in memory.

## 4. Summary of Benefits
*   **Stability**: Eliminates `KeyError` by letting vLLM handle complex base weight fusion.
*   **Memory Efficiency**: `layered_summon` prevents OOMs during weight collection.
*   **Speed**: Transferring tiny LoRA adapters is orders of magnitude faster than transferring the full model weights between training and inference ranks.

---
**Note**: Always ensure your `MODEL_PATH` contains valid `.safetensors` files and a `model.safetensors.index.json` for this mechanism to work correctly.
