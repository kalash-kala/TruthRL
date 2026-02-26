# Memory Optimization Guide for VSR Qwen2.5-VL (LoRA + GRPO)

This document explains the relationship between training parameters and System RAM (CPU memory) / GPU VRAM usage during Vision-RL training.

## The "Experience Buffer" Equation

In the `verl` framework, the total amount of data generated and held in memory per global step is defined as:

> **Total Experience (S) = (Prompts per Step [P]) × (Rollouts per Prompt [G])**

### Configuration Comparisons

| Run Type              | Prompts ($P$) | Rollouts ($G$) | Total Images ($S$) | RAM Status     |
| :-------------------- | :------------ | :------------ | :----------------- | :------------- |
| **Full Fine-Tuning**  | 16            | 2             | **32**             | ✅ STABLE       |
| **Initial LoRA Run**  | 64            | 8             | **512**            | ❌ OOM (Host RAM) |
| **Updated LoRA Run**  | 16            | 4             | **64**             | ❌ OOM (Transient) |
| **2-GPU A100 Run**    | 16            | 8             | **128**            | ✅ STABLE       |

## Systemic Root Cause Analysis: LoRA vs. Full Fine-tuning

The primary bottleneck for scaling Vision-RL on Node 1 (8 GPUs, 250GB RAM) is the **RAM-per-GPU ratio**, which is only **~31GB**.

### 1. The Fixed Overhead (The "Ghost" in the Machine)
In `actor_rollout_ref` mode, `verl` loads a **Reference Model** copies for every worker. By default, these are wrapped in `CPUOffload`, meaning they live entirely in **System RAM**.
*   **Cost:** ~7GB per worker.
*   **Total:** 7GB × 8 GPUs = **56GB** (Hidden fixed cost before any data is processed).

### 2. Transient Replication (The LoRA "Last Straw")
*   **Full Fine-tuning:** Parameters are sharded by FSDP *immediately* during load. Each worker only keeps $1/8^{\text{th}}$ of the model in RAM (~0.9GB).
*   **LoRA Training:** Initializing a `PeftModel` requires loading the **full base model** into CPU memory before adapters are created. This creates a **transitory 7GB peak per worker**.
*   **The Math of Failure:** 7GB (Ref) + 7GB (LoRA Init) + 15GB (vLLM & Data) = **~29GB per worker**.
*   On 8 GPUs: 29GB × 8 = **232GB**. Combined with OS and Ray overhead, this hits the 250GB limit.

### 3. Hardware Strategy: High RAM per GPU
The solution is not more total RAM, but more **RAM per process**.
*   **Server A (8 GPUs, 250GB RAM):** 31GB per GPU. (FAIL for LoRA)
*   **Server B (2 GPUs, 210GB RAM):** 105GB per GPU. (STABLE for LoRA)

## Key Learnings

### 1. System RAM is the Bottleneck for Vision-RL
While LoRA reduces **GPU VRAM** usage (by only training small adapters), it does NOT reduce the **System RAM** required to store the images. Each rollout includes vision tokens and original images. When $S = 512$, the host machine (250GB RAM) had to hold hundreds of high-resolution images across all 8 Ray workers, causing the "most recently scheduled task" to be killed.

### 2. The GRPO Advantage Trade-off
GRPO relies on comparing multiple rollouts from the same prompt to calculate "relative" advantage without a separate Critic model.
*   **$G=2$:** Fast, but "relative" rewards are noisy (only two versions to compare).
*   **$G=4$ to $G=8$:** Highly recommended for stable reasoning exploration.

### 3. PPO Batching Relationship
Optimizing the model happens after sampling:
*   **`ppo_mini_batch_size` (M):** The actual number of sequences processed per gradient update.
*   **Cycles:** $(P \times G) / M$ = Number of optimization updates from a single "sampling heartbeat."

### 4. Epoch and Step Calculation
Progress through the dataset (epochs) is determined ONLY by how many unique prompts are pulled from the pool.

> **Steps per Epoch = Total Prompts in Dataset / Prompts per Step (P)**

*   **Example:** With 750 prompts and $P=16$, one epoch $\approx$ 47 global steps.
*   **Crucial Note:** Changing the group size ($G$) or mini-batch size ($M$) does NOT change the number of steps per epoch, it only changes the amount of work/updates done *within* each step.

### 5. Detailed Step-by-Step Cycle (Example Step)
In a single **Global Step** with $P=16, G=4, M=16$:

1.  **Sampling Phase:**
    *   Pull **16 unique prompts** from the pool.
    *   Generate **4 rollouts** for each prompt (Total: **64 sequences**).
    *   Calculate rewards and GRPO relative advantages for all 64 sequences.
2.  **Optimization Phase:**
    *   Take the **64 experience sequences** collected above.
    *   Divide them by the `ppo_mini_batch_size` ($M=16$).
    *   Perform **4 gradient updates** (64 / 16 = 4) to the model weights.
    *   If a gradient update is too large for GPU VRAM, it is further split by the **PPO Micro-batch Size**.

## Recommended Deployment Strategy

### For Node 1 (8 GPUs / 250GB RAM)
*   **USE FULL FINE-TUNING.** It is more memory-efficient during initialization due to immediate sharding.
*   If LoRA is mandatory: Reduce to **4 GPUs** (`CUDA_VISIBLE_DEVICES=0,1,2,3`) to double the RAM-per-GPU headroom.

### For A100 Node (2 GPUs / 210GB RAM)
*   **USE LORA.** You have 105GB per GPU, plenty of room for transient peaks.
*   **Group Size ($G$):** 8 (Safe to increase for better reasoning metrics).
*   **vLLM Util:** 0.75 (Safe given 80GB VRAM).

---
**Conclusion:** Total System RAM must be $\ge (\text{Worker Overhead} \times \text{Number of GPUs})$. For Qwen2.5-VL-3B LoRA, aim for $> 50\text{GB}$ per GPU.
