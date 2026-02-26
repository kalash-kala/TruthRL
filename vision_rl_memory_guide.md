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
| **Updated LoRA Run**  | 16            | 4             | **64**             | ✅ STABLE       |

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

## Recommended Settings for 8-GPU Node (250GB RAM)
To maintain stability with Qwen2.5-VL-3B:

*   **Prompts per step ($P$):** 16 (Keeps vision data manageable in Ray/Host memory).
*   **Group Size ($G$):** 4 to 8 (Essential for GRPO "reasoning" stability).
*   **vLLM Util:** 0.7 - 0.75 (Safe for LoRA, lower to 0.4 for Full Fine-tuning).
*   **`val_before_train`:** **Set to False**. The initial validation broadcast of 200+ images is the most likely time for a System RAM spike.

---
**Note:** To prioritize getting the script running, use $P=16, G=4$. If stable, trial $G=8$ with $P=16$ for better reasoning quality.
