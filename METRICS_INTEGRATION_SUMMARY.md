# TruthRL Metrics Integration & Training Monitoring Summary

This document summarizes the enhancements made to the **TruthRL** training pipeline (based on the `verl` framework) to enable comprehensive tracking, visualization, and safety monitoring for both training and validation processes.

## 1. Objective
Common challenges in Reinforcement Learning (RL) training include high metric noise, "reward hacking," and model collapse due to KL divergence spikes. Our integration aims to provide:
*   **Mathematical Smoothness**: Clean per-epoch averages for noisy metrics like Policy Loss.
*   **Evaluation Alignment**: Side-by-side comparison of training progress and validation accuracy.
*   **Proactive Safety**: Real-time detection of training instability (KL spikes).

---

## 2. Implemented Features

### A. Epoch-Level Metric Aggregation
We modified the `RayPPOTrainer` to include a centralized aggregator that collects metrics from every training step.
*   **Feature**: Metrics are accumulated during the 46 steps of an epoch and averaged at the end.
*   **Impact**: New metrics are logged under the **`epoch/`** prefix (e.g., `epoch/actor/pg_loss`, `epoch/critic/vf_loss`).
*   **Benefit**: This eliminates the high-frequency noise from "per-step" graphs, making it easier to see long-term training convergence.

### B. Integral Training Accuracy Tracking
The VSR (Visual Spatial Reasoning) reward function was upgraded to return binary accuracy alongside the reward score.
*   **Feature**: The training loop now extracts `accuracy` from every batch.
*   **Integration**: Training accuracy is now logged as both `training/accuracy` (per step) and `epoch/training/accuracy` (per epoch).
*   **Benefit**: Users can monitor the model's performance on the *training set* in real-time without waiting for the validation loop.

### C. Proactive KL Spike Detection
To protect training from collapse, a "Spike Warning System" was integrated into the driver process.
*   **Mechanism**: A rolling 10-step history of KL Divergence (`actor/ppo_kl`) is maintained.
*   **Logic**: If the current KL exceeds **2.5x** the recent average (and a minimum threshold of 0.1), a high-visibility warning is emitted in the console.
*   **Benefit**: Immediate notification of "Reward Hacking" or sudden model drift, enabling manual intervention before wasting compute resources.

### D. Validation & Logging Alignment
Training scripts were optimized to ensure evaluation matches the real training progress.
*   **`test_freq=46`**: Adjusted to match exactly one epoch (for batch size 16 on VSR sample).
*   **Logger Integration**: TensorBoard was added as a default backend alongside Console and WandB.
*   **Adaptive KL Control**: Training scripts now default to `algorithm.kl_ctrl.type=adaptive` to automatically stabilize KL throughout the run.

---

## 3. Distributed Architecture Reliability
The tracking system is designed for **Ray/Multi-GPU** settings:
1.  **Single-Source Truth**: The `RayPPOTrainer` driver acts as the global controller for aggregation.
2.  **Global Averaging**: Metrics received by the driver are already averaged across all GPUs/Nodes via Ray’s reduction mechanism.
3.  **Low Overhead**: Aggregation happens on CPU at the end of batches, ensuring minimal impact on GPU throughput.

---

## 4. Key Metrics to Watch
| Metric | Category | Meaning |
| :--- | :--- | :--- |
| `epoch/training/accuracy` | Performance | Average % correct on current training data. |
| `val-core/VSR/acc/mean` | Evaluation | Generalization performance on unseen test data. |
| `epoch/actor/pg_loss` | Convergence | Clean trend of Policy Gradient updates. |
| `actor/ppo_kl` | Stability | Model drift from the reference policy (Watch for Spike Warnings). |
| `actor/kl_coef` | Control | The current "strength" of the KL penalty (High = Pulling back). |
