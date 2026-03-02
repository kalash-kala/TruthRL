# Dynamic Progressive Reward for VSR Training

This implementation introduces a dynamic negative reward (penalty) for incorrect or hallucinated answers during the training of the Visual Spatial Reasoning (VSR) model. The goal is to progressively motivate the model to abstain ("I don't know") when it is unsure, while allowing for exploration in the early stages of training.

## 🚀 Reward Mechanism

The reward function evaluates model outputs against ground truth labels ("True"/"False"). The scoring schedule is as follows:

| Outcome | Reward Score |
| :--- | :--- |
| **Correct Answer** | `+1.0` |
| **"I don't know" / Refusal** | `0.0` |
| **Incorrect / Hallucination** | **`-1.0 * (1.0 + current_step / steps_per_epoch)`** |

### Progressive Penalty Schedule
The penalty for incorrect answers scales linearly with the training progress. For a training run with **5 epochs**, the penalty evolves as follows:

- **Start (Step 0):** `-1.0`
- **End of Epoch 1:** `-2.0`
- **End of Epoch 2:** `-3.0`
- **...**
- **End of Epoch 5:** **`-6.0`**

Since the reward for "I don't know" remains constant at `0.0`, the model is increasingly incentivized to refuse to answer rather than risk a wrong guess as training progresses.

---

## 🛠️ Implementation Details

### 1. Reward Function: `vsr_lexical_dynamic.py`
Located at: `verl/utils/reward_score/vsr_lexical_dynamic.py`
- Implements the progressive scoring logic.
- Maintains a module-level `_StepTracker` as a fallback if environment variables are not present.
- Normalizes predicted answers and handles extraction from model output blocks (e.g., `/box[...]`).

### 2. Trainer Integration: `ray_trainer.py`
Located at: `verl/trainer/ppo/ray_trainer.py`
To synchronize the reward function with the training progress, the following environment variables are exported in the training loop:
- `VERL_GLOBAL_STEP`: Current global training step.
- `VERL_TOTAL_STEPS_PER_EPOCH`: Number of steps in a single epoch.
- `VERL_TOTAL_TRAINING_STEPS`: Total steps planned for the entire run.

---

## 🔦 How to Use

To enable the dynamic reward in your training pipeline, update the `REWARD_FN_PATH` in your launch script (e.g., `train_grpo_vsr_qwen2_5_vl_3b_2gpu_a100_lora.sh`):

```bash
# Update the reward function path to point to the dynamic implementation
REWARD_FN_PATH=/home/debarpanb1/kalashkala/TruthRL/training/verl/verl/utils/reward_score/vsr_lexical_dynamic.py
```

No other changes to the launch command or YAML configuration are required.

---

## 📈 Monitoring
The reward function returns the raw penalty value in the `negative_reward` key. You can monitor this value in your logs or dashboard (e.g., Weights & Biases) under `critic/rewards/negative_reward` to verify the penalty schedule is progressing as expected.
