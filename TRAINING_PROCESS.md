# TruthRL Training Process Documentation

This document explains the overall training process and important Python functions that get called when executing the [`train_grpo.sh`](file:///home/sriramg/kalashabhayk/TruthRL/train_grpo.sh) script.

## Table of Contents
- [Overview](#overview)
- [Training Script Entry Point](#training-script-entry-point)
- [Main Training Pipeline](#main-training-pipeline)
- [Key Components and Functions](#key-components-and-functions)
- [Training Loop Details](#training-loop-details)
- [Data Flow](#data-flow)
- [Response Display and Logging](#response-display-and-logging)

---

## Overview

TruthRL uses **GRPO (Group Relative Policy Optimization)** to fine-tune language models with reinforcement learning. The training process uses:

- **VERL** (Versatile Event-driven Reinforcement Learning) framework for distributed training
- **Ray** for distributed computing and resource management
- **FSDP** (Fully Sharded Data Parallel) for efficient model parallelism
- **vLLM** for fast inference during rollout generation
- **External Verifier** (Google Gemma 3 27B Instruct) for reward computation via OpenAI-compatible API

---

## Training Script Entry Point

### 1. **Shell Script: `train_grpo.sh`**

The training starts by executing:
```bash
bash ./train_grpo.sh
```

**Key Configuration:**
- **Actor Model**: `meta-llama/Llama-3.1-8B-Instruct` (with LoRA)
- **Verifier Model**: Hosted at `http://10.128.0.30:8000/v1` (Google Gemma 3 27B Instruct)
- **Algorithm**: GRPO (advantage estimator)
- **Batch Size**: 8
- **Learning Rate**: 1e-6
- **Hardware**: Single H100 GPU with memory optimizations

**Main Python Command:**
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    actor_rollout_ref.model.path=$MODEL_NAME \
    # ... (many more Hydra configuration overrides)
```

---

## Main Training Pipeline

### 2. **Entry Point: `main_ppo.py`**
**Location:** [`training/verl/verl/trainer/main_ppo.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/main_ppo.py)

#### **`main(config)`** (Line 33-40)
- Decorated with `@hydra.main` for configuration management
- Calls `run_ppo(config)` to start the training process

#### **`run_ppo(config)`** (Line 44-83)
**Purpose:** Initialize Ray cluster and launch distributed training

**Key Steps:**
1. **Initialize Ray** (lines 53-61)
   - Sets up distributed computing environment
   - Configures runtime environment with PPO-specific settings
   
2. **Create Remote TaskRunner** (lines 64-77)
   - Instantiates `TaskRunner` as a Ray remote actor
   - Executes `TaskRunner.run()` remotely
   
3. **Optional Timeline Generation** (lines 79-83)
   - Saves performance timeline for analysis

---

### 3. **TaskRunner Class** (Line 86-244)

#### **`TaskRunner.run(config)`** (Line 94-244)
**Purpose:** Main orchestration of the training workflow

**Critical Steps:**

##### **A. Model and Tokenizer Setup** (lines 115-127)
```python
# Download checkpoint from HDFS/remote storage to local
local_path = copy_to_local(config.actor_rollout_ref.model.path, ...)

# Load tokenizer and processor
tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
```

##### **B. Worker Class Definition** (lines 129-168)
```python
# For FSDP strategy
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

# Define worker classes based on configuration
actor_rollout_cls = ActorRolloutRefWorker  # or AsyncActorRolloutRefWorker
```

##### **C. Resource Pool Setup** (lines 170-208)
```python
# Map roles to worker classes
role_worker_mapping = {
    Role.ActorRollout: ray.remote(actor_rollout_cls),
    Role.Critic: ray.remote(CriticWorker),
}

# Define resource pool specification (GPU allocation)
resource_pool_spec = {
    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
```

##### **D. Load Reward Manager** (lines 210-216)
```python
reward_fn = load_reward_manager(config, tokenizer, num_examine=0, ...)
val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, ...)
```
- **Function:** [`load_reward_manager()`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/reward.py#L93-L148)
- Loads custom or default reward function
- Uses OpenAI API to call external verifier model
- Returns `AbstractRewardManager` instance

##### **E. Dataset Creation** (lines 219-224)
```python
train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
train_sampler = create_rl_sampler(config.data, train_dataset)
```

**Function:** `create_rl_dataset()` (lines 247-294)
- Loads data from parquet files (training and validation)
- Uses `RLHFDataset` or custom dataset class
- Tokenizes prompts and prepares data for RL training

**Function:** `create_rl_sampler()` (lines 297-336)
- Creates `RandomSampler` or `SequentialSampler` for data iteration
- Supports curriculum learning with custom samplers

##### **F. Initialize Trainer** (lines 226-240)
```python
trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    processor=processor,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    collate_fn=collate_fn,
    train_sampler=train_sampler,
)
```

##### **G. Start Training** (lines 241-244)
```python
trainer.init_workers()  # Initialize distributed workers
trainer.fit()            # Start the training loop
```

---

## Key Components and Functions

### 4. **RayPPOTrainer Class**
**Location:** [`training/verl/verl/trainer/ppo/ray_trainer.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py)

#### **`init_workers()`** (Lines 780-883)
**Purpose:** Initialize all distributed worker groups

**Steps:**
1. **Create Resource Pools** (line 787)
   ```python
   self.resource_pool_manager.create_resource_pool()
   ```

2. **Initialize Actor-Rollout Worker** (lines 792-800)
   - Hybrid engine that handles both actor training and rollout generation
   - Uses FSDP for training, vLLM for fast inference

3. **Initialize Critic Worker** (lines 804-809)
   - Value function estimator (if using GAE/PPO)
   - Not used in GRPO

4. **Initialize Reference Policy** (lines 811-820)
   - Frozen copy of initial model for KL divergence computation
   - Used if `use_kl_loss=True` or `use_kl_in_reward=True`

5. **Initialize Reward Model Worker** (lines 822-827)
   - Optional model-based reward function
   - Not used when using external API-based verifier

6. **Create Worker Groups** (lines 829-883)
   ```python
   for resource_pool, class_dict in self.resource_pool_to_cls.items():
       worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
       wg_dict = self.ray_worker_group_cls(
           resource_pool=resource_pool,
           ray_cls_with_init=worker_dict_cls,
       )
       spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
       all_wg.update(spawn_wg)
   ```
   - Creates colocated worker groups for efficient resource utilization
   - Initializes models for each worker group
   - **Important:** Rollout worker initialized last for better vLLM memory estimation

---

#### **`fit()`** (Lines 1039-1398)
**Purpose:** Main training loop

**Initialization Steps:**

1. **Setup Logging** (lines 1050-1055)
   ```python
   logger = Tracking(
       project_name=self.config.trainer.project_name,
       experiment_name=self.config.trainer.experiment_name,
       default_backend=self.config.trainer.logger,  # ["console", "wandb"]
       config=OmegaConf.to_container(self.config, resolve=True),
   )
   ```

2. **Load Checkpoint** (line 1060)
   ```python
   self._load_checkpoint()
   ```
   - Resumes from latest checkpoint if available
   - Loads actor, critic, and dataloader states

3. **Initial Validation** (lines 1064-1070)
   ```python
   if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
       val_metrics = self._validate()
       logger.log(data=val_metrics, step=self.global_steps)
   ```

**Training Loop (Per Step):**

The loop iterates over epochs and batches (lines 1092-1398):

```python
for epoch in range(self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        # ... training step
```

---

### Training Loop Details

Each training step consists of the following phases:

#### **Phase 1: Batch Preparation** (Lines 1104-1134)
```python
batch: DataProto = DataProto.from_single_dict(batch_dict)

# Add unique IDs to each sample
batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

# Separate generation inputs from full batch
gen_batch = batch.pop(
    batch_keys=["input_ids", "attention_mask", "position_ids"],
    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", ...]
)

# Repeat batch for multiple rollout samples (n=2 by default)
gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
```

#### **Phase 2: Rollout Generation** (Lines 1138-1147)
```python
with marked_timer("gen", timing_raw, color="red"):
    if not self.async_rollout_mode:
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
    else:
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
```

**Function:** `ActorRolloutRefWorker.generate_sequences()` 
**Location:** [`training/verl/verl/workers/fsdp_workers.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L743-L778)

**Purpose:** Generate response sequences using vLLM

**Key Steps:**
1. Synchronize LoRA weights from FSDP actor to vLLM engine
2. Generate responses using vLLM for fast batched inference
3. Return `DataProto` with generated token IDs and rollout log probabilities

#### **Phase 3: Reward Computation** (Lines 1186-1202)
```python
with marked_timer("reward", timing_raw, color="yellow"):
    # Model-based reward (if enabled)
    if self.use_rm:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
    
    # Custom reward function (external verifier API)
    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

**Function:** `compute_reward()`
**Location:** [`training/verl/verl/trainer/ppo/reward.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/reward.py#L151-L169)

**Flow:**
1. `reward_fn(data, return_dict=True)` → calls reward manager
2. Reward manager decodes responses and calls custom `compute_score` function
3. For TruthRL: Calls OpenAI API (`http://10.128.0.30:8000/v1`) with Gemma 3 27B Instruct verifier
4. Returns token-level reward tensor

#### **Phase 4: Recompute Log Probabilities** (Lines 1203-1220)
```python
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    batch = batch.union(old_log_prob)
```

**Function:** `ActorRolloutRefWorker.compute_log_prob()`
**Location:** [`training/verl/verl/workers/fsdp_workers.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L780-L820)

**Purpose:** Compute log probabilities using current FSDP actor model for PPO ratio calculation

#### **Phase 5: Reference Policy Log Probabilities** (Lines 1221-1228)
```python
if self.use_reference_policy:
    with marked_timer("ref", timing_raw, color="olive"):
        if not self.ref_in_actor:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        else:
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

**Function:** `ActorRolloutRefWorker.compute_ref_log_prob()`
**Location:** [`training/verl/verl/workers/fsdp_workers.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L822-L857)

**Purpose:** Compute log probabilities from frozen reference model for KL penalty

#### **Phase 6: Value Estimation** (Lines 1230-1234)
```python
if self.use_critic:
    with marked_timer("values", timing_raw, color="cyan"):
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)
```
*Note:* Not used in GRPO (critic_warmup is set to 0)

#### **Phase 7: Advantage Computation** (Lines 1236-1269)
```python
with marked_timer("adv", timing_raw, color="brown"):
    # Apply KL penalty to rewards (if enabled)
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    
    # Compute advantages
    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,  # "grpo"
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=self.config.algorithm,
    )
```

**Function:** `apply_kl_penalty()`
**Location:** [`training/verl/verl/trainer/ppo/ray_trainer.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L154-L193)
- Computes KL divergence: `KL = log_prob - ref_log_prob`
- Updates rewards: `reward = score - kl_coef * KL`

**Function:** `compute_advantage()`
**Location:** [`training/verl/verl/trainer/ppo/ray_trainer.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L214-L291)

**GRPO Advantage Estimator:**
1. Groups responses by prompt UID
2. Computes mean reward per group
3. Advantage = `(reward - mean_reward) / std_reward` (normalized)
4. Encourages diverse, high-reward responses

#### **Phase 8: Update Critic** (Lines 1271-1276)
```python
if self.use_critic:
    with marked_timer("update_critic", timing_raw, color="pink"):
        critic_output = self.critic_wg.update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```
*Note:* Skipped in GRPO (critic_warmup > global_steps)

#### **Phase 9: Update Actor** (Lines 1278-1285)
```python
if self.config.trainer.critic_warmup <= self.global_steps:
    with marked_timer("update_actor", timing_raw, color="red"):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
```

**Function:** `ActorRolloutRefWorker.update_actor()`
**Location:** [`training/verl/verl/workers/fsdp_workers.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L699-L741)

**Purpose:** Update actor model using PPO loss

**Key Steps:**
1. Create mini-batches from full batch
2. For each mini-batch:
   - Compute new log probabilities
   - Calculate PPO clipped loss
   - Calculate KL loss (if enabled)
   - Backpropagate and update LoRA parameters
3. Synchronize FSDP model gradients
4. Return training metrics (loss, learning rate, etc.)

#### **Phase 10: Validation** (Lines 1314-1324)
```python
if (
    self.val_reward_fn is not None
    and self.config.trainer.test_freq > 0
    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
):
    with marked_timer("testing", timing_raw, color="green"):
        val_metrics: dict = self._validate()
        if is_last_step:
            last_val_metrics = val_metrics
    metrics.update(val_metrics)
```

**Function:** `_validate()`
**Location:** [`training/verl/verl/trainer/ppo/ray_trainer.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L628-L778)

**Steps:**
1. Generate responses for validation dataset
2. Compute rewards using `val_reward_fn`
3. Calculate metrics (accuracy, mean reward, etc.)
4. Log generations to WandB
5. Save validation results to disk

#### **Phase 11: Checkpoint Saving** (Lines 1326-1346)
```python
esi_close_to_expiration = should_save_ckpt_esi(...)
if self.config.trainer.save_freq > 0 and (
    is_last_step
    or self.global_steps % self.config.trainer.save_freq == 0
    or esi_close_to_expiration
):
    with marked_timer("save_checkpoint", timing_raw, color="green"):
        self._save_checkpoint()
```

**Function:** `_save_checkpoint()`
**Location:** [`training/verl/verl/trainer/ppo/ray_trainer.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L885-L941)

**Saves:**
- Actor model (LoRA adapters)
- Critic model (if used)
- Dataloader state (for resumption)

#### **Phase 12: Logging and Progress** (Lines 1362-1389)
```python
# Collect all metrics
metrics.update({
    "training/global_step": self.global_steps,
    "training/epoch": epoch,
})
metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

# Log to WandB/console
logger.log(data=metrics, step=self.global_steps)

# Update progress bar
progress_bar.update(1)
self.global_steps += 1
```

---

## Data Flow

Here's a visual representation of the data flow through one training step:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Batch from DataLoader                                   │
│    → prompts, attention_mask, metadata                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Generate Responses (vLLM Rollout)                            │
│    Input:  gen_batch (prompts)                                  │
│    Output: responses, rollout_log_probs                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Compute Rewards (External Verifier)                          │
│    Input:  prompts + responses                                  │
│    API Call: Gemma 3 27B @ http://10.128.0.30:8000/v1           │
│    Output: token_level_scores                                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Recompute Log Probs (Current Actor - FSDP)                   │
│    Input:  prompts + responses                                  │
│    Output: old_log_probs, entropys                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Reference Log Probs (Frozen Reference Policy)                │
│    Input:  prompts + responses                                  │
│    Output: ref_log_probs                                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Apply KL Penalty & Compute Advantages                        │
│    • KL = old_log_probs - ref_log_probs                         │
│    • token_level_rewards = scores - kl_coef * KL                │
│    • GRPO: advantages = (reward - mean) / std (per group)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Update Actor (PPO Loss)                                      │
│    Input:  advantages, old_log_probs, responses                 │
│    • Compute new_log_probs                                      │
│    • ratio = exp(new_log_probs - old_log_probs)                 │
│    • policy_loss = -min(ratio * adv, clip(ratio) * adv)         │
│    • kl_loss = KL divergence                                    │
│    • total_loss = policy_loss + kl_coef * kl_loss               │
│    • Backprop → Update LoRA parameters                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Response Display and Logging

During training, you'll see LLM responses and their scores printed to the console. This section explains which files are responsible and how to control the output.

### Files Responsible for Display

#### **1. Primary File: `truthrl_qa.py`** ⭐
**Location:** [`training/verl/verl/utils/reward_score/truthrl_qa.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/utils/reward_score/truthrl_qa.py)

This is the **main file** responsible for the printed output during training.

**Key Functions:**

**a) Verifier API Client Setup** (Lines 395-399)
```python
client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"),  # http://10.128.0.30:8000/v1
    api_key=os.environ.get("OPENAI_API_KEY"),    # token-abc123
)
```

**b) LLM-as-a-Judge Scoring** (Lines 536-603)
- Function: `compute_score_llm_as_a_judge_binary(solution_str, ground_truth)`
- **Prints approximately 1 out of every 64 samples** due to randomization:
  ```python
  do_print = random.randint(1, 64) == 1  # Line 541
  ```

**Output Format:**
```
===================================
Question: <question from dataset>
Golden answers: <ground_truth>
Out-of-knowledge: False
Extracted answer: <model's answer from \boxed{}>
Solution string: <full model response>
>>>>>> Reward: 1 (exact match)
```
OR if using LLM judge:
```
>>>>>> Reward: 1 (LLM-as-a-judge: The prediction matches the ground truth.)
```
OR if incorrect:
```
>>>>>> Reward: -1 (no answer box)
>>>>>> Reward: -1 (LLM-as-a-judge: incorrect prediction)
```

**c) API Call to Verifier** (Lines 409-429)
```python
def attempt_api_call(messages, max_retries=3):
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",  # Your verifier model
        messages=messages,
        temperature=0,
        top_p=0.9,
        max_tokens=512,
    )
    return response.choices[0].message.content
```

**Verifier Prompt Structure:**
- **System Message**: Detailed instructions with 20+ in-context examples (Lines 20-215)
- **User Message**: `Question: {query}\n Ground truth: {gt}\n Prediction: {prediction}\n`
- **Response**: JSON with `{"score": 0 or 1, "explanation": "..."}`

**Other Scoring Variants:**
- `compute_score_llm_as_a_judge_ternary()` - Returns -1, 0, or 1
- `compute_score_llm_as_a_judge_binary_OOK()` - Handles out-of-knowledge questions
- `compute_score_llm_as_a_judge_ternary_OOK()` - Ternary scoring with OOK support

#### **2. Reward Manager: `naive.py`**
**Location:** [`training/verl/verl/workers/reward_manager/naive.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/reward_manager/naive.py)

The `NaiveRewardManager.__call__()` method (Lines 46-122) can also print output:

**Lines 100-111:**
```python
if already_print_data_sources[data_source] < self.num_examine:
    already_print_data_sources[data_source] += 1
    print("[prompt]", prompt_str)
    print("[response]", response_str)
    print("[ground_truth]", ground_truth)
    if isinstance(score, dict):
        for key, value in score.items():
            print(f"[{key}]", value)
    else:
        print("[score]", score)
```

**Control Parameter:**
- `self.num_examine` - Number of samples to print per data source
- Set when loading reward manager in [`main_ppo.py`][def]:
  - **Training**: `num_examine=0` (Line 212) → No printing from this manager
  - **Validation**: `num_examine=1` (Line 215) → Prints 1 sample per data source

#### **3. Reward Loading: `reward.py`**
**Location:** [`training/verl/verl/trainer/ppo/reward.py`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/reward.py)

**`load_reward_manager()`** (Lines 93-148) orchestrates the setup:
```python
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    # Load custom reward function from truthrl_qa.py
    compute_score = get_custom_reward_fn(config)
    
    # Get reward manager class (default: NaiveRewardManager)
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)
    
    # Instantiate with custom scoring function
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,  # Your truthrl_qa function
        reward_fn_key=config.data.reward_fn_key,
    )
```

### Data Flow for Logging

```
Training Step
    ↓
compute_reward(batch, reward_fn)
    ↓
NaiveRewardManager.__call__()
    • Decodes prompts and responses
    • Calls self.compute_score() for each sample
    ↓
compute_score_llm_as_a_judge_binary()  [truthrl_qa.py]
    • Extract answer from \boxed{...}
    • Check exact match first
    • If no match, call verifier API
    • Print output (1/64 samples)
    ↓
OpenAI API Call → Gemma 3 27B @ http://10.128.0.30:8000/v1
    • LLM-as-a-judge evaluates the prediction
    • Returns JSON: {"score": 1, "explanation": "..."}
    ↓
Return reward: -1 or 1
```

### Controlling Print Frequency

**Option 1: Modify `truthrl_qa.py` randomization**
- **Current**: `do_print = random.randint(1, 64) == 1` → Prints ~1.5% of samples
- **More frequent**: Change to `random.randint(1, 10) == 1` → Prints ~10% of samples
- **Less frequent**: Change to `random.randint(1, 100) == 1` → Prints ~1% of samples
- **Always print**: Change to `do_print = True` → Prints every sample (verbose!)

**Option 2: Modify `num_examine` parameter**
- This only affects `NaiveRewardManager` printing (currently disabled with `num_examine=0`)
- To enable: Change line 212 in `main_ppo.py` from `num_examine=0` to `num_examine=5`

### Scoring Logic

The reward computation follows this priority:

1. **No answer extracted** → Reward: `-1`
   - No `\boxed{...}` found in response

2. **Out-of-knowledge questions** (if applicable)
   - Answer "I don't know" → Reward: `1`
   - Other answer → Reward: `-1`

3. **Exact match** → Reward: `1`
   - Normalized prediction exactly matches ground truth
   - Normalization: lowercase, remove articles (a/an/the), remove punctuation

4. **LLM-as-a-judge evaluation** → Reward: `1` or `-1`
   - Calls Gemma 3 27B verifier
   - Uses detailed rubric with 20+ in-context examples
   - Checks for:
     - Numerical accuracy (with rounding tolerance)
     - Semantic equivalence
     - Self-contradictions
     - Answer relevance to question

### Example Output During Training

```bash
===================================
Question: what is the capital of france?
Golden answers: ['paris']
Out-of-knowledge: False
Extracted answer: paris
Solution string: <think>
France is a country in Europe. The capital city is Paris, which is known for the Eiffel Tower.
</think>
Final Answer: \boxed{paris}
>>>>>> Reward: 1 (exact match)
```

Or with LLM-as-a-judge:
```bash
===================================
Question: who is taller, michael jordan or lebron james?
Golden answers: ['lebron james']
Out-of-knowledge: False
Extracted answer: lebron james is taller
Solution string: <think>
Michael Jordan is 6'6" (1.98m) and LeBron James is 6'9" (2.06m).
</think>
Final Answer: \boxed{lebron james is taller}
>>>>>> Reward: 1 (LLM-as-a-judge: The prediction correctly identifies LeBron James as taller.)
```

---

## Summary of Key Functions

| **Function** | **Location** | **Purpose** |
|-------------|-------------|-------------|
| `main()` | [`main_ppo.py:33`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/main_ppo.py#L33-L40) | Hydra entry point |
| `run_ppo()` | [`main_ppo.py:44`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/main_ppo.py#L44-L83) | Initialize Ray and launch training |
| `TaskRunner.run()` | [`main_ppo.py:94`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/main_ppo.py#L94-L244) | Setup workers, datasets, and trainer |
| `create_rl_dataset()` | [`main_ppo.py:247`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/main_ppo.py#L247-L294) | Load and tokenize training data |
| `load_reward_manager()` | [`reward.py:93`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/reward.py#L93-L148) | Initialize reward function |
| `RayPPOTrainer.init_workers()` | [`ray_trainer.py:780`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L780-L883) | Create distributed worker groups |
| `RayPPOTrainer.fit()` | [`ray_trainer.py:1039`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L1039-L1398) | Main training loop |
| `ActorRolloutRefWorker.generate_sequences()` | [`fsdp_workers.py:743`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L743-L778) | Generate responses using vLLM |
| `compute_reward()` | [`reward.py:151`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/reward.py#L151-L169) | Call external verifier for rewards |
| `ActorRolloutRefWorker.compute_log_prob()` | [`fsdp_workers.py:780`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L780-L820) | Recompute log probs with current actor |
| `ActorRolloutRefWorker.compute_ref_log_prob()` | [`fsdp_workers.py:822`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L822-L857) | Compute reference log probs |
| `apply_kl_penalty()` | [`ray_trainer.py:154`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L154-L193) | Add KL penalty to rewards |
| `compute_advantage()` | [`ray_trainer.py:214`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/ppo/ray_trainer.py#L214-L291) | GRPO advantage estimation |
| `ActorRolloutRefWorker.update_actor()` | [`fsdp_workers.py:699`](file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/workers/fsdp_workers.py#L699-L741) | PPO policy update |

---

## Configuration Details

The training is controlled by Hydra configuration with many command-line overrides. Key parameters from [`train_grpo.sh`](file:///home/sriramg/kalashabhayk/TruthRL/train_grpo.sh):

**Data:**
- `data.train_files`: Path to training parquet file
- `data.train_batch_size=8`: Samples per training step
- `data.max_prompt_length=4096`: Maximum prompt tokens
- `data.max_response_length=1024`: Maximum response tokens

**Model:**
- `actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct`: Base model
- `actor_rollout_ref.model.lora_rank=16`: LoRA rank
- `actor_rollout_ref.model.lora_alpha=32`: LoRA alpha
- `actor_rollout_ref.model.use_remove_padding=True`: Memory optimization

**Training:**
- `actor_rollout_ref.actor.optim.lr=1e-6`: Learning rate
- `actor_rollout_ref.actor.ppo_mini_batch_size=8`: PPO mini-batch size
- `actor_rollout_ref.actor.use_kl_loss=True`: Enable KL divergence loss
- `actor_rollout_ref.actor.kl_loss_coef=0.001`: KL loss coefficient
- `actor_rollout_ref.actor.kl_loss_type=low_var_kl`: KL loss variant

**Rollout:**
- `actor_rollout_ref.rollout.name=vllm`: Use vLLM for generation
- `actor_rollout_ref.rollout.n=2`: Generate 2 responses per prompt
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.8`: vLLM memory limit
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`: No tensor parallelism

**FSDP Optimizations:**
- `actor_rollout_ref.actor.fsdp_config.param_offload=True`: Offload params to CPU
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`: Offload optimizer states
- `actor_rollout_ref.model.enable_gradient_checkpointing=True`: Activation checkpointing

**Algorithm:**
- `algorithm.adv_estimator=grpo`: Use GRPO advantage estimator
- `algorithm.use_kl_in_reward=True`: Add KL penalty to rewards

**Trainer:**
- `trainer.n_gpus_per_node=1`: Single GPU training
- `trainer.total_epochs=1`: One pass through data
- `trainer.save_freq=10`: Save checkpoint every 10 steps
- `trainer.test_freq=5`: Validate every 5 steps

---

## Additional Notes

### Memory Optimizations for Single H100 GPU
The configuration includes several memory-saving techniques:
1. **LoRA** instead of full fine-tuning (16 rank, 32 alpha)
2. **FSDP parameter offloading** to CPU
3. **Optimizer state offloading** to CPU
4. **Gradient checkpointing** for activations
5. **Remove padding** from sequences
6. **vLLM memory limit** (0.8 utilization)
7. **Small batch sizes** (8 samples/step, 4 micro-batch)

### External Verifier Model
The Google Gemma 3 27B Instruct verifier model must be hosted separately with an OpenAI-compatible API:
- **Endpoint**: `http://10.128.0.30:8000/v1`
- **Model**: `google/gemma-3-27b-it`
- **API Key**: `token-abc123` (placeholder)
- **Usage**: Called during reward computation to evaluate response quality using LLM-as-a-judge

### Distributed Setup
While this example runs on a single GPU, the framework supports multi-node, multi-GPU training:
- Ray handles distributed resource management
- FSDP shards model parameters across GPUs
- vLLM supports tensor parallelism for large models

---

---

## Monitoring Training and GPU Performance

Understanding `nvidia-smi` is crucial for optimizing your single-GPU training.

### Key Metrics Explained

#### **1. GPU-Util (The "98%" reading)**
This represents the **execution time** utilization.
- **Definition**: The percentage of time over the past sample period during which one or more kernels were executing on the GPU.
- **Healthy Range**: 80%-100% during the forward/backward pass.
- **Low Utilization (e.g., 20-30%)**: Occurs during data loading, reward computation (waiting for the external 70B verifier API), or FSDP synchronization.

#### **2. Memory-Usage (The "63GB/80GB" reading)**
This represents the **space** utilization.
- **Static Allotment**: vLLM pre-allocates memory for the KV cache (controlled by `gpu_memory_utilization=0.8`).
- **Dynamic Usage**: FSDP manages model weights and gradients.
- **Note**: Memory can stay high while Utilization is low. This is normal.

#### **3. Volatile Uncorr. ECC**
The word "Volatile" in the `nvidia-smi` header often confuses users. 
- In the context of **ECC (Error Correction Code)**, "Volatile" means the error counters are per-boot and will reset if the GPU is reset or the machine reboots.
- It is **not** related to the GPU Utilization percentage directly.

### Identifying Bottlenecks

| **Symptom** | **Likely Cause** | **Solution** |
|-------------|------------------|--------------|
| **Low GPU-Util for long periods** | Waiting for Verifier API | Check network latency to `10.128.0.30` or increase verifier throughput. |
| **GPU-Util spikes to 100% then drops to 0%** | Small batch size / CPU overhead | Increase `ppo_mini_batch_size` or use more rollout samples (`n`). |
| **Target Memory exceeds 80GB (OOM)** | vLLM cache too large | Decrease `rollout.gpu_memory_utilization`. |

### Performance Pro-Tip: `watch` command
To monitor training in real-time without manual commands:
```bash
watch -n 1 nvidia-smi
```
This updates the display every second, allowing you to see the transitions between Rollout (inference) and Actor Update (training).


[def]: file:///home/sriramg/kalashabhayk/TruthRL/training/verl/verl/trainer/main_ppo.py