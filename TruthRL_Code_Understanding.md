# TruthRL Code Understanding

This document serves as a guide for the data flow and column utilization within the TruthRL codebase. It covers how data is read and processed during both training and evaluation phases.

## Table of Contents
1. [Data Loading Entry Points](#1-data-loading-entry-points)
2. [Dataset Columns & Their Roles](#2-dataset-columns--their-roles)
3. [Important Framework Components](#3-important-framework-components)
4. [Evaluation Metrics Explained](#4-evaluation-metrics-explained)
5. [Data Filtering & Pruning](#5-data-filtering--pruning)
6. [Reward Scoring Mechanism](#6-reward-scoring-mechanism)

---


## 1. Data Loading Entry Points

The data reading logic is split between training and evaluation, following different paths based on the objective.

### Training Data (GRPO/PPO)
Training data is read from local **Parquet** files via the `verl` framework.
- **Entry Point:** `train_grpo.sh` passes paths via `data.train_files` and `data.val_files`.
- **Loader Class:** `RLHFDataset` in `training/verl/verl/utils/dataset/rl_dataset.py`.
- **Logic:** The `_read_files_and_tokenize` method uses the Hugging Face `datasets` library to load and concatenate the Parquet files.

### Evaluation Data
Evaluation data is typically fetched from the Hugging Face Hub.
- **Entry Point:** `evaluation/evaluate.py`
- **Logic:** The `generate_results` function loads the dataset using `datasets.load_dataset(f'weizhepei/TruthRL-{dataset_name}', split=split)`.

---

## 2. Dataset Columns & Their Roles

The dataset contains various columns, but only a subset is "active" in the core logic. Below is a breakdown of how the Parquet columns are utilized.

### Column Mapping Table

| Column Name | Training Usage | Evaluation Usage | Role |
| :--- | :---: | :---: | :--- |
| **`prompt`** | ✅ (Primary) | ❌ | Message history used to build model tokens. |
| **`query`** | ❌ | ✅ | The question text used in the evaluation prompt. |
| **`answer`** / **`target`** | ✅ (via `reward_model`) | ✅ | Ground truth for correctness comparison. |
| **`retrieved_chunks`** | ❌ | ✅ | Context snippets injected for RAG. |
| **`query_time`** | ❌ | ✅ | Temporal context (current time) for the model. |
| **`out_of_knowledge`** | ✅ (via `reward_model`) | ❌ | Boolean flag triggering "I don't know" logic. |
| **`data_source`** | ✅ | ❌ | Routes to the correct Reward Function (e.g., `TruthRL_crag`). |
| **`reward_model`** | ✅ | ❌ | A JSON blob containing `target`, `problem`, and `out_of_knowledge`. |
| **`extra_info`** | ✅ | ❌ | Carries metadata like `interaction_id` to reward functions. |
| **`interaction_id`** | ✅ (via `extra_info`) | ✅ | Unique identifier used for logging and result tracking. |
| **`domain`** | ❌ | ✅ | Used for categorized performance analysis. |
| **`question_type`** | ❌ | ✅ | Used for categorized performance analysis. |
| **`static_or_dynamic`** | ❌ | ✅ | Used for categorized performance analysis. |
| **`completion`** | ❌ | ❌ | Usually ignored (used in SFT, not RL/PPO). |

---

## 3. Important Framework Components

### NaiveRewardManager
- **Located in:** `training/verl/verl/workers/reward_manager/naive.py`
- **Role:** Orchestrates the reward calculation. It decodes the model's response and calls the `truthrl_qa` functions using data from the `reward_model` and `extra_info` columns.

### truthrl_qa.py
- **Located in:** `training/verl/verl/utils/reward_score/truthrl_qa.py`
- **Role:** Contains the actual scoring logic, including:
  - **LLM-as-a-judge:** Using a verifier model (like Gemma-3) to grade reasoning.
  - **Exact Match:** Conventional string matching for final answers.
  - **OOK Logic:** Special rewards for correctly identifying out-of-knowledge questions.

### evaluate.py
- **Located in:** `evaluation/evaluate.py`
- **Role:** The testing harness. It builds RAG prompts using `retrieved_chunks`, generates responses using `InstructModel`, and calculates final metrics (Accuracy, Hallucination rate, etc.) based on the `answer` and `alt_ans` columns.

---

## 4. Evaluation Metrics Explained

The evaluation process (run via `evaluate.py`) produces a JSON result containing several key metrics. These are calculated categorized by the model's response and the verifier's judgment.

### Core Metrics Table

| Metric | Definition | Calculation Logic |
| :--- | :--- | :--- |
| **`score`** | Overall performance score. | `(2 * n_correct + n_miss - total) / total` |
| **`accuracy`** | General correctness. | `n_correct / total` |
| **`em`** | Exact Match rate. | `n_exact_match / total` |
| **`missing`** | "I don't know" rate. | `n_miss / total` |
| **`hallucination (incl. no boxed)`**| Total failure rate. | `(total - n_correct - n_miss) / total` |
| **`hallucination (excl. no boxed)`**| Factual error rate. | `(total - n_correct - n_miss - n_no_boxed) / total` |

### Understanding the Scoring Logic
- **Correct (`n_correct`)**: Either an exact match to the ground truth or validated by the LLM-as-a-judge.
- **Miss (`n_miss`)**: When the model explicitly states it doesn't know (e.g., "I don't know").
- **No Boxed (`n_no_boxed`)**: When the model fails to follow formatting rules (i.e., answer is not in `\boxed{}`).
- **Hallucination**: Any boxed response that is factually incorrect.

---

## 5. Data Filtering & Pruning

During training, the dataset is dynamically pruned based on token length to ensure stability and fit within the defined context window.

### Where Pruning Happens
- **Location:** `RLHFDataset.maybe_filter_out_long_prompts` in `training/verl/verl/utils/dataset/rl_dataset.py`.
- **Trigger:** This runs during the initialization of the data loader before training steps begin.

### Key Factors
1. **Context Length Check:** The code applies the model's chat template to the `prompt` and counts the resulting tokens.
2. **Filtering:** Any sample where the token count exceeds `data.max_prompt_length` (defined in `train_grpo.sh`) is removed from the dataset.
3. **Truncation Strategy:** If `data.truncation` is set to `'error'`, any sample that happens to exceed the limit during processing will crash the training to prevent silent data corruption.

### Logging
You can observe this in the logs by looking for the line:
`Filtering prompts longer than XXXX tokens...` followed by `filter dataset len: XXX`.

---

## 6. Reward Scoring Mechanism

The reward scoring logic in `truthrl_qa.py` is the core of the RL training (GRPO/PPO). It uses a multi-stage approach to promote honesty and factual accuracy.

### Ternary Outcome Scoring
Specifically implemented in `compute_score_llm_as_a_judge_ternary_OOK`, the reward follows a **+1/0/-1** schema:

| Response Type | Reward | Condition |
| :--- | :---: | :--- |
| **Correct** | **+1** | Validated by Exact Match or LLM-as-a-judge. |
| **Correct "IDK"** | **+1** | Model says "I don't know" to an **Out-of-Knowledge (OOK)** question. |
| **Neutral "IDK"** | **0** | Model says "I don't know" to a **known** question (treated as a "miss"). |
| **Hallucination** | **-1** | Factually incorrect answer or answering an OOK question. |
| **Format Error** | **-1** | Answer is missing the `\boxed{}` tags. |

### Process & Reasoning Scoring (PRM)
The system also evaluates the model's internal "thought process" (the content in `<think>` tags) via `compute_process_score`.

1.  **Direct Judging**: The full reasoning string is sent to the LLM-as-a-judge.
2.  **Usefulness Check**: The judge evaluates if the reasoning is **precise, logical, and evidence-based**.
3.  **Scoring**:
    *   **1**: High-quality reasoning that directly supports the answer.
    *   **0**: Vague, unrelated, or "unsubstantiated conclusions" (e.g., "I guess it's around 20...").
4.  **Integration**: In the current implementation (`__init__.py`), the process reward is dynamically calculated but can be gated by the outcome reward (e.g., only rewarding reasoning if the final answer is correct).

### The Verifier (LLM-as-a-judge)
- **Model**: Traditionally uses a high-parameter model (e.g., `Gemma-3-27B-IT`) hosted on an OpenAI-compatible API.
- **Rules**: The judge uses a complex system message with multiple in-context examples to handle rounding, abbreviations, and set-matching (e.g., "William Shakespeare" vs "W. Shakespeare").

