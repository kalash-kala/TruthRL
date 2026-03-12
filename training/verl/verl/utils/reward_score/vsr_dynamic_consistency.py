
import re
import os
import json
import threading
from datetime import datetime

from openai import OpenAI, APIConnectionError, RateLimitError

# ---------------------------------------------------------------------------
# LLM-as-a-Judge: Reasoning-Answer Consistency Evaluation
# ---------------------------------------------------------------------------
# This prompt asks the judge LLM to compare the reasoning trace
# (<reasoning start> ... <reasoning end>) with the final answer
# (/box[...]/) and classify how well they align.
# ---------------------------------------------------------------------------

CONSISTENCY_JUDGE_INSTRUCTIONS = """You are an expert evaluator. You will be given a model-generated response that contains:
1. A reasoning section enclosed between <reasoning start> and <reasoning end> tags.
2. A final answer enclosed in /box[...]/ tags.

Your task is to evaluate how well the reasoning supports the final answer. Classify the relationship into EXACTLY ONE of the following four labels:

- "strongly supported": The reasoning clearly, logically, and directly leads to the final answer. The answer is a natural conclusion of the reasoning chain.
- "weakly supported": The reasoning is somewhat related to the final answer but has gaps, jumps in logic, or only partially justifies the answer.
- "unsupported": The reasoning does not connect to the final answer. The answer appears to be a random guess that ignores the reasoning entirely.
- "contradictory": The reasoning actively contradicts the final answer. The reasoning leads to one conclusion but the answer states something different.

Additionally, check for "lazy refusals" — cases where the model's reasoning reaches a clear conclusion but the answer is a refusal (e.g., "I don't know") instead of the conclusion the reasoning supports. Treat lazy refusals as "contradictory", because the model's own reasoning contradicts its decision to abstain.

### Output a JSON blob with:
- "label": one of "strongly supported", "weakly supported", "unsupported", "contradictory"
- "explanation": a brief explanation of your judgment (1-2 sentences)

Example output:
{"label": "strongly supported", "explanation": "The reasoning systematically evaluates the visual relationship and concludes True, matching the final answer."}
"""

CONSISTENCY_IN_CONTEXT_EXAMPLES = """Here are some examples to guide your judgment:

Example 1 (Strongly Supported — correct True):
Caption: "The banana is at the left side of the orange."
Reasoning: "The banana is positioned to the left of the orange, with its curved shape and yellow color clearly visible."
Answer: "True"
Output: {"label": "strongly supported", "explanation": "The reasoning directly identifies the spatial relationship stated in the caption and the answer follows logically."}

Example 2 (Strongly Supported — correct False):
Caption: "The dog is inside the car."
Reasoning: "The image shows a dog standing on a skateboard in an outdoor setting, not inside a car."
Answer: "False"
Output: {"label": "strongly supported", "explanation": "The reasoning clearly refutes the caption's claim with concrete visual evidence, and the False answer is the natural conclusion."}

Example 3 (Weakly Supported — proxy-object reasoning):
Caption: "The pizza is in the dog."
Reasoning: "The picture shows a puppy holding a toy that looks like a slice of pizza. The toy is in the dog's mouth."
Answer: "True"
Output: {"label": "weakly supported", "explanation": "The reasoning reaches the answer through a proxy object ('a toy that looks like a slice of pizza') rather than directly verifying the stated object, making the support plausible but not fully decisive."}

Example 4 (Weakly Supported — underspecified reasoning):
Caption: "The cake consists of the car."
Reasoning: "The picture shows a cake that looks like a car."
Answer: "True"
Output: {"label": "weakly supported", "explanation": "The reasoning is extremely short and only states appearance without explicitly explaining the composition relation in the caption."}

Example 5 (Unsupported — wrong entity pair):
Caption: "The car is far from the chair."
Reasoning: "The picture shows a dog in a car looking out the window. The dog is not near the chair."
Answer: "False"
Output: {"label": "unsupported", "explanation": "The reasoning shifts to the dog-chair relation instead of evaluating the car-chair relation asked in the caption, so the final answer is not supported."}

Example 6 (Unsupported — insufficient evidence):
Caption: "The person is touching the sandwich."
Reasoning: "The picture contains a sandwich cut in half and a hand in the background."
Answer: "False"
Output: {"label": "unsupported", "explanation": "The reasoning mentions the presence of a hand but provides no direct evidence about whether the hand is touching the sandwich."}

Example 7 (Contradictory — reasoning leads to opposite answer):
Caption: "The keyboard is on top of the cat."
Reasoning: "The image shows a cat lying on a computer keyboard. The keyboard is positioned directly under the cat's body, with its keys visible beneath the cat's paws. This arrangement suggests that the keyboard is resting on top of the cat."
Answer: "False"
Output: {"label": "contradictory", "explanation": "The reasoning concludes the keyboard is resting on top of the cat (True), but the final answer contradicts this by stating False."}

Example 8 (Contradictory — reasoning supports True but answer is False):
Caption: "The teddy bear is touching the bed."
Reasoning: "The teddy bear is positioned among the books on the bed, with its head resting on one of the books. The bed's surface is visible around the teddy bear, indicating it is indeed touching the bed."
Answer: "False"
Output: {"label": "contradictory", "explanation": "The reasoning asserts the teddy bear 'is indeed touching the bed' (True), but the answer states False, directly contradicting the reasoning."}

Example 9 (Contradictory — lazy refusal despite clear reasoning):
Caption: "The bowl is under the sandwich."
Reasoning: "To determine if the bowl is under the sandwich, we need to analyze the positions of the objects in the image. The bowl appears to be placed next to the sandwich on the tray. The sandwich is positioned on top of the paper towel, which is on top of the tray. Therefore, the bowl is not directly under the sandwich; it is adjacent to it."
Answer: "I don't know"
Output: {"label": "contradictory", "explanation": "The reasoning reaches a clear conclusion ('the bowl is not directly under the sandwich') but the model refuses to commit to an answer, contradicting its own definitive reasoning."}

Example 10 (Contradictory — lazy refusal with thorough analysis):
Caption: "The pizza is in the dog."
Reasoning: "The statement claims that the pizza is in the dog. However, the image shows a puppy holding a toy shaped like a slice of pizza, not the actual pizza itself. The puppy is not consuming the toy; it's merely holding it. Therefore, the pizza is not physically inside the dog."
Answer: "I don't know"
Output: {"label": "contradictory", "explanation": "The reasoning thoroughly evaluates the scene and reaches a strong conclusion, yet the model abstains instead of providing the answer its own reasoning supports — directly contradicting the reasoning."}
"""


# ---------------------------------------------------------------------------
# Label-to-reward mapping
# ---------------------------------------------------------------------------
LABEL_TO_REWARD = {
    "strongly supported": 1.0,
    "weakly supported": 0.5,
    "unsupported": -0.5,
    "contradictory": -1.0,
}


# ---------------------------------------------------------------------------
# OpenAI client (same pattern as truthrl_qa.py)
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """
    Normalize the answer by lowercasing and stripping whitespace/punctuation.
    Simple normalization for "True"/"False" string matching.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = s.rstrip(".")
    return s


def log_reward_detail(data):
    """Logs detailed reward information to a JSONL file. Only active if enabled."""
    if os.environ.get("TRUTHRL_ENABLE_TRAIN_LOGS") != "1":
        return

    log_name = os.environ.get("TRUTHRL_LOG_NAME", "training_default")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
    log_dir = os.path.join(project_root, "outputs/reward_logs", log_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"consistency_reward_detail_{os.getpid()}.jsonl")

    data["timestamp"] = datetime.now().isoformat()
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Failed to write consistency reward log: {e}")


def attempt_api_call(messages, max_retries=3):
    """Call the judge LLM with retries (same pattern as truthrl_qa.py)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.environ.get("CONSISTENCY_JUDGE_MODEL", "google/gemma-3-27b-it"),
                messages=messages,
                temperature=0,
                top_p=0.9,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError) as e:
            print(f"[ConsistencyJudge] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                return None
        except Exception as e:
            print(f"[ConsistencyJudge] Unexpected error: {e}")
            return None


def parse_consistency_response(response: str):
    """
    Parse the judge LLM response to extract the consistency label.

    Returns:
        (label, explanation) where label is one of the four valid labels
        or None if parsing fails.
    """
    if response is None:
        return None, None

    # Find JSON blob in response
    matches = re.findall(r"\{([^}]*)\}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"

    try:
        # Extract label
        label_pattern = r'"label"\s*:\s*"([^"]+)"'
        label_match = re.search(label_pattern, text)
        if label_match:
            label = label_match.group(1).lower().strip()
        else:
            return None, None

        # Validate label
        if label not in LABEL_TO_REWARD:
            print(f"[ConsistencyJudge] Invalid label '{label}' in response: {response}")
            return None, None

        # Extract explanation
        explanation_pattern = r'"explanation"\s*:\s*"([^"]*)"'
        explanation_match = re.search(explanation_pattern, text)
        explanation = explanation_match.group(1) if explanation_match else text

        return label, explanation

    except Exception as e:
        print(f"[ConsistencyJudge] Parsing error: {e}, response: {response}")
        return None, None


def extract_reasoning(solution_str):
    """
    Extract the reasoning section from the model output.
    Looks for text between <reasoning start> and <reasoning end> tags.
    """
    pattern = r"<reasoning start>(.*?)<reasoning end>"
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_consistency_judge_reward(reasoning_text, answer_text, caption_text=None):
    """
    Call the judge LLM to get a consistency label between reasoning and answer.

    Args:
        reasoning_text: The model's reasoning trace.
        answer_text: The model's final answer.
        caption_text: (optional) The original caption/statement being evaluated.

    Returns:
        (judge_reward, label, explanation)
        - judge_reward: float from LABEL_TO_REWARD (defaults to 0.33 on failure)
        - label: the string label or None
        - explanation: the judge's explanation or None
    """
    system_message = CONSISTENCY_JUDGE_INSTRUCTIONS + "\n" + CONSISTENCY_IN_CONTEXT_EXAMPLES

    # Build the user content — include caption if available
    if caption_text:
        user_content = (
            f"Caption: \"{caption_text}\"\n"
            f"Reasoning: \"{reasoning_text}\"\n"
            f"Answer: \"{answer_text}\"\n"
        )
    else:
        # Fallback: no caption available (should not happen in normal VSR flow)
        user_content = (
            f"Reasoning: \"{reasoning_text}\"\n"
            f"Answer: \"{answer_text}\"\n"
        )

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": user_content,
        },
    ]

    llm_response = attempt_api_call(messages)
    if llm_response is None:
        # API failure — return neutral default
        return 0.33, None, "API call failed"

    label, explanation = parse_consistency_response(llm_response)
    if label is None:
        return 0.33, None, f"Parse failure: {llm_response}"

    judge_reward = LABEL_TO_REWARD[label]
    return judge_reward, label, explanation


# ---------------------------------------------------------------------------
# Module-level step tracker (identical to vsr_lexical_dynamic.py)
# ---------------------------------------------------------------------------

class _StepTracker:
    """Thread-safe, module-level tracker for the current training step."""

    def __init__(self):
        self._lock = threading.Lock()
        self._call_count = 0
        self._current_step = 0
        self._batch_size = None
        self._total_steps = None

    def get_progress(self, total_steps_kwarg=None, batch_size_kwarg=None):
        """Return (current_step, total_steps_per_epoch) as ints."""
        if total_steps_kwarg is not None:
            total_steps = int(total_steps_kwarg)
        elif os.environ.get("VERL_TOTAL_STEPS_PER_EPOCH"):
            total_steps = int(os.environ["VERL_TOTAL_STEPS_PER_EPOCH"])
        else:
            total_steps = 250

        env_step = os.environ.get("VERL_GLOBAL_STEP")
        if env_step is not None:
            current_step = int(env_step)
        else:
            bs = batch_size_kwarg or self._batch_size or 8
            with self._lock:
                current_step = self._call_count // int(bs)

        return current_step, total_steps

    def tick(self):
        """Increment the internal call counter (one call = one sample)."""
        with self._lock:
            self._call_count += 1


_step_tracker = _StepTracker()


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_score(solution_str, ground_truth, method="strict", format_score=-1.0,
                  score=1.0, **kwargs):
    """
    VSR Dynamic Consistency Reward Function.

    This reward function combines the base lexical reward from
    vsr_lexical_dynamic with an LLM-as-a-judge consistency bonus/penalty
    that evaluates whether the model's reasoning actually supports its
    final answer.

    The workflow is:
        1. Compute the base reward (same logic as vsr_lexical_dynamic):
           - Correct answer (True/False matches ground truth) → +score (default 1.0)
           - Abstention ("I don't know")                      → 0.0
           - Incorrect / hallucinated                         → dynamic negative
           - Missing /box[]/ format                           → format_score

        2. Extract the reasoning from <reasoning start>...<reasoning end>.

        3. Call an external LLM judge to compare reasoning vs. final answer.
           The judge returns one of four labels:
               "strongly supported" → judge_reward = 1.0
               "weakly supported"   → judge_reward = 0.5
               "unsupported"        → judge_reward = -0.5
               "contradictory"      → judge_reward = -1.0

        4. Apply consistency adjustment to the base reward:
           - For True/False answers (correct OR incorrect):
                 adjustment = +lambda_factor * judge_reward
                 This *rewards* answers whose reasoning supports them
                 and *penalises* answers with contradictory reasoning.
                 If reasoning is "strongly supported" (judge_reward=1.0)
                 the full +lambda_factor bonus is applied.  If
                 "contradictory" (judge_reward=-1.0) the full
                 -lambda_factor penalty is applied.

           - For abstention answers ("I don't know"):
                 adjustment = +mu_factor * judge_reward
                 This *rewards* well-reasoned abstentions.  If the model's
                 reasoning genuinely supports the decision to abstain
                 (judge_reward=1) it gets the full +mu_factor bonus.

        5. Final reward = base_reward + adjustment

    Configurable kwargs:
        total_steps_per_epoch (int)   - for dynamic negative reward scheduling
        batch_size (int)              - for fallback step counter
        lambda_factor (float)         - weight of consistency penalty for
                                        True/False answers  (default 0.5)
        mu_factor (float)             - weight of consistency bonus for
                                        abstention answers   (default 0.5)
    """

    # -- Tick the module-level counter --
    _step_tracker.tick()

    # -- Resolve step progress --
    total_steps_kwarg = kwargs.get("total_steps_per_epoch", None)
    batch_size_kwarg = kwargs.get("batch_size", None)
    current_step, total_steps = _step_tracker.get_progress(
        total_steps_kwarg=total_steps_kwarg,
        batch_size_kwarg=batch_size_kwarg,
    )

    # -- Configurable consistency weights --
    lambda_factor = float(kwargs.get("lambda_factor", 0.5))
    mu_factor = float(kwargs.get("mu_factor", 0.5))

    # -- Dynamic negative reward (same as vsr_lexical_dynamic) --
    progress_ratio = current_step / max(total_steps, 1)
    negative_reward = -1.0 * (1.0 + progress_ratio)

    # -- 1. Extract Ground Truth --
    target = ground_truth
    if isinstance(ground_truth, dict):
        target = ground_truth.get("ground_truth", "False")

    # -- 2. Check Format & Extract Answer --
    box_match = re.search(r'/box\[(.*?)\]/', solution_str)
    if box_match:
        answer_to_check = box_match.group(1)
    else:
        # Penalise for missing /box[]/ format
        penalty = format_score if format_score < 0 else negative_reward

        # Debug logging
        log_path = "/home/debarpanb1/kalashkala/TruthRL/58-Cluster-scripts/consistency_reward_debug.log"
        if not os.path.exists(os.path.dirname(log_path)):
            log_path = "/home/kalashkala/TruthRL/58-Cluster-scripts/consistency_reward_debug.log"
        try:
            if os.path.exists(os.path.dirname(log_path)):
                with open(log_path, "a") as f:
                    f.write(
                        f"[CONSISTENCY REWARD - FORMAT MISSING] "
                        f"step: {current_step}/{total_steps} | penalty: {penalty:.4f}\n"
                    )
        except Exception:
            pass

        return {
            'score': penalty,
            'accuracy': 0.0,
            'negative_reward': negative_reward,
            'consistency_label': None,
            'consistency_reward': None,
            'consistency_adjustment': 0.0,
        }

    # -- 3. Normalize Prediction --
    pred_str = normalize_answer(answer_to_check)
    target_str = normalize_answer(target)

    # -- 4. Determine the base reward and answer type --
    unknown_triggers = ["i don't know", "i dont know", "unsure", "i do not know"]
    is_abstention = any(trigger in pred_str for trigger in unknown_triggers)
    is_correct = (pred_str == target_str)

    if is_abstention:
        base_reward = 0.0
        answer_type = "abstention"
    elif is_correct:
        base_reward = score  # default +1.0
        answer_type = "correct"
    else:
        base_reward = negative_reward  # dynamic negative
        answer_type = "incorrect"

    # -- 5. Extract reasoning and get consistency judgment --
    reasoning_text = extract_reasoning(solution_str)
    consistency_label = None
    consistency_explanation = None
    judge_reward = None
    consistency_adjustment = 0.0

    # -- Extract caption from prompt text (injected by NaiveRewardManager) --
    caption_text = None
    extra_info = kwargs.get("extra_info", {})
    prompt_text = extra_info.get("prompt_text", "") if isinstance(extra_info, dict) else ""
    if prompt_text:
        # The prompt user message looks like: "<image>\nThe car is under the surfboard."
        # Strip the <image> tag prefix to get the raw caption.
        caption_text = prompt_text
        if "\n" in caption_text:
            # Take the last line of the user content as the caption
            caption_text = caption_text.split("\n")[-1].strip()
        # Remove any remaining image tags
        caption_text = re.sub(r'<image>\s*', '', caption_text).strip()
        if not caption_text:0
            caption_text = None

    if reasoning_text is not None:
        # judge_reward, consistency_label, consistency_explanation = \
        #     get_consistency_judge_reward(reasoning_text, answer_to_check)
        judge_reward, consistency_label, consistency_explanation = \
            get_consistency_judge_reward(reasoning_text, answer_to_check, caption_text=caption_text)

        if judge_reward is not None:
            if is_abstention:
                # For abstention: reward well-reasoned abstentions
                # adjustment = +mu_factor * judge_reward
                consistency_adjustment = mu_factor * judge_reward
            else:
                # For True/False answers (correct or incorrect):
                # adjustment = +lambda_factor * judge_reward
                consistency_adjustment = lambda_factor * judge_reward

    # -- 6. Compute final reward --
    final_reward = base_reward + consistency_adjustment

    # -- 7. Logging --
    log_path = "/home/debarpanb1/kalashkala/TruthRL/58-Cluster-scripts/consistency_reward_debug.log"
    if not os.path.exists(os.path.dirname(log_path)):
        log_path = "/home/kalashkala/TruthRL/58-Cluster-scripts/consistency_reward_debug.log"
    try:
        if os.path.exists(os.path.dirname(log_path)):
            with open(log_path, "a") as f:
                f.write(
                    f"[CONSISTENCY REWARD] "
                    f"step: {current_step}/{total_steps} | "
                    f"type: {answer_type} | "
                    f"base: {base_reward:.4f} | "
                    f"judge: {judge_reward} ({consistency_label}) | "
                    f"adj: {consistency_adjustment:.4f} | "
                    f"final: {final_reward:.4f}\n"
                )
    except Exception:
        pass

    log_reward_detail({
        "answer_type": answer_type,
        "prediction": answer_to_check,
        "ground_truth": target,
        "base_reward": base_reward,
        "consistency_label": consistency_label,
        "consistency_explanation": consistency_explanation,
        "judge_reward": judge_reward,
        "lambda_factor": lambda_factor,
        "mu_factor": mu_factor,
        "consistency_adjustment": consistency_adjustment,
        "final_reward": final_reward,
        "step": current_step,
        "total_steps": total_steps,
        "progress_ratio": progress_ratio,
        "negative_reward": negative_reward,
        "reasoning_present": reasoning_text is not None,
    })

    return {
        'score': final_reward,
        'accuracy': 1.0 if is_correct else 0.0,
        'negative_reward': negative_reward,
        'consistency_label': consistency_label,
        'consistency_reward': judge_reward,
        'consistency_adjustment': consistency_adjustment,
    }
