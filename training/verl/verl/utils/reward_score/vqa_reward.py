
import re
import os
import json
from datetime import datetime

from openai import OpenAI, APIConnectionError, RateLimitError


# ---------------------------------------------------------------------------
# LLM-as-a-Judge: VQA Answer Equivalence Evaluation
# ---------------------------------------------------------------------------
# The judge compares the model's predicted answer (from /box[...]/) against
# the list of acceptable human answers and determines whether the prediction
# is semantically equivalent.
# ---------------------------------------------------------------------------

VQA_JUDGE_INSTRUCTIONS = """You are an expert evaluator for Visual Question Answering tasks.

You will be given:
1. A visual question that was asked about an image.
2. A list of acceptable human-provided ground truth answers.
3. A model-predicted answer.

Your task is to judge whether the predicted answer is semantically equivalent to any of the acceptable answers.

Rules:
- The predicted answer does NOT need to be an exact string match.
- Minor spelling differences, synonyms, or paraphrases of the same concept should be treated as CORRECT.
- Answers that are more specific but still correct (e.g., "golden retriever" when acceptable answers include "dog") should be considered CORRECT.
- Answers that are too vague, wrong, or unrelated should be considered INCORRECT.
- If the answer is a reasonable response to the question but not covered by the acceptable answers, still mark it INCORRECT — we only reward answers consistent with human consensus.

### Output a JSON blob with exactly two fields:
- "verdict": either "CORRECT" or "INCORRECT"
- "explanation": a brief explanation of your judgment (1-2 sentences)

Example output:
{"verdict": "CORRECT", "explanation": "The predicted answer 'carrots' matches several human annotations including 'carrot' and 'carrots'."}
"""

VQA_JUDGE_IN_CONTEXT_EXAMPLES = """Here are examples to guide your judgment:

Example 1 (CORRECT — exact match):
Question: "What color is the car?"
Acceptable answers: ["red", "red", "dark red", "red", "maroon", "red", "red", "dark red", "red", "red"]
Predicted answer: "red"
Output: {"verdict": "CORRECT", "explanation": "The prediction exactly matches the majority human answer."}

Example 2 (CORRECT — synonym / minor variation):
Question: "What type of plant is shown here?"
Acceptable answers: ["carrots, bok choy", "vegetables", "carrot", "carrots", "carrots", "carrot", "vegetable", "vegetables", "carrot", "collard greens"]
Predicted answer: "carrots"
Output: {"verdict": "CORRECT", "explanation": "'carrots' directly matches multiple human annotations."}

Example 3 (CORRECT — semantically equivalent):
Question: "What are the people doing?"
Acceptable answers: ["surfing", "surfing", "surf", "surfing", "water surfing", "riding waves", "surfing", "surfing", "surfing", "surfing"]
Predicted answer: "they are surfing"
Output: {"verdict": "CORRECT", "explanation": "The core concept 'surfing' matches the consensus human answer despite the extra words."}

Example 4 (INCORRECT — wrong answer):
Question: "How many people are in the image?"
Acceptable answers: ["3", "3", "three", "3", "3", "3", "3", "three", "3", "3"]
Predicted answer: "5"
Output: {"verdict": "INCORRECT", "explanation": "The predicted count '5' does not match the human consensus of '3'."}

Example 5 (INCORRECT — related but wrong):
Question: "What animal is shown?"
Acceptable answers: ["cat", "cat", "kitten", "cat", "cat", "cat", "cat", "cat", "kitten", "cat"]
Predicted answer: "dog"
Output: {"verdict": "INCORRECT", "explanation": "The prediction 'dog' is a different animal from the consensus answer 'cat'."}

Example 6 (CORRECT — more specific but valid):
Question: "What is on the table?"
Acceptable answers: ["food", "food", "meal", "food", "dinner", "food", "food", "food", "pizza", "food"]
Predicted answer: "pizza"
Output: {"verdict": "CORRECT", "explanation": "'pizza' is a specific type of food and matches at least one human annotation directly."}
"""


# ---------------------------------------------------------------------------
# OpenAI / vLLM client (module-level, same pattern as vsr_dynamic_consistency.py)
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "empty"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """
    Lightweight text normalization for VQA answers.
    Lowercase, strip whitespace and trailing punctuation.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = s.rstrip(".")
    return s


def log_reward_detail(data):
    """Optional JSONL logging — activate with TRUTHRL_ENABLE_TRAIN_LOGS=1."""
    if os.environ.get("TRUTHRL_ENABLE_TRAIN_LOGS") != "1":
        return

    log_name = os.environ.get("TRUTHRL_LOG_NAME", "training_default")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
    log_dir = os.path.join(project_root, "outputs/reward_logs", log_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"vqa_reward_detail_{os.getpid()}.jsonl")
    data["timestamp"] = datetime.now().isoformat()
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Failed to write VQA reward log: {e}")


def attempt_api_call(messages, max_retries=3):
    """Call the judge LLM with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.environ.get(
                    "VQA_JUDGE_MODEL",
                    "/home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct",
                ),
                messages=messages,
                temperature=0,
                top_p=0.9,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError) as e:
            print(f"[VQAJudge] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                return None
        except Exception as e:
            print(f"[VQAJudge] Unexpected error: {e}")
            return None


def parse_judge_response(response):
    """
    Parse the judge LLM response to extract the verdict.

    Returns:
        (verdict, explanation) where verdict is "CORRECT" / "INCORRECT"
        or (None, None) on parse failure.
    """
    if response is None:
        return None, None

    # Find JSON blob in response
    matches = re.findall(r"\{([^}]*)\}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"

    try:
        verdict_pattern = r'"verdict"\s*:\s*"([^"]+)"'
        verdict_match = re.search(verdict_pattern, text)
        if verdict_match:
            verdict = verdict_match.group(1).upper().strip()
        else:
            return None, None

        if verdict not in ("CORRECT", "INCORRECT"):
            print(f"[VQAJudge] Invalid verdict '{verdict}' in response: {response}")
            return None, None

        explanation_pattern = r'"explanation"\s*:\s*"([^"]*)"'
        explanation_match = re.search(explanation_pattern, text)
        explanation = explanation_match.group(1) if explanation_match else text

        return verdict, explanation

    except Exception as e:
        print(f"[VQAJudge] Parsing error: {e}, response: {response}")
        return None, None


def get_vqa_judge_verdict(question, acceptable_answers, predicted_answer):
    """
    Call the judge LLM to evaluate whether the predicted answer is
    semantically equivalent to any acceptable human answer.

    Args:
        question: The original visual question.
        acceptable_answers: List of human-annotated answers.
        predicted_answer: The model's extracted answer string.

    Returns:
        (verdict, explanation)
        - verdict: "CORRECT", "INCORRECT", or None on failure
        - explanation: The judge's reasoning string
    """
    system_message = VQA_JUDGE_INSTRUCTIONS + "\n" + VQA_JUDGE_IN_CONTEXT_EXAMPLES

    # De-duplicate answers for a cleaner judge prompt
    unique_answers = list(dict.fromkeys(acceptable_answers))

    user_content = (
        f'Question: "{question}"\n'
        f'Acceptable answers: {json.dumps(unique_answers)}\n'
        f'Predicted answer: "{predicted_answer}"\n'
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_content},
    ]

    llm_response = attempt_api_call(messages)
    if llm_response is None:
        return None, "API call failed"

    verdict, explanation = parse_judge_response(llm_response)
    return verdict, explanation


def compute_vqa_accuracy(predicted, acceptable_answers):
    """
    Standard VQAv2 soft accuracy metric as a fast fallback.
    accuracy = min(1.0, num_exact_matches / 3)

    Used when the LLM judge is unavailable (API failure).
    """
    pred_norm = normalize_answer(predicted)
    match_count = sum(
        1 for ans in acceptable_answers
        if normalize_answer(ans) == pred_norm
    )
    return min(1.0, match_count / 3.0)


# ---------------------------------------------------------------------------
# Main reward function — entry point called by VERL
# ---------------------------------------------------------------------------

def compute_score(solution_str, ground_truth, method="strict",
                  format_score=-1.0, score=1.0, **kwargs):
    """
    VQA LLM-as-Judge Reward Function.

    Reward flow:
        1. Extract the model's answer from /box[...]/.
           Missing format → format_score penalty (default -1.0).

        2. Check for abstention ("I don't know" and variants).
           Abstention → 0.0 reward (safe refusal, better than hallucinating).

        3. Quick-check: compute the standard VQA soft accuracy
           (min(1, matches/3)) as a baseline.

        4. Call the LLM judge to determine semantic equivalence
           between the predicted answer and the list of acceptable
           human answers.

        5. Assign reward:
           - Abstention          →  0.0  (safe refusal)
           - CORRECT verdict     →  +score  (default +1.0)
           - INCORRECT verdict   →  -1.0 (hallucination penalty)
           - Judge failure        →  fall back to soft accuracy
             (returns the continuous VQAv2 accuracy as reward)

    Args:
        solution_str:  The model's full generated output string.
        ground_truth:  Dict from the parquet reward_model column:
                       {
                         "acceptable_answers": [...],
                         "multiple_choice_answer": "...",
                         "style": "vqa_llm_judge"
                       }
        format_score:  Penalty for missing /box[]/ format.
        score:         Reward for a correct answer.

    Returns:
        dict with keys: score, accuracy, judge_verdict, judge_explanation
    """

    # -- 1. Extract acceptable answers from ground_truth --
    if isinstance(ground_truth, dict):
        acceptable_answers = ground_truth.get("acceptable_answers", [])
        mc_answer = ground_truth.get("multiple_choice_answer", "")
    else:
        # Shouldn't happen with properly formatted parquet, but be safe
        acceptable_answers = [str(ground_truth)]
        mc_answer = str(ground_truth)

    # -- 2. Check format & extract answer from /box[...]/ --
    box_match = re.search(r'/box\[(.*?)\]/', solution_str)
    if box_match:
        predicted_answer = box_match.group(1)
    else:
        # Penalise missing format
        penalty = format_score if format_score < 0 else -1.0

        log_reward_detail({
            "event": "format_missing",
            "score": penalty,
            "solution_preview": solution_str[:200],
        })

        return {
            "score": penalty,
            "accuracy": 0.0,
            "judge_verdict": None,
            "judge_explanation": "Format /box[]/ not found",
        }

    # -- 3. Check for abstention ("I don't know" — safe refusal) --
    pred_normalized = normalize_answer(predicted_answer)
    unknown_triggers = ["i don't know", "i dont know", "unsure", "i do not know",
                        "not sure", "cannot determine", "can't determine",
                        "unable to determine", "cannot tell", "can't tell"]
    if any(trigger in pred_normalized for trigger in unknown_triggers):
        log_reward_detail({
            "event": "abstention",
            "predicted_answer": predicted_answer,
            "score": 0.0,
            "solution_preview": solution_str[:200],
        })
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "judge_verdict": "ABSTENTION",
            "judge_explanation": "Model chose to abstain rather than hallucinate.",
        }

    # -- 4. Quick VQA soft accuracy (used as fallback & for logging) --
    soft_accuracy = compute_vqa_accuracy(predicted_answer, acceptable_answers)

    # -- 5. Extract the question from prompt (passed via extra_info) --
    extra_info = kwargs.get("extra_info", {})
    prompt_text = ""
    if isinstance(extra_info, dict):
        prompt_text = extra_info.get("prompt_text", "")
    # Strip <image> tag prefix to get the raw question
    question_for_judge = prompt_text
    if question_for_judge and "\n" in question_for_judge:
        question_for_judge = question_for_judge.split("\n")[-1].strip()
    question_for_judge = re.sub(r'<image>\s*', '', question_for_judge).strip()
    if not question_for_judge:
        # Fallback: try to reconstruct from acceptable answers context
        question_for_judge = "(question unavailable)"

    # -- 6. Call LLM Judge --
    verdict, explanation = get_vqa_judge_verdict(
        question_for_judge, acceptable_answers, predicted_answer
    )

    # -- 7. Assign reward --
    if verdict == "CORRECT":
        final_score = score  # default +1.0
        accuracy = 1.0
    elif verdict == "INCORRECT":
        final_score = -1.0
        accuracy = 0.0
    else:
        # Judge failed — fall back to soft accuracy as a continuous reward
        # This gives partial credit proportional to human agreement
        # Maps [0, 1] soft_accuracy → [-1, +1] reward range
        final_score = (soft_accuracy * 2.0) - 1.0
        accuracy = soft_accuracy
        explanation = f"Judge unavailable, fell back to soft accuracy: {soft_accuracy:.2f}"

    # -- 8. Logging --
    log_reward_detail({
        "event": "scored",
        "predicted_answer": predicted_answer,
        "acceptable_answers": acceptable_answers[:5],  # truncate for log size
        "mc_answer": mc_answer,
        "soft_accuracy": soft_accuracy,
        "judge_verdict": verdict,
        "judge_explanation": explanation,
        "final_score": final_score,
        "question": question_for_judge,
    })

    return {
        "score": final_score,
        "accuracy": accuracy,
        "judge_verdict": verdict,
        "judge_explanation": explanation,
    }
