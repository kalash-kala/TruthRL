
import sys
import os

# Add the directory containing vsr_lexical to sys.path
sys.path.append('/home/kalashkala/TruthRL/training/verl/verl/utils/reward_score')

from vsr_lexical import compute_score

def test_reward():
    test_cases = [
        # (prediction, ground_truth, expected_reward)
        ("True", "True", 1.0),
        ("False", "True", -1.0),
        ("True.", "True", 1.0), # Normalization check
        (" true ", "True", 1.0), # Normalization check
        ("I don't know", "True", 0.0), # Refusal check
        ("I'm not sure", "True", -1.0), # Not in trigger list yet (as per current code)
        ("I dont know", "True", 0.0), # Refusal check
        ("Random output", "False", -1.0),
        ("False", "False", 1.0),
        ({"ground_truth": "True"}, "True", 1.0), # This case is unlikely based on how Verl calls it, but let's test the dict handling in my code
    ]

    print("Running reward function tests...")
    for i, (pred, gt, expected) in enumerate(test_cases):
        # Note: In Verl, ground_truth is often passed as the target value or a dict
        reward = compute_score(pred, gt)
        status = "PASSED" if reward == expected else "FAILED"
        print(f"Test {i}: Pred='{pred}', GT='{gt}', Expected={expected}, Got={reward} -> {status}")
        if reward != expected:
             print(f"  Error: {pred} vs {gt}")

if __name__ == "__main__":
    test_reward()
