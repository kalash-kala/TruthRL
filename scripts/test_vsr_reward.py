
import sys
import os

# Add the package root to sys.path
sys.path.append('/home/sriramg/kalashabhayk/TruthRL/training/verl')

from verl.utils.reward_score.vsr_lexical import compute_score

def test_reward():
    test_cases = [
        # (prediction, ground_truth, expected_reward)
        ("<reasoning start> The statement correctly describes the spatial relationship. <reasoning end> /box[True]/", "True", 1.0),
        ("<reasoning start> The object is on the right, not left. <reasoning end> /box[False]/", "True", -1.0),
        ("/box[True]/", "True", 1.0), 
        ("/box[False]/", "False", 1.0),
        ("<reasoning start> Based on my analysis, I am unable to confirm the spatial relationship. <reasoning end> /box[I don't know]/", "True", 0.0),
        ("True", "True", 1.0), # Backward compatibility check
        ("I don't know", "True", 0.0), # Backward compatibility check
        ("Gibberish without box", "True", -1.0),
        ("<reasoning start> Reasoning but no box <reasoning end>", "True", -1.0),
    ]

    print("Running reward function tests...")
    for i, (pred, gt, expected) in enumerate(test_cases):
        # Note: In Verl, ground_truth is often passed as the target value or a dict
        reward_dict = compute_score(pred, gt)
        reward = reward_dict['score']
        status = "PASSED" if reward == expected else "FAILED"
        print(f"Test {i}: Pred='{pred}', GT='{gt}', Expected={expected}, Got={reward} -> {status}")
        if reward != expected:
             print(f"  Error: {pred} vs {gt}")

if __name__ == "__main__":
    test_reward()
