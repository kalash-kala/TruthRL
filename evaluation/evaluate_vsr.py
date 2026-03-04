
import os
import json
import torch
import pandas as pd
import argparse
import ast
import copy
from tqdm import tqdm
import sys

# Add the local verl directory to sys.path so we can import the reward function
import_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "verl"))
sys.path.insert(0, import_path)

from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from verl.utils.reward_score.vsr_lexical import compute_score
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration & Argument Parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on VSR Task")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint or huggingface model")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to load processor/tokenizer from (defaults to model_path). Use base model path when evaluating VeRL checkpoints that lack processor files.")
    parser.add_argument("--data_path", type=str, default="/home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test.parquet", help="Path to the test parquet file")
    parser.add_argument("--output_dir", type=str, default="results/vsr_eval", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--name", type=str, default="eval_run", help="Name of the evaluation run (for logging)")
    parser.add_argument("--no_timestamp", action="store_true", help="Disable timestamp in output directory name")
    return parser.parse_args()

def normalize_text(text):
    """Normalize text for consistent scoring/logging"""
    if not isinstance(text, str):
        return ""
    return text.lower().strip().rstrip(".")

def main():
    args = parse_args()
    # If --processor_path not specified, fall back to model_path
    processor_path = args.processor_path if args.processor_path else args.model_path
    
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.no_timestamp:
        run_dir = os.path.join(args.output_dir, args.name)
    else:
        run_dir = os.path.join(args.output_dir, f"{args.name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f"Starting VSR Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {run_dir}")
    print(f"==================================================")

    # -----------------------------------------------------------------------------
    # Load Model & Processor
    # -----------------------------------------------------------------------------
    print(f"Loading model processing tools...")
    print(f"Processor path: {processor_path}")
    try:
        # Load processor from processor_path (base model for VeRL checkpoints, or model_path for HF models)
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        
        # Load model using bfloat16 and Flash Attention 2 for efficiency
        # Using a specific device 'cuda:0' to avoid device mismatch issues with device_map="auto"
        # for multimodal models. 3B model easily fits on one GPU (approx 6-8GB VRAM).
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
            trust_remote_code=True
        )
        model.eval() # Set to eval mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # -----------------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------------
    print(f"Loading test data...")
    try:
        df = pd.read_parquet(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Total samples: {len(df)}")

    # -----------------------------------------------------------------------------
    # Comparison Logic (using vsr_lexical reward function)
    # -----------------------------------------------------------------------------
    # We will use compute_score from training/verl/verl/utils/reward_score/vsr_lexical.py
    # Returns:
    #   1.0: Correct (True/False match)
    #   0.0: "I don't know" (Safe Refusal)
    #  -1.0: Incorrect (Hallucination)

    results = []
    
    # Metrics Counters
    n_correct = 0      # Score = 1.0
    n_incorrect = 0    # Score = -1.0
    n_refusal = 0      # Score = 0.0 (I don't know)
    n_total = 0

    # -----------------------------------------------------------------------------
    # Inference Loop
    # -----------------------------------------------------------------------------
    print("Starting inference...")

    # Open a JSONL file for live logging results
    detail_log_path = os.path.join(run_dir, "evaluation_details.jsonl")
    
    with open(detail_log_path, "w") as detail_file:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # 1. Prepare Inputs
                # Convert pyarrow/numpy arrays to list and deepcopy to avoid mutating source dataframe
                raw_prompt = row['prompt']
                if hasattr(raw_prompt, 'tolist'):
                    raw_prompt = raw_prompt.tolist()
                messages = copy.deepcopy(list(raw_prompt))
                
                # Patch system prompt to encourage reasoning-first behavior
                # for msg in messages:
                #     if msg['role'] == 'system':
                #         msg['content'] = (
                #             "You are a visual spatial reasoning expert. Analyze the image and the statement. "
                #             "First, provide your detailed reasoning in the <reasoning start> reasoning <reasoning end> format. "
                #             "Then, based on your reasoning, answer exactly 'True', 'False', or 'I don't know' "
                #             "in the /box[<answer>]/ format (e.g., /box[True]/)."
                #         )
                
                # Extract image path correctly based on observed structure:
                image_path = "unknown"
                if 'images' in row and len(row['images']) > 0:
                    first_img = row['images'][0]
                    if isinstance(first_img, dict) and 'image' in first_img:
                        image_path = first_img['image']
                    elif hasattr(first_img, 'get'): # Handles dict-like numpy structures
                        image_path = first_img.get('image', first_img)
                    elif isinstance(first_img, str):
                        image_path = first_img
                elif 'image' in row:
                     # Some datasets might use singular 'image'
                     image_path = row['image']
                
                # Fix the messages format to include the image for Qwen2-VL / Qwen2.5-VL
                # Qwen expects the user message content to be a list of dicts:
                # [{"type": "image", "image": "file:///..."}, {"type": "text", "text": "..."}]
                if image_path != "unknown":
                    for msg in messages:
                        if msg['role'] == 'user':
                            orig_content = msg['content']
                            # Remove the literal '<image>' tag if it's there
                            clean_text = orig_content.replace("<image>\n", "").replace("<image>", "").strip()
                            msg['content'] = [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": clean_text}
                            ]
                
                # Extract ground truth from reward_model dict
                # reward_model: {'ground_truth': 'False', 'style': 'lexical'}
                ground_truth = "unknown"
                if 'reward_model' in row:
                    rm_data = row['reward_model']
                    if isinstance(rm_data, dict):
                        ground_truth = str(rm_data.get('ground_truth', 'unknown'))
                    elif isinstance(rm_data, str):
                        try:
                            rm_dict = ast.literal_eval(rm_data)
                            ground_truth = str(rm_dict.get('ground_truth', 'unknown'))
                        except:
                            pass
                
                # Preprocess for Qwen2.5-VL
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                inputs = inputs.to("cuda:0")

                # 2. Generate
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                prediction = output_text.strip()

                # 3. Compute Score using Custom Reward Function
                # compute_score expects(solution_str, ground_truth, ...)
                # Note: vsr_lexical.py in this repo handles 'ground_truth' potentially being a dict or value.
                # We pass our extracted ground_truth string directly.
                score_dict = compute_score(prediction, ground_truth)
                score = score_dict['score']

                # 4. Update Metrics
                n_total += 1
                if score == 1.0:
                    n_correct += 1
                    result_type = "correct"
                elif score == 0.0:
                    n_refusal += 1
                    result_type = "refusal" 
                else: # -1.0
                    n_incorrect += 1
                    result_type = "incorrect"

                # 5. Log Result
                # User requested: Caption, Image Location, Model Answer, Ground Truth, Verdict
                
                # Extract image path from row if available
                log_image_path = "unknown"
                if 'images' in row and len(row['images']) > 0:
                    first_img = row['images'][0]
                    if isinstance(first_img, dict) and 'image' in first_img:
                        log_image_path = first_img['image']
                    elif hasattr(first_img, 'get'):
                        log_image_path = first_img.get('image', first_img)
                    else:
                        log_image_path = first_img
                elif 'image' in row:
                     log_image_path = row['image']
                
                # Extract Caption (User Prompt) - usually the last user message before assistant
                caption = "unknown"
                for msg in row['prompt']:  # Use the original row prompt which might just have strings
                    if msg['role'] == 'user':
                        caption = msg['content']

                result_entry = {
                    "index": index,
                    "prompt": text,
                    "caption": caption,
                    "image_location": str(log_image_path),
                    "model_answer": prediction,
                    "ground_truth": ground_truth,
                    "verdict": result_type,  # correct, refusal, incorrect
                    "score": score
                }
                
                results.append(result_entry)
                detail_file.write(json.dumps(result_entry) + "\n")
                detail_file.flush() # Force write to disk for real-time monitoring

            except Exception as e:
                print(f"Error processing item {index}: {e}")
                continue

    # -----------------------------------------------------------------------------
    # Final Metrics & Summary
    # -----------------------------------------------------------------------------
    accuracy = n_correct / n_total if n_total > 0 else 0
    refusal_rate = n_refusal / n_total if n_total > 0 else 0
    hallucination_rate = n_incorrect / n_total if n_total > 0 else 0
    
    # Truthfulness Score (Net Score)
    # Designed to reward correctness (+1) and punish hallucinations (-1), while being neutral to valid refusals (0)
    # Score = (n_correct - n_incorrect) / n_total
    truthfulness_score = (n_correct - n_incorrect) / n_total if n_total > 0 else 0
    
    summary = {
        "model_path": args.model_path,
        "n_samples": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_refusal": n_refusal,
        "accuracy": accuracy,
        "refusal_rate": refusal_rate,
        "hallucination_rate": hallucination_rate,
        "truthfulness_score": truthfulness_score,
        "timestamp": timestamp
    }
    
    summary_path = os.path.join(run_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "="*50)
    print(f"Evaluation Complete")
    print(f"Accuracy: {accuracy:.2%} ({n_correct}/{n_total})")
    print(f"Truthfulness Score: {truthfulness_score:.4f} (Net: +1/-1/0)")
    print(f"Refusal Rate (I don't know): {refusal_rate:.2%} ({n_refusal}/{n_total})")
    print(f"Hallucination/Error Rate: {hallucination_rate:.2%} ({n_incorrect}/{n_total})")
    print(f"Results saved to: {run_dir}")
    print("="*50)

if __name__ == "__main__":
    main()