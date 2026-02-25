# command to run
# python3 /home/debarpanb1/kalashkala/TruthRL/scripts/preprocess_vsr.py --train_path /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/data/train_sampled.jsonl --test_path /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/data/test_sampled.jsonl --val_path /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/data/val_sampled.jsonl --output_dir /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet --image_dir /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/images
# python3 /home/debarpanb1/kalashkala/TruthRL/scripts/preprocess_vsr.py --train_path /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/data/train_sampled.jsonl --test_path /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/data/test_sampled.jsonl --val_path /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/data/val_sampled.jsonl --output_dir /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet --image_dir /home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/images


import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoProcessor

def preprocess_vsr(train_path, test_path, val_path, output_dir, image_base_dir, model_name="google/gemma-3-4b-it"):
    """
    Preprocesses the VSR dataset (JSONL) into Verl-compatible Parquet format.
    Following the structure expected by verl.utils.dataset.rl_dataset.RLHFDataset
    """

    os.makedirs(output_dir, exist_ok=True)

    # Verl requires the literal '<image>' tag in the prompt for its internal processing.
    image_token = "<image>"

    def process_file(file_path, split_name):
        print(f"Processing {split_name} data from {file_path}...")
        df = pd.read_json(file_path, lines=True)

        processed_data = []
        for _, row in df.iterrows():
            # 1. Resolve Image Path
            image_filename = row['image']
            image_path = os.path.join(image_base_dir, image_filename)
            
            # Using file:// prefix to ensure fetch_image handles it correctly
            image_url = f"file://{image_path}"

            # 2. Map Label to Text
            label_text = "True" if row['label'] == 1 else "False"

            # 3. Construct Prompt (Messages with standard <image> token)
            caption = row['caption']
            
            system_prompt_with_idk = (
                "You are a visual spatial reasoning expert. Analyze the image and the statement. "
                "First, provide your detailed reasoning in the <reasoning start> reasoning <reasoning end> format. "
                "Then, based on your reasoning, answer exactly 'True', 'False', or 'I don't know' "
                "in the /box[<answer>]/ format (e.g., /box[True]/)."
            )

            system_prompt_without_idk = (
                "You are a visual spatial reasoning expert. Analyze the image and the statement. "
                "First, provide your detailed reasoning in the <reasoning start> reasoning <reasoning end> format. "
                "Then, based on your reasoning, answer exactly 'True' or 'False' "
                "in the /box[<answer>]/ format (e.g., /box[True]/)."
            )

            # Important: Use '<image>' as expected by Verl
            messages = [
                {"role": "system", "content": system_prompt_with_idk},
                {"role": "user", "content": f"{image_token}\n{caption}"}
            ]

            # 4. Construct Data Item
            data_item = {
                "prompt": messages,
                "images": [{"image": image_url}], # Top-level column as list of dicts
                "ability": "visual_spatial_reasoning",
                "reward_model": {"style": "lexical", "ground_truth": label_text},
                "extra_info": {"split": split_name, "index": _}
            }
            processed_data.append(data_item)

        # Convert to Hugging Face Dataset and save as Parquet
        dataset = Dataset.from_list(processed_data)
        output_path = os.path.join(output_dir, f"{split_name}.parquet")
        dataset.to_parquet(output_path)
        print(f"Saved {len(dataset)} examples to {output_path}")

    split = 'with_idk'

    # Process Splits
    process_file(train_path, f"train_{split}")
    if test_path and os.path.exists(test_path):
        process_file(test_path, f"test_{split}")
    if val_path and os.path.exists(val_path):
        process_file(val_path, f"validation_{split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess VSR dataset for Verl/TruthRL")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train_sampled.jsonl")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test_sampled.jsonl")
    parser.add_argument("--val_path", type=str, default=None, help="Path to val_sampled.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save parquet files")
    parser.add_argument("--image_dir", type=str, required=True, help="Absolute path to image directory")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it", help="Model name on HuggingFace")

    args = parser.parse_args()
    abs_image_dir = os.path.abspath(args.image_dir)
    preprocess_vsr(args.train_path, args.test_path, args.val_path, args.output_dir, abs_image_dir, args.model_name)
