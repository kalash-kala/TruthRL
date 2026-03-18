#!/usr/bin/env python3
"""
Convert GQA Spatial Parquet → VERL-compatible training Parquet.

This script:
  1. Reads the GQA Spatial Parquet.
  2. Restructures each row into the VERL-expected schema:
       prompt, images, ability, reward_model, extra_info
  3. Saves the result as a Parquet file ready for training.

Usage:
  python3 /home/kalashkala/TruthRL/scripts/convert_gqa_spatial_for_verl.py \
    --input_parquet /home/kalashkala/Datasets/GQA/val_spatial_instructions_1k.parquet \
    --output_path /home/kalashkala/Datasets/GQA/val_spatial_for_verl.parquet
"""

import os
import argparse
import pandas as pd
from datasets import Dataset


# ────────────────────────────────────────────────────────────────────────────
# System prompt — matches the one used in convert_vsr_open_text_for_verl.py
# ────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a visual question answering expert. \n"
    "Analyze the image and answer the question. \n"
    "First, provide your detailed reasoning in the "
    "<reasoning start> reasoning <reasoning end> format. \n"
    "Then, provide your final answer in the /box[<answer>] format.\n"
    "Adhere to the following rules: \n"
    "1. If you are not sure about the answer, respond with 'I don't know'. "
    "2. If you are sure about the answer, then answer the question in 1-2 sentences covering the points which you deem important. \n"
    "3. Do not repeat the question in your answer. \n"
    "Here is an example: \n"
    "Question: Where is the cat relative to the dog? \n"
    "Your reasoning format: <reasoning start> The cat is lying next to the dog on the floor. The dog is positioned near the cat's head, and the cat appears to be resting or sleeping beside it. <reasoning end>"
    "Your answer format: /box[The cat is beside the dog.]"
)


def build_parquet_row(row, index, image_dir):
    """Transform a single GQA row into the VERL-expected schema."""
    question = str(row.get('question', '')).strip('"\'')
    image_id = row['imageId']
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    
    # Core requirements requested by user
    extra_info = {
        "index": index,
        "id": str(row.get('id', '')),
        "imageId": str(image_id),
        "answer": str(row.get('answer', '')),
        "split": "val_spatial"
    }
    
    # Adding 'other stuff' as simple strings to avoid complex type errors in HF datasets
    optional_columns = ['isBalanced', 'groups', 'entailed', 'equivalent', 'types', 'annotations', 'semantic', 'semanticStr']
    for col in optional_columns:
        if col in row:
            val = row[col]
            extra_info[col] = str(val) if val is not None else ""

    return {
        # The prompt format expected by Qwen
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{question}"}
        ],
        # The image format expected by VERL / Qwen
        "images": [{"image": f"file://{image_path}"}],
        # Passed directly to the reward function's `ground_truth` argument
        "reward_model": {
            "ground_truth": str(row.get('fullAnswer', '')),
            "style": "open_text"
        },
        "ability": "visual_spatial_reasoning",
        "extra_info": extra_info
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert GQA Spatial Parquet to VERL-compatible Parquet"
    )
    parser.add_argument(
        "--input_parquet", type=str, 
        default="/home/kalashkala/Datasets/GQA/val_spatial_instructions_1k.parquet",
        help="Path to the source GQA spatial parquet file"
    )
    parser.add_argument(
        "--output_path", type=str, 
        default="/home/kalashkala/Datasets/GQA/val_spatial_for_verl.parquet",
        help="Path to save the output parquet file"
    )
    parser.add_argument(
        "--image_dir", type=str, 
        default="/home/kalashkala/Datasets/GQA/val_images",
        help="Directory where GQA images are stored"
    )
    args = parser.parse_args()

    # ── 1. Load Parquet ──────────────────────────────────────────────────
    print(f"Loading {args.input_parquet} ...")
    df = pd.read_parquet(args.input_parquet)
    print(f"  Total rows: {len(df)}")

    # ── 2. Apply the transformation ──────────────────────────────────────
    formatted_data = [build_parquet_row(row, idx, args.image_dir) for idx, row in df.iterrows()]

    # ── 3. Save to Parquet ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    dataset = Dataset.from_list(formatted_data)
    dataset.to_parquet(args.output_path)
    print(f"✅ Successfully saved {len(dataset)} examples to {args.output_path}")

    # ── 4. Quick sanity check ─────────────────────────────────────────────
    print("\n── Sanity Check (first row) ──")
    sample = formatted_data[0]
    print(f"  Prompt roles:      {[m['role'] for m in sample['prompt']]}")
    print(f"  Question:          {sample['prompt'][1]['content'][:80]}...")
    print(f"  Image path:        {sample['images'][0]['image']}")
    print(f"  Ground truth:      {sample['reward_model']['ground_truth']}")
    print(f"  Ability:           {sample['ability']}")


if __name__ == "__main__":
    main()
