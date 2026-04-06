#!/usr/bin/env python3
"""
Convert VQA Parquet -> VERL-compatible validation Parquet.

Specifically handles filtering out images already processed or perturbed.
Outputs to: /home/kalashkala/Datasets/VQAv2/test_vqa_for_verl.parquet
"""

import os
import argparse
import pandas as pd
from datasets import Dataset

# Syncing with the SYSTEM_PROMPT used in convert_gqa_spatial_for_verl.py
SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n"
    "The reasoning process MUST BE enclosed within <think></think> tags.\n"
    "The answer MUST BE enclosed within <answer></answer> tags.\n"
    "Do not output anything outside these tags.\n"
    "Base your reasoning only on the visible image evidence.\n"
    "If the image does not support a confident answer, output I don't know inside <answer></answer>.\n"
    "Example 1:\n"
    "<think>\n"
    "The cup is positioned on the right side of the plate. The plate is clearly to the left of the cup, and there is no stronger relation such as above or behind.\n"
    "</think>\n"
    "<answer>The cup is to the right of the plate.</answer>\n\n"
    "Example 2:\n"
    "<think>\n"
    "The lamp appears vertically higher than the sofa and is not merely near it. The strongest visible relation is above.\n"
    "</think>\n"
    "<answer>The lamp is above the sofa.</answer>\n\n"
    "Example 3:\n"
    "<think>\n"
    "The target object is partially occluded and the relative position is unclear. Multiple relations are possible, but the image does not support one answer confidently.\n"
    "</think>\n"
    "<answer>I don't know</answer>"
)

def build_parquet_row(row, index, image_save_dir):
    question = str(row.get('question', '')).strip('"\'')
    
    image_id = str(row.get('image_id', ''))
    image_val = row.get('image', {})
    
    # Save the binary image data to disk
    save_path = ""
    if isinstance(image_val, dict) and 'bytes' in image_val:
        save_path = os.path.join(image_save_dir, f"{image_id}.jpg")
        with open(save_path, "wb") as f:
            f.write(image_val['bytes'])
    else:
        # Fallback if image data is missing or in different format
        print(f"Warning: No binary data for image_id {image_id}")
        if isinstance(image_val, dict):
            save_path = image_val.get('path', '')
        else:
            save_path = str(image_val)
        
    extra_info = {
        "index": index,
        "question_id": str(row.get('question_id', '')),
        "image_id": image_id,
        "question_type": str(row.get('question_type', '')),
        "answer_type": str(row.get('answer_type', '')),
        "split": "test_vqa"
    }

    # Extract multiple_choice_answer as the ground truth
    ground_truth = str(row.get('multiple_choice_answer', ''))

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{question}"}
        ],
        "images": [{"image": f"file://{save_path}"}],
        "reward_model": {
            "ground_truth": ground_truth,
            "style": "open_text"
        },
        "ability": "visual_question_answering",
        "extra_info": extra_info
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, default="/home/kalashkala/Datasets/VQAv2/vqa_stratified_300.parquet")
    parser.add_argument("--output_path", type=str, default="/home/kalashkala/Datasets/VQAv2/test_vqa_for_verl.parquet")
    parser.add_argument("--image_save_dir", type=str, default="/home/kalashkala/Datasets/VQAv2/test_vqa_images")
    parser.add_argument("--processed_dir", type=str, default="/home/kalashkala/Datasets/VQAv2/processed_for_verl/images")
    parser.add_argument("--perturbed_dir", type=str, default="/home/kalashkala/Datasets/VQAv2/sam_perturbed_images")
    args = parser.parse_args()

    # 0. Setup image save directory
    os.makedirs(args.image_save_dir, exist_ok=True)
    print(f"Images will be saved to: {args.image_save_dir}")

    # 1. Collect excluded image names (basenames)
    excluded_images = set()
    for directory in [args.processed_dir, args.perturbed_dir]:
        if os.path.exists(directory):
            print(f"Reading exclusion directory: {directory}")
            for f in os.listdir(directory):
                excluded_images.add(os.path.basename(f))
        else:
            print(f"Warning: Exclusion directory {directory} does not exist.")

    print(f"Total excluded unique image filenames: {len(excluded_images)}")

    # 2. Load dataset
    print(f"Loading {args.input_parquet} ...")
    df = pd.read_parquet(args.input_parquet)
    print(f"Original dataset rows: {len(df)}")

    # 3. Filter rows based on image path basename
    def get_basename(img_val):
        if isinstance(img_val, dict):
            p = img_val.get('path', '')
            if not p: # use image_id if path is absent
                return ""
        else:
            p = str(img_val)
        return os.path.basename(p)

    df['image_basename'] = df['image'].apply(get_basename)
    
    initial_count = len(df)
    # If image_basename is empty, we might need a different strategy for filtering,
    # but based on the previous run, it successfully filtered 1050 images.
    filtered_df = df[~df['image_basename'].isin(excluded_images)].copy()
    final_count = len(filtered_df)
    
    print(f"Filtered out {initial_count - final_count} overlapping images.")
    print(f"Rows remaining for validation: {final_count}")

    # 4. Transform to VERL format and Save images
    print("Formatting data for VERL and saving images to disk...")
    records = filtered_df.to_dict(orient="records")
    formatted_data = [build_parquet_row(row, idx, args.image_save_dir) for idx, row in enumerate(records)]

    # 5. Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    dataset = Dataset.from_list(formatted_data)
    dataset.to_parquet(args.output_path)
    print(f"✅ Successfully saved {len(dataset)} examples to {args.output_path}")

if __name__ == "__main__":
    main()
