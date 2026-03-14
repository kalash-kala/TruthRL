#!/usr/bin/env python3
"""
Convert VQAv2 HF Parquet → VERL-compatible training Parquet.

This script:
  1. Reads the VQAv2 parquet downloaded from Hugging Face.
  2. Extracts embedded image bytes → saves as .jpg files on disk.
  3. Restructures each row into the schema expected by VERL's RLHFDataset:
       prompt, images, ability, reward_model, extra_info
  4. Saves the result as a Parquet file ready for training.

Usage:
  python3 /home/kalashkala/TruthRL/scripts/convert_vqa_for_verl.py \
    --input_parquet /home/kalashkala/Datasets/VQAv2/vqa_train.parquet \
    --output_dir /home/kalashkala/Datasets/VQAv2/processed_for_verl \
    --output_name train_vqa.parquet \
    --max_samples 0
"""

import os
import io
import argparse
from PIL import Image
import pandas as pd
from datasets import Dataset
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────
# System prompt placeholder — UPDATE THIS with your actual prompt later
# ────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a visual question answering expert. "
    "Analyze the image and answer the question. "
    "First, provide your detailed reasoning in the "
    "<reasoning start> reasoning <reasoning end> format. "
    "Then, provide your final answer in the /box[<answer>]/ format."
    "Adhere to the following rules: "
    "1. If you are not sure about the answer, respond with 'I don't know'. "
    "2. If you are sure about the answer, then answer the question in 1-2 sentences covering the points which you deem important. "
    "3. Do not repeat the question in your answer. "
)


def extract_and_save_image(image_dict, image_dir, row_index):
    """
    Extract image bytes from the HF dataset row and save as a .jpg.
    Uses the original filename from the dataset when available,
    falling back to a generated name.

    Returns:
        Absolute path to the saved image file.
    """
    image_bytes = image_dict["bytes"]
    original_name = image_dict.get("path")

    if original_name:
        # Use original filename (e.g. COCO_val2014_000000034257.jpg)
        image_filename = original_name
    else:
        image_filename = f"vqa_image_{row_index}.jpg"

    image_path = os.path.join(image_dir, image_filename)

    # Skip if already extracted (idempotent)
    if not os.path.exists(image_path):
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path, format="JPEG")

    return image_path


def build_verl_row(row, image_path, row_index):
    """
    Transform a single VQAv2 row into the VERL-expected schema.

    Columns produced:
      - prompt:       list[dict]   — chat-style messages (system + user)
      - images:       list[dict]   — [{"image": "file:///..."}]
      - ability:      str          — routing key for reward function
      - reward_model: dict         — ground truth info for the reward fn
      - extra_info:   dict         — metadata for logging / debugging
    """
    question = row["question"]

    # Collect ALL human annotator answers (typically 10 per question)
    all_answers = [ans["answer"] for ans in row["answers"]]

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"<image>\n{question}"},
        ],
        "images": [{"image": f"file://{image_path}"}],
        "ability": "visual_question_answering",
        "reward_model": {
            "acceptable_answers": all_answers,
            "multiple_choice_answer": row.get("multiple_choice_answer", ""),
            "style": "vqa_llm_judge",
        },
        "extra_info": {
            "index": row_index,
            "question_id": int(row["question_id"]),
            "question_type": row.get("question_type", ""),
            "answer_type": row.get("answer_type", ""),
            "split": "train",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert VQAv2 HF Parquet to VERL-compatible training format"
    )
    parser.add_argument(
        "--input_parquet", type=str, required=True,
        help="Path to the source VQAv2 parquet file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for output parquet and extracted images"
    )
    parser.add_argument(
        "--output_name", type=str, default="train_vqa.parquet",
        help="Filename for the output parquet"
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Limit rows to process (0 = all)"
    )
    args = parser.parse_args()

    # ── 1. Set up directories ─────────────────────────────────────────────
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # ── 2. Load source parquet ────────────────────────────────────────────
    print(f"Loading {args.input_parquet} ...")
    df = pd.read_parquet(args.input_parquet)
    total = len(df)
    print(f"  Total rows in source: {total}")

    if args.max_samples > 0:
        df = df.head(args.max_samples)
        print(f"  Limiting to first {args.max_samples} rows")

    # ── 3. Process rows ──────────────────────────────────────────────────
    formatted_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        # Extract image bytes → save to disk
        image_path = extract_and_save_image(row["image"], image_dir, idx)
        # Build the VERL-formatted row
        verl_row = build_verl_row(row, image_path, idx)
        formatted_data.append(verl_row)

    # ── 4. Save as Parquet ────────────────────────────────────────────────
    print("\nSaving VERL-compatible Parquet ...")
    dataset = Dataset.from_list(formatted_data)
    output_path = os.path.join(args.output_dir, args.output_name)
    dataset.to_parquet(output_path)
    print(f"✅ Saved {len(dataset)} examples to {output_path}")

    # ── 5. Quick sanity check ─────────────────────────────────────────────
    print("\n── Sanity Check (first row) ──")
    sample = formatted_data[0]
    print(f"  Prompt roles:    {[m['role'] for m in sample['prompt']]}")
    print(f"  Image path:      {sample['images'][0]['image']}")
    print(f"  Ability:         {sample['ability']}")
    print(f"  # answers:       {len(sample['reward_model']['acceptable_answers'])}")
    print(f"  MC answer:       {sample['reward_model']['multiple_choice_answer']}")
    print(f"  Question ID:     {sample['extra_info']['question_id']}")


if __name__ == "__main__":
    main()
