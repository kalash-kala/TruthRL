#!/usr/bin/env python3
"""
Convert VSR binary spatial reasoning captions to open-ended questions.

This script:
  1. Reads the VSR dataset (JSONL format).
  2. Filters only data points where ground truth label is True.
  3. Keeps one entry per unique image (deduplication).
  4. Uses Meta-Llama-3.1-8B-Instruct (served via vLLM) to convert
     each True binary caption into an open-ended visual question.
  5. Saves results (image_location, ground_truth_caption, llm_question) as CSV.

Usage:
  Step 1 — Start the vLLM server (in a separate terminal):
    python3 -m vllm.entrypoints.openai.api_server \
      --model /home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct \
      --dtype auto \
      --port 8000 \
      --gpu-memory-utilization 0.85

  Step 2 — Run this script:
    python3 /home/kalashkala/TruthRL/scripts/convert_vsr_to_open_text.py \
      --input_path /home/kalashkala/visual-spatial-reasoning/truthrl-sample/data/train_sampled.jsonl \
      --output_path /home/kalashkala/TruthRL/vsr_open_text_train.csv \
      --image_dir /home/kalashkala/visual-spatial-reasoning/truthrl-sample/images \
      --api_base http://localhost:8000/v1 \
      --batch_size 32
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ────────────────────────────────────────────────────────────────────────────
# ▸ PROMPT — Customise this section with your own prompt
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at converting spatial relationship statements into natural open-ended questions.

You are given a caption that describes a true spatial relation in an image.

Your task is to rewrite the caption as a short open-ended question such that:
1. The question asks about the spatial relation.
2. The question must NOT be answerable by "yes" or "no".
3. The question must preserve the exact entities mentioned in the caption.
4. The question must be natural and grammatically correct.
5. The question should make the expected answer a short phrase describing the relation.
6. Do not add new objects, attributes, or assumptions.
7. Output only the question and nothing else.

Examples:

Caption: "The dog is to the left of the boy."
Question: "Where is the dog relative to the boy?"

Caption: "The book is on the table."
Question: "Where is the book?"

Caption: "The lamp is above the sofa."
Question: "Where is the lamp relative to the sofa?"

Caption: "The cat is under the chair."
Question: "Where is the cat relative to the chair?"
"""

USER_PROMPT_TEMPLATE = """Convert this TRUE spatial relationship statement into an open-ended visual question:

Statement: "{caption}"

Open-ended question:"""

# ────────────────────────────────────────────────────────────────────────────


def create_client(api_base: str) -> OpenAI:
    """Create an OpenAI-compatible client pointing at the vLLM server."""
    return OpenAI(
        base_url=api_base,
        api_key="empty",  # vLLM does not require an API key
    )


def generate_question(client: OpenAI, model_name: str, caption: str,
                       temperature: float = 0.7, max_tokens: int = 128) -> str:
    """Call the LLM to convert a single binary caption into an open-ended question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(caption=caption)},
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] Failed for caption '{caption}': {e}")
        return ""


def generate_questions_batch(client: OpenAI, model_name: str,
                              captions: list[str], temperature: float = 0.7,
                              max_tokens: int = 128) -> list[str]:
    """
    Generate questions one-by-one through the chat API.
    (vLLM's OpenAI-compatible server handles concurrent requests efficiently,
     so we use a simple loop with a progress bar.)
    """
    results = []
    for caption in tqdm(captions, desc="Generating questions", unit="q"):
        q = generate_question(client, model_name, caption, temperature, max_tokens)
        results.append(q)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert VSR binary captions (True only) to open-ended questions via LLM"
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the VSR JSONL file (e.g. train.jsonl)"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the output CSV"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Base directory for images. If provided, image paths will be "
             "prefixed with this directory."
    )
    parser.add_argument(
        "--api_base", type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="/home/kalashkala/Models/Meta-Llama-3.1-8B-Instruct",
        help="Model name/path as registered with vLLM"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="LLM generation temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128,
        help="Maximum tokens for LLM generation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="(reserved for future batching — currently processes sequentially)"
    )

    args = parser.parse_args()

    # ── 1. Load data ───────────────────────────────────────────────────────
    print(f"Loading data from {args.input_path} ...")
    df = pd.read_json(args.input_path, lines=True)
    print(f"  Total rows: {len(df)}")

    # ── 2. Filter: keep only True labels ───────────────────────────────────
    df_true = df[df["label"] == 1].copy()
    print(f"  Rows with label=True: {len(df_true)}")

    # ── 3. Deduplicate: keep one caption per unique image ──────────────────
    df_unique = df_true.drop_duplicates(subset="image", keep="first")
    print(f"  Unique images (after dedup): {len(df_unique)}")

    # ── 4. Resolve image paths ─────────────────────────────────────────────
    if args.image_dir:
        df_unique["image_location"] = df_unique["image"].apply(
            lambda x: os.path.join(args.image_dir, x)
        )
    else:
        df_unique["image_location"] = df_unique["image"]

    captions = df_unique["caption"].tolist()
    image_locations = df_unique["image_location"].tolist()

    # ── 5. Connect to vLLM and generate questions ──────────────────────────
    print(f"\nConnecting to vLLM at {args.api_base} ...")
    client = create_client(args.api_base)

    print(f"Generating open-ended questions for {len(captions)} captions ...\n")
    questions = generate_questions_batch(
        client, args.model_name, captions,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # ── 6. Build output DataFrame and save ─────────────────────────────────
    output_df = pd.DataFrame({
        "image_location": image_locations,
        "ground_truth_caption": captions,
        "llm_question": questions,
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    output_df.to_csv(args.output_path, index=False)
    print(f"\n✅ Saved {len(output_df)} rows to {args.output_path}")

    # Quick preview
    print("\n── Preview (first 5 rows) ──")
    for i, row in output_df.head(5).iterrows():
        print(f"  Image:   {row['image_location']}")
        print(f"  Caption: {row['ground_truth_caption']}")
        print(f"  Question: {row['llm_question']}")
        print()


if __name__ == "__main__":
    main()
