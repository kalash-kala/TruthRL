#!/usr/bin/env python3
"""
Normalized VSR/MLLM parquet files to use the literal '<image>' tag.
Verl requires the '<image>' tag for its multimodal message processing logic,
regardless of the specific model's internal image token (e.g., Qwen's <|image_pad|> or Gemma's 🖼️).
"""

import os
import argparse
import pandas as pd
from transformers import AutoProcessor

def normalize_parquet(path, model_name):
    """
    Ensures prompts in the parquet file use '<image>' instead of model-specific tokens.
    """
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
        return

    # If it's a directory, process all parquets inside
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".parquet"):
                normalize_parquet(os.path.join(path, f), model_name)
        return

    print(f"--- Processing {path} ---")
    
    # Load processor to identify the special token
    print(f"Loading processor for {model_name}...")
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        image_token = processor.image_token
    except Exception as e:
        print(f"Warning: Could not load processor for {model_name}. Error: {e}")
        print("Will attempt to look for common MLLM tokens if found.")
        image_token = None

    df = pd.read_parquet(path)
    
    # Common tokens used by various MLLMs that should be normalized to <image>
    # Adding <|image_pad|> specifically as it's common for Qwen
    tokens_to_replace = ["<|image_pad|>", "<|vision_start|><|vision_end|>"]
    if image_token and image_token not in tokens_to_replace:
        tokens_to_replace.append(image_token)
        
    print(f"Normalizing tokens {tokens_to_replace} to '<image>'...")

    def fix_prompt(prompt):
        # Handle Verl's list-of-dicts format
        if not isinstance(prompt, (list, tuple)):
            return prompt
            
        modified = False
        for msg in prompt:
            content = msg.get('content')
            if isinstance(content, str):
                original_content = content
                for token in tokens_to_replace:
                    if token in content:
                        content = content.replace(token, '<image>')
                
                if content != original_content:
                    msg['content'] = content
                    modified = True
        return prompt

    df['prompt'] = df['prompt'].apply(fix_prompt)
    
    # Save back
    print(f"Saving normalized file to {path}...")
    df.to_parquet(path)
    print("✓ Done")

    # Verification sample
    if len(df) > 0:
        print("\nVerification (First prompt):")
        for msg in df.iloc[0]['prompt']:
             print(f"  [{msg['role']}]: {repr(msg['content'][:100])}...")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize MLLM parity files to use standard <image> tags.")
    parser.add_argument("--path", type=str, required=True, help="Path to a .parquet file or directory containing .parquet files")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name to identify specific image tokens")
    
    args = parser.parse_args()
    normalize_parquet(args.path, args.model)
