
import os
import argparse
import pandas as pd
from transformers import AutoProcessor

def fix_parquet(parquet_path, model_name):
    """
    Replaces literal '<image>' strings in parquet prompts with the 
    special image token required by the model's processor.
    """
    if not os.path.exists(parquet_path):
        print(f"Error: Path {parquet_path} does not exist.")
        return

    print(f"Loading processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    image_token = processor.image_token
    
    print(f"Special Image Token for {model_name} is: '{image_token}'")

    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    def replace_token(prompt):
        # Handle list of dictionary format (standard Verl prompt)
        for msg in prompt:
            if isinstance(msg.get('content'), str):
                msg['content'] = msg['content'].replace('<image>', image_token)
        return prompt

    print("Applying token replacement...")
    df['prompt'] = df['prompt'].apply(replace_token)
    
    # Save back
    print(f"Saving updated file to {parquet_path}...")
    df.to_parquet(parquet_path)
    print("SUCCESS: Parquet file fixed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Parquet image tokens for MLLM training.")
    parser.add_argument("--path", type=str, required=True, help="Path to the .parquet file")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="Model name on HuggingFace")
    
    args = parser.parse_args()
    fix_parquet(args.path, args.model)
