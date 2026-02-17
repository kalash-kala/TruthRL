"""
Fix VSR parquet files to use <image> tags instead of processor.image_token.
This ensures compatibility with verl's _build_messages() logic.
"""
import pandas as pd
from transformers import AutoProcessor

MODEL_NAME = "google/gemma-3-4b-it"
TRAIN_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"
TEST_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test.parquet"

def fix_dataset(parquet_path):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Get the special image character
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    image_token_char = processor.image_token  # This is '🖼️' for Gemma 3
    
    print(f"Replacing '{image_token_char}' with '<image>'...")
    
    def fix_prompt(prompt_list):
        for msg in prompt_list:
            if isinstance(msg.get('content'), str):
                # Replace the special character with the literal tag
                msg['content'] = msg['content'].replace(image_token_char, '<image>')
        return prompt_list
    
    df['prompt'] = df['prompt'].apply(fix_prompt)
    
    # Save back
    print(f"Saving fixed dataset to {parquet_path}...")
    df.to_parquet(parquet_path)
    print("✓ Done")
    
    # Verify
    sample = df.iloc[0]
    print(f"\nVerification - First prompt:")
    for msg in sample['prompt']:
        print(f"  [{msg['role']}]: {repr(msg['content'][:80])}...")

if __name__ == "__main__":
    fix_dataset(TRAIN_PATH)
    print()
    fix_dataset(TEST_PATH)
