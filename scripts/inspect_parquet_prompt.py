
import pandas as pd
from transformers import AutoProcessor

DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"
MODEL_NAME = "google/gemma-3-4b-it"

print(f"Reading {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

print("\n--- Prompt Messages ---")
for msg in sample['prompt']:
    print(f"Role: {msg['role']}")
    print(f"Content: {repr(msg['content'])}")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"\nExpected Image Token: {repr(processor.image_token)}")

# Check if token is in content
has_token = processor.image_token in sample['prompt'][1]['content']
print(f"\nIs image token present? {has_token}")
