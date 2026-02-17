"""
Print the exact structure loaded from parquet (handling numpy arrays)
"""
import copy
import re
import pandas as pd

DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"

df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

raw_messages = sample['prompt']

print(f"Type of raw_messages: {type(raw_messages)}")
print(f"Length: {len(raw_messages)}")

print("\nRaw messages:")
for i, msg in enumerate(raw_messages):
    print(f"\nMessage {i}:")
    print(f"  Type: {type(msg)}")
    print(f"  Keys: {msg.keys() if hasattr(msg, 'keys') else 'N/A'}")
    if hasattr(msg, 'keys'):
        for key, value in msg.items():
            print(f"  {key}: {repr(value)[:100]}")
