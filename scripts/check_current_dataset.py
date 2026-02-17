#!/usr/bin/env python3
import pandas as pd

df = pd.read_parquet('/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet')
msg = df.iloc[0]['prompt'][1]

print(f"User message content: {repr(msg['content'])}")
print(f"\nContains '<image>': {'<image>' in msg['content']}")
print(f"Contains '🖼️': {'🖼️' in msg['content']}")
print(f"Contains '<start_of_image>': {'<start_of_image>' in msg['content']}")
