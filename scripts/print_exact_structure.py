"""
Print the exact structure loaded from parquet
"""
import copy
import re
import pandas as pd
import json

import numpy as np

def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

DATA_PATH = ["/home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test.parquet", 
"/home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"]

for data_path in DATA_PATH:
    print("\n" + "="*60)
    print(data_path.split("/")[-1])
    print("="*60)
    df = pd.read_parquet(data_path)
    sample = df.iloc[0]

    raw_messages = sample['prompt']

    print("Raw messages as loaded from parquet:")
    print(json.dumps(make_serializable(raw_messages), indent=2))

    print("\n" + "="*60)
    print("After _build_messages processing:")
    print("="*60)

    def _build_messages_verl_style(prompt_list):
        messages = prompt_list
        
        for message in messages:
            content = message["content"]
            content_list = []
            
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            
            for segment in segments:
                if segment == "<image>":
                    content_list.append({"type": "image"})
                elif segment == "<video>":
                    content_list.append({"type": "video"})
                else:
                    content_list.append({"type": "text", "text": segment})
            
            message["content"] = content_list
        
        return messages

    messages = copy.deepcopy(raw_messages)
    processed = _build_messages_verl_style(messages)

    print(json.dumps(make_serializable(processed), indent=2))
