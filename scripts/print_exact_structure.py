"""
Print the exact structure loaded from parquet
"""
import copy
import re
import pandas as pd
import json

DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"

df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

raw_messages = sample['prompt']

print("Raw messages as loaded from parquet:")
print(json.dumps(raw_messages, indent=2))

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

print(json.dumps(processed, indent=2))
