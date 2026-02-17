"""
Simple check: what are the roles in the processed messages?
"""
import copy
import re
import pandas as pd
from transformers import AutoProcessor

MODEL_NAME = "google/gemma-3-4b-it"
DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"

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

df = pd.read_parquet(DATA_PATH)
sample = df.iloc[0]

raw_messages = sample['prompt']
print("Original messages:")
for i, msg in enumerate(raw_messages):
    print(f"  {i}: role={msg['role']}, content type={type(msg['content'])}")

# Deep copy to avoid mutation
messages = copy.deepcopy(raw_messages)
processed_messages = _build_messages_verl_style(messages)

print("\nProcessed messages:")
for i, msg in enumerate(processed_messages):
    print(f"  {i}: role={msg['role']}, content type={type(msg['content'])}")

print("\nTrying apply_chat_template...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

try:
    prompt = processor.apply_chat_template(processed_messages, add_generation_prompt=True, tokenize=False)
    print("✓ SUCCESS!")
    print(f"Prompt preview: {repr(prompt[:150])}...")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nDebugging - message roles sequence:")
    roles = [msg['role'] for msg in processed_messages]
    print(f"Roles: {roles}")
