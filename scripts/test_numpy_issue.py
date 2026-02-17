"""
Test if numpy array causes issues with apply_chat_template
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
print(f"Type before deepcopy: {type(raw_messages)}")

# Deep copy to avoid mutation AND convert numpy array to list
messages = copy.deepcopy(raw_messages)
messages_list = list(messages) if hasattr(messages, '__iter__') else messages

print(f"Type after deepcopy: {type(messages_list)}")
print(f"First message type: {type(messages_list[0])}")

processed = _build_messages_verl_style(messages_list)

print(f"\nProcessed messages type: {type(processed)}")
print(f"Processed first message: {processed[0]}")
print(f"Processed second message content: {processed[1]['content']}")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("\nTrying apply_chat_template...")
try:
    prompt = processor.apply_chat_template(processed, add_generation_prompt=True, tokenize=False)
    print("✓ SUCCESS!")
    print(f"Prompt contains image token: {processor.image_token in prompt}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    
    # Try converting to plain list of dicts
    print("\nRetrying with explicit list conversion...")
    plain_list = [dict(msg) for msg in processed]
    try:
        prompt2 = processor.apply_chat_template(plain_list, add_generation_prompt=True, tokenize=False)
        print("✓ SUCCESS with plain list!")
        print(f"Prompt contains image token: {processor.image_token in prompt2}")
    except Exception as e2:
        print(f"❌ Still failed: {e2}")
