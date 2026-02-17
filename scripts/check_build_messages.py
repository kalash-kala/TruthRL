"""
Quick check: what does _build_messages actually do to our dataset?
"""
import re
import pandas as pd

DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"

def _build_messages_verl_style(example_dict, prompt_key='prompt', image_key='images', video_key='videos'):
    """
    Exact copy from rl_dataset.py lines 190-209
    """
    messages = example_dict.pop(prompt_key)
    
    if image_key in example_dict or video_key in example_dict:
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
sample = df.iloc[0].to_dict()

print("BEFORE _build_messages:")
for msg in sample['prompt']:
    print(f"  [{msg['role']}]: {repr(msg['content'][:60])}")

print(f"\nHas 'images' key: {'images' in sample}")

messages = _build_messages_verl_style(sample.copy())

print("\nAFTER _build_messages:")
for msg in messages:
    print(f"  [{msg['role']}]:")
    if isinstance(msg['content'], list):
        for item in msg['content']:
            print(f"    {item}")
    else:
        print(f"    {repr(msg['content'][:60])}")
