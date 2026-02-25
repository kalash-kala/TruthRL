import pandas as pd
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import copy
import ast

model_path = '/home/debarpanb1/models/Qwen2.5-VL-3B-Instruct'
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
data_path = '/home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test.parquet'
df = pd.read_parquet(data_path)
row = df.iloc[0]

messages = list(row['prompt'])
image_path = 'file:///home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/images/000000224306.jpg'

for msg in messages:
    if msg['role'] == 'user':
        orig_content = msg['content']
        clean_text = orig_content.replace('<image>\n', '').replace('<image>', '').strip()
        msg['content'] = [
            {'type': 'image', 'image': image_path},
            {'type': 'text', 'text': clean_text}
        ]

print('--- MESSAGES ---')
print(messages)

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print('\n--- APPLIED CHAT TEMPLATE ---')
print(text)

image_inputs, video_inputs = process_vision_info(messages)
print('\n--- VISION INFO ---')
print('image_inputs length:', len(image_inputs) if image_inputs else 0)

