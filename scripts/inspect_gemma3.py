
from transformers import AutoModelForCausalLM
import torch

model_name = "google/gemma-3-4b-it"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)

for name, module in model.named_modules():
    if "SiglipMultiheadAttentionPoolingHead" in str(type(module)):
        print(f"Found at: {name}")
        print(f"Class: {type(module)}")

def check_attr(m, indent=''):
    for attr in dir(m):
        if not attr.startswith('_') and "SiglipMultiheadAttentionPoolingHead" in str(getattr(m, attr, '')):
             print(f"{indent}Attr {attr} might contain it")

check_attr(model)
check_attr(model.vision_tower, '  ')
