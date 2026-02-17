import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_name = "google/gemma-3-4b-it"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

print("Listing all module names:")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"Linear: {name}")
