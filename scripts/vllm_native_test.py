
import os
import torch
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
IMAGE_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/images/000000460783.jpg"

def vllm_native_test():
    print(f"Loading processor for {MODEL_NAME} to get special tokens...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # We will try all possible ways to signal an image
    formats = [
        ("Special Token (processor.image_token)", f"{processor.image_token}\nDescribe this."),
        ("BOI Token", f"{processor.boi_token}\nDescribe this."),
        ("String <image>", "<image>\nDescribe this."),
        ("String [image]", "[image]\nDescribe this."),
    ]

    print(f"\nInitializing vLLM...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        enforce_eager=True
    )
    
    sampling_params = SamplingParams(temperature=0, max_tokens=20)
    image = Image.open(IMAGE_PATH).convert("RGB")

    for name, prompt in formats:
        print(f"\n--- Testing Format: {name} ---")
        print(f"Prompt preview: {repr(prompt[:50])}...")
        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
        
        try:
            outputs = llm.generate(vllm_input, sampling_params=sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
                print(f"SUCCESS! vLLM Response: '{generated_text.strip()}'")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    vllm_native_test()
