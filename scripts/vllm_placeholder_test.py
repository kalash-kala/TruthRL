
import os
import torch
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
IMAGE_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/images/000000460783.jpg"

def vllm_placeholder_test():
    print(f"Loading tokenizer and processor for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Gemma 3 boi_token is typically ID 255999
    # We want to put THIS ID as a single placeholder in the input_ids.
    placeholder_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    print(f"Found placeholder token: {processor.image_token} (ID: {placeholder_id})")

    # Construct a simple prompt manually
    # <bos><start_of_turn>user\n[placeholder]\nDescribe this image.<end_of_turn><start_of_turn>model\n
    text = f"Describe this image."
    messages = [{"role": "user", "content": f"{processor.image_token}\n{text}"}]
    
    # We use the TOKENIZER (not processor) to get the baseline IDs.
    # The tokenizer won't expand the image into 256 tokens.
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    
    print(f"\nBaseline Input IDs length: {len(input_ids)}")
    print(f"Placeholder ID {placeholder_id} count: {input_ids.count(placeholder_id)}")
    
    # Initialize vLLM
    print(f"\nInitializing vLLM...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        enforce_eager=True
    )
    
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("\n--- Running Inference with Placeholder IDs ---")
    vllm_input = {
        "prompt_token_ids": input_ids,
        "multi_modal_data": {"image": image}
    }
    
    try:
        outputs = llm.generate(vllm_input, sampling_params=sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"\nSUCCESS! vLLM Response: '{generated_text.strip()}'")
    except Exception as e:
        print(f"\nFAILED: {e}")

if __name__ == "__main__":
    vllm_placeholder_test()
