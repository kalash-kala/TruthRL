import torch
from transformers import AutoProcessor, AutoConfig
from PIL import Image
import os

# CONFIG
MODEL_NAME = "google/gemma-3-4b-it"
IMAGE_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/images/000000460783.jpg"

def diagnose():
    print(f"--- 1. Model Configuration ---")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print(f"Image Token Index (Config): {config.image_token_index}")
    print(f"Image Seq Length (Processor): {processor.image_seq_length}")
    print(f"Image Size (Processor): {processor.image_processor.size}")
    
    # Check Pan and Scan settings
    ip = processor.image_processor
    print(f"\n--- 2. Pan and Scan (Processor Defaults) ---")
    print(f"do_pan_and_scan: {getattr(ip, 'do_pan_and_scan', 'N/A')}")
    print(f"pan_and_scan_max_num_crops: {getattr(ip, 'pan_and_scan_max_num_crops', 'N/A')}")
    print(f"pan_and_scan_min_crop_size: {getattr(ip, 'pan_and_scan_min_crop_size', 'N/A')}")

    print(f"\n--- 3. Tokenization Test ---")
    # Using the special token found earlier
    prompt = [{"role": "user", "content": f"{processor.image_token}\nTest"}]
    formatted = processor.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Real Image Size: {image.size} (WxH)")

    inputs = processor(text=formatted, images=image, return_tensors="pt")
    
    ids = inputs['input_ids'][0]
    total_len = len(ids)
    
    # We are looking for the placeholder tokens that vLLM might be counting
    soft_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    
    soft_count = (ids == soft_token_id).sum().item()
    img_count = (ids == image_token_id).sum().item()
    
    print(f"Input IDs Total Length: {total_len}")
    print(f"Special Image Token ('{processor.image_token}') Count: {img_count} (Expected 1)")
    print(f"Soft Token ('<image_soft_token>') Count: {soft_count} (Expected 256 or 512)")
    
    # Analysis
    if soft_count == 256:
        print("\nDIAGNOSIS: Processor produced 256 tokens. vLLM is likely expecting 512 because it thinks this image size requires a crop.")
    elif soft_count == 512:
        print("\nDIAGNOSIS: Processor produced 512 tokens. If vLLM crashes here, it might be looking for a different token ID.")

if __name__ == "__main__":
    diagnose()