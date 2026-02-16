
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer, AutoConfig

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"

def _collapse_multimodal_tokens(token_ids: list[int], model_hf_config) -> list[int]:
    """Collapse expanded image soft tokens back to the single placeholder token."""
    image_token_index = getattr(model_hf_config, "image_token_index", None)
    if image_token_index is None:
        return token_ids

    # Gemma 3 specific token indices
    soft_token_id = image_token_index
    boi_token_id = getattr(model_hf_config, "boi_token_index", None)
    eoi_token_id = getattr(model_hf_config, "eoi_token_index", None)

    collapsed = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == soft_token_id:
            # Found soft tokens (image features).
            # Consume the entire block.
            while i < len(token_ids) and token_ids[i] == soft_token_id:
                i += 1
            # Consume EOI if present.
            if eoi_token_id is not None and i < len(token_ids) and token_ids[i] == eoi_token_id:
                i += 1
            
            # CRITICAL FIX: Ensure the single BOI placeholder (255999) is present.
            # If inserted.
            if boi_token_id is not None:
                if not collapsed or collapsed[-1] != boi_token_id:
                    collapsed.append(boi_token_id)
        else:
            collapsed.append(token_ids[i])
            i += 1
    return collapsed

def vllm_dry_run():
    print(f"Loading processor for {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load image from dataset
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    sample = df.iloc[0]
    image_info = sample['images']
    image_url = image_info[0]['image']
    image_path = image_url.replace("file://", "")
    image = Image.open(image_path).convert("RGB")
    print("Image path: ", image_path)
    print(f"Image size: {image.size}")
    
    # Initialize vLLM with bfloat16 (Gemma 3's native dtype) and pan_and_scan OFF
    print("\nInitializing vLLM (dtype=bfloat16, pan_and_scan=OFF)...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        mm_processor_kwargs={"do_pan_and_scan": False}
    )
    
    sampling_params = SamplingParams(temperature=0, max_tokens=16)

    # ===== TEST 1: Text-only (sanity check) =====
    print("\n" + "="*60)
    print("TEST 1: Text-only inference (no image)")
    print("="*60)

    try:
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text_outputs = llm.generate(
            formatted_prompt,
            sampling_params=sampling_params
        )
        for out in text_outputs:
            print(f"Response: '{out.outputs[0].text}'")
    except Exception as e:
        print(f"ERROR: {e}")

    # ===== TEST 2: Multimodal with STRING prompt (let vLLM tokenize) =====
    print("\n" + "="*60)
    print("TEST 2: Multimodal with string prompt (vLLM tokenizes)")
    print("="*60)
    try:
        mm_string_input = {
            "prompt": "<bos><start_of_turn>user\n<start_of_image>\nWhat is in this image? Describe it briefly.<end_of_turn>\n<start_of_turn>model\n",
            "multi_modal_data": {"image": image}
        }
        mm_outputs = llm.generate(mm_string_input, sampling_params=sampling_params)
        for out in mm_outputs:
            print(f"Response IDs: {out.outputs[0].token_ids[:20]}...")
            print(f"Response: '{out.outputs[0].text}'")
    except Exception as e:
        print(f"ERROR: {e}")

    # ===== TEST 3: Multimodal with collapsed token IDs (verl path) =====
    print("\n" + "="*60)
    print("TEST 3: Multimodal with collapsed token IDs (verl path)")
    print("="*60)
    try:

        messages = [
            {'content': "You are a visual spatial reasoning expert. Analyze the image and the statement. Answer exactly 'True', 'False', or 'I don't know'. Do not provide explanations.", 'role': 'system'},
            {'content': f'{processor.image_token}<\nThe dog is under the bowl.', 'role': 'user'}
        ]
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = processor(text=[raw_prompt], images=image, return_tensors="pt")
        input_ids = model_inputs['input_ids'][0].tolist()
        collapsed_ids = _collapse_multimodal_tokens(input_ids, config)
        print(f"Original IDs: {len(input_ids)} -> Collapsed: {len(collapsed_ids)}")
        
        vllm_input = {
            "prompt_token_ids": collapsed_ids,
            "multi_modal_data": {"image": image}
        }
        outputs = llm.generate(vllm_input, sampling_params=sampling_params)
        for out in outputs:
            print(f"Response IDs: {out.outputs[0].token_ids[:20]}...")
            print(f"Response: '{out.outputs[0].text}'")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    vllm_dry_run()
