"""
Trace the full execution flow from <image> tag to vLLM input.
This simulates exactly what verl does during training.
"""
import re
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoConfig

MODEL_NAME = "google/gemma-3-4b-it"
DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"

def _build_messages_verl_style(prompt_list, image_key='images', video_key='videos'):
    """
    Mimics verl's _build_messages() logic from rl_dataset.py
    """
    messages = prompt_list
    
    # Check if we have images (this check happens in verl)
    has_multimodal = True  # We know we have images
    
    if has_multimodal:
        for message in messages:
            content = message["content"]
            content_list = []
            
            # Split by <image> or <video> tags
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

def test_full_flow():
    print("=" * 80)
    print("STEP 1: Load dataset")
    print("=" * 80)
    df = pd.read_parquet(DATA_PATH)
    sample = df.iloc[0]
    
    raw_messages = sample['prompt']
    print(f"Raw prompt from dataset:")
    for msg in raw_messages:
        print(f"  [{msg['role']}]: {repr(msg['content'][:60])}...")
    
    # Check if <image> tag is present
    user_content = raw_messages[1]['content']
    has_image_tag = '<image>' in user_content
    print(f"\n✓ Contains <image> tag: {has_image_tag}")
    
    print("\n" + "=" * 80)
    print("STEP 2: verl's _build_messages() - Convert to structured format")
    print("=" * 80)
    # Deep copy and convert numpy array to list to avoid mutation issues
    import copy
    messages = list(copy.deepcopy(raw_messages))
    messages = _build_messages_verl_style(messages)
    print("Structured messages:")
    for msg in messages:
        print(f"  [{msg['role']}]:")
        if isinstance(msg['content'], list):
            for item in msg['content']:
                if item['type'] == 'image':
                    print(f"    - [IMAGE]")
                elif item['type'] == 'text':
                    print(f"    - TEXT: {repr(item['text'][:50])}...")
        else:
            print(f"    - {repr(msg['content'][:50])}...")
    
    print("\n" + "=" * 80)
    print("STEP 3: processor.apply_chat_template() - Generate prompt string")
    print("=" * 80)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(f"Raw prompt string (first 200 chars):\n{repr(raw_prompt[:200])}...")
    
    # Check if the Gemma 3 special token is in the string
    image_token_char = processor.image_token
    has_gemma3_token = image_token_char in raw_prompt
    print(f"\n✓ Contains Gemma 3 image token '{image_token_char}': {has_gemma3_token}")
    
    print("\n" + "=" * 80)
    print("STEP 4: processor() - Tokenize and expand")
    print("=" * 80)
    
    # Load image
    image_info = sample['images']
    image_url = image_info[0]['image']
    image_path = image_url.replace("file://", "")
    image = Image.open(image_path).convert("RGB")
    
    model_inputs = processor(text=[raw_prompt], images=image, return_tensors="pt")
    input_ids = model_inputs['input_ids'][0].tolist()
    
    print(f"Total token IDs: {len(input_ids)}")
    print(f"First 20 token IDs: {input_ids[:20]}")
    
    # Check for specific tokens
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    boi_id = config.boi_token_index  # 255999
    soft_id = config.image_token_index  # 262144
    eoi_id = config.eoi_token_index  # 256000
    
    has_boi = boi_id in input_ids
    has_soft = soft_id in input_ids
    has_eoi = eoi_id in input_ids
    count_soft = input_ids.count(soft_id)
    
    print(f"\n✓ Contains BOI ({boi_id}): {has_boi}")
    print(f"✓ Contains Soft tokens ({soft_id}): {has_soft} (count: {count_soft})")
    print(f"✓ Contains EOI ({eoi_id}): {has_eoi}")
    
    if not has_soft:
        print("⚠ WARNING: No soft tokens found! The processor didn't expand the image.")
        return False
    
    print("\n" + "=" * 80)
    print("STEP 5: _collapse_multimodal_tokens() - Collapse for vLLM")
    print("=" * 80)
    
    # Import the collapse function from vllm_rollout_spmd.py
    # Or implement it here
    def _collapse_multimodal_tokens(token_ids, model_hf_config):
        image_token_index = getattr(model_hf_config, "image_token_index", None)
        if image_token_index is None:
            return token_ids

        soft_token_id = image_token_index
        boi_token_id = getattr(model_hf_config, "boi_token_index", None)
        eoi_token_id = getattr(model_hf_config, "eoi_token_index", None)

        collapsed = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == soft_token_id:
                while i < len(token_ids) and token_ids[i] == soft_token_id:
                    i += 1
                if eoi_token_id is not None and i < len(token_ids) and token_ids[i] == eoi_token_id:
                    i += 1
                
                if boi_token_id is not None:
                    if not collapsed or collapsed[-1] != boi_token_id:
                        collapsed.append(boi_token_id)
            else:
                collapsed.append(token_ids[i])
                i += 1
        return collapsed
    
    collapsed_ids = _collapse_multimodal_tokens(input_ids, config)
    
    print(f"Original token count: {len(input_ids)}")
    print(f"Collapsed token count: {len(collapsed_ids)}")
    print(f"First 20 collapsed IDs: {collapsed_ids[:20]}")
    
    has_boi_collapsed = boi_id in collapsed_ids
    has_soft_collapsed = soft_id in collapsed_ids
    
    print(f"\n✓ Contains BOI ({boi_id}) after collapse: {has_boi_collapsed}")
    print(f"✓ Contains Soft tokens ({soft_id}) after collapse: {has_soft_collapsed}")
    
    if not has_boi_collapsed:
        print("❌ ERROR: BOI token missing after collapse! vLLM will fail.")
        return False
    
    if has_soft_collapsed:
        print("⚠ WARNING: Soft tokens still present after collapse!")
        return False
    
    print("\n" + "=" * 80)
    print("STEP 6: vLLM input format")
    print("=" * 80)
    
    vllm_input = {
        "prompt_token_ids": collapsed_ids,
        "multi_modal_data": {"image": image}
    }
    
    print(f"✓ vLLM will receive:")
    print(f"  - prompt_token_ids: {len(collapsed_ids)} tokens (contains BOI: {has_boi_collapsed})")
    print(f"  - multi_modal_data: 1 image of size {image.size}")
    
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if has_image_tag and has_gemma3_token and has_soft and has_boi_collapsed and not has_soft_collapsed:
        print("✅ ALL CHECKS PASSED - The flow should work with vLLM!")
        return True
    else:
        print("❌ SOME CHECKS FAILED - There may be issues")
        return False

if __name__ == "__main__":
    success = test_full_flow()
    exit(0 if success else 1)
