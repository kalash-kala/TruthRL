
import os
import pandas as pd
from PIL import Image
from transformers import AutoProcessor
import torch

# Configuration
MODEL_NAME = "google/gemma-3-4b-it"
DATA_PATH = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"

def dry_run():
    print(f"Loading processor for {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    sample = df.iloc[0]
    
    prompt = sample['prompt']
    image_info = sample['images']
    
    # Replace literal '<image>' with the special token required by the processor
    for msg in prompt:
        if isinstance(msg['content'], str):
            msg['content'] = msg['content'].replace("<image>", processor.image_token)
    
    print("\n--- Corrected Prompt Structure ---")
    for msg in prompt:
        print(f"[{msg['role']}]: {msg['content']}")
        
    # Extract image path
    image_url = image_info[0]['image']
    image_path = image_url.replace("file://", "")
    
    print(f"\nLoading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"ERROR: Image path {image_path} does not exist!")
        return
        
    image = Image.open(image_path).convert("RGB")
    
    # Process inputs
    # Note: Gemma 3 uses a chat template that we should apply
    print("\nApplying chat template...")
    formatted_prompt = processor.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    print(f"Formatted Prompt:\n{formatted_prompt}")
    
    print("\nProcessing inputs...")
    try:
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to("cuda", torch.bfloat16)
        print("Processor SUCCESS: Inputs created and moved to GPU.")
    except Exception as e:
        print(f"Processor FAILED: {e}")
        return

    # Load Model for Inference
    from transformers import Gemma3ForConditionalGeneration
    print(f"Loading model {MODEL_NAME} for inference test...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    print("\n--- Input Analysis ---")
    # THE CORRECT TOKEN CHECK:
    # We should check for the ID of processor.image_token (the 🖼️ character)
    # Based on previous investigation, this is 255999
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    print(f"Special Image Token ('{processor.image_token}') ID: {image_token_id}")
    
    input_ids = inputs['input_ids']
    count = (input_ids == image_token_id).sum().item()
    print(f"Number of special image tokens in input_ids: {count}")

    if count > 0:
        print("SUCCESS: Special image token is present.")
    else:
        print("WARNING: Special image token is STILL MISSING in input_ids!")

    print("\nRunning Inference...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)
    
    # Decode the response
    # We only want the generated part
    prompt_len = inputs['input_ids'].shape[1]
    response_ids = output[0][prompt_len:]
    response_text = processor.decode(response_ids, skip_special_tokens=True)
    
    print(f"\nModel Response: '{response_text.strip()}'")
    
    if response_text.strip() in ["True", "False"]:
         print("INFRENECE SUCCESS: Model generated a valid VSR answer.")
    else:
         print(f"INFERENCE NOTE: Model generated '{response_text.strip()}'. Check if this makes sense for the image.")
        
    # Check for image token
    # Gemma 3 image token index is usually specific. 
    # Let's find the token for '<image>'
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    print(f"Image token ('<image>') ID: {image_token_id}")
    
    count = (inputs['input_ids'] == image_token_id).sum().item()
    print(f"Number of '<image>' tokens in input_ids: {count}")
    
    if count == 0:
        print("WARNING: No <image> tokens found in input_ids!")
    else:
        print("SUCCESS: <image> token is present in the tokenized input.")

if __name__ == "__main__":
    dry_run()
