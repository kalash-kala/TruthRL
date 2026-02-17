
from transformers import AutoProcessor

MODEL_NAME = "google/gemma-3-4b-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = processor.tokenizer

image_token = processor.image_token
print(f"Processor Image Token: {repr(image_token)}")

# Test 3: Structured content but only Type: Text containing the token
# This simulates what verl does if it fails to split on <image>
messages_text_struct = [
    {"role": "user", "content": [
        {"type": "text", "text": f"{image_token}\nThe dog is under the bowl."}
    ]}
]

print("\n--- Test 3 (Structured Text-Only) ---")
try:
    prompt_text_struct = processor.apply_chat_template(messages_text_struct, add_generation_prompt=True, tokenize=False)
    print(f"Prompt: {repr(prompt_text_struct)}")
    
    # Check if token survived
    if image_token in prompt_text_struct:
        print("SUCCESS: Image token survived.")
    else:
        print("FAILURE: Image token vanished!")
        
    tokens = tokenizer(prompt_text_struct, add_special_tokens=False)['input_ids']
    print(f"Token IDs: {tokens[:10]}...")
    print(f"Contains 255999? {255999 in tokens}")

except Exception as e:
    print(f"ERROR: {e}")
