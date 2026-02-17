
from transformers import AutoProcessor, AutoTokenizer

MODEL_NAME = "google/gemma-3-4b-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = processor.tokenizer

print(f"Processor Image Token: {repr(processor.image_token)}")

# Test 1: Literal text with special char
messages_text = [
    {"role": "user", "content": f"{processor.image_token}\nThe dog is under the bowl."}
]
prompt_text = processor.apply_chat_template(messages_text, add_generation_prompt=True, tokenize=False)
print(f"\n--- Test 1 (Literal Text) ---\n{repr(prompt_text)}")

tokens_text = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
print(f"Token IDs: {tokens_text[:10]}...")
print(f"Contains 255999? {255999 in tokens_text}")
print(f"Contains 262144? {262144 in tokens_text}")


# Test 2: Structured content (verl style)
# Note: verl converts <image> to {"type": "image"}
messages_struct = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "\nThe dog is under the bowl."}
    ]}
]

try:
    prompt_struct = processor.apply_chat_template(messages_struct, add_generation_prompt=True, tokenize=False)
    print(f"\n--- Test 2 (Structured) ---\n{repr(prompt_struct)}")
    
    tokens_struct = tokenizer(prompt_struct, add_special_tokens=False)['input_ids']
    print(f"Token IDs: {tokens_struct[:10]}...")
    print(f"Contains 255999? {255999 in tokens_struct}")
    print(f"Contains 262144? {262144 in tokens_struct}")
except Exception as e:
    print(f"\n--- Test 2 (Structured) FAILED ---\n{e}")
