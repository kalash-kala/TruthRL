"""
Test if Gemma 3 chat template supports structured content for system messages
"""
from transformers import AutoProcessor

MODEL_NAME = "google/gemma-3-4b-it"

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Test 1: Simple string content (should  work)
print("=" * 60)
print("Test 1: String content for all messages")
print("=" * 60)
messages1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]

try:
    prompt1 = processor.apply_chat_template(messages1, add_generation_prompt=True, tokenize=False)
    print("✓ SUCCESS")
    print(f"Prompt: {repr(prompt1[:100])}...")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Structured content for system message
print("\n" + "=" * 60)
print("Test 2: Structured content for system message")
print("=" * 60)
messages2 = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": "What is 2+2?"}
]

try:
    prompt2 = processor.apply_chat_template(messages2, add_generation_prompt=True, tokenize=False)
    print("✓ SUCCESS")
    print(f"Prompt: {repr(prompt2[:100])}...")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Structured content for user message only
print("\n" + "=" * 60)
print("Test 3: Structured content for user message with image")
print("=" * 60)
messages3 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "\nWhat is this?"}]}
]

try:
    prompt3 = processor.apply_chat_template(messages3, add_generation_prompt=True, tokenize=False)
    print("✓ SUCCESS")
    print(f"Prompt: {repr(prompt3[:150])}...")
    print(f"Contains image token: {processor.image_token in prompt3}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 4: Both structured
print("\n" + "=" * 60)
print("Test 4: Structured content for BOTH messages")
print("=" * 60)
messages4 = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "\nWhat is this?"}]}
]

try:
    prompt4 = processor.apply_chat_template(messages4, add_generation_prompt=True, tokenize=False)
    print("✓ SUCCESS")
    print(f"Prompt: {repr(prompt4[:150])}...")
    print(f"Contains image token: {processor.image_token in prompt4}")
except Exception as e:
    print(f"❌ FAILED: {e}")
