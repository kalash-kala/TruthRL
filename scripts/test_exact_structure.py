"""
Exact replication of the dataset structure
"""
from transformers import AutoProcessor

MODEL_NAME = "google/gemma-3-4b-it"

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# System + User with image (exactly like our dataset after _build_messages)
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a visual spatial reasoning expert. Analyze the image and the statement. Answer exactly 'True', 'False', or 'I don't know'. Do not provide explanations."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "\nThe dog is under the bowl."}
        ]
    }
]

print("Messages:")
for msg in messages:
    print(f"  Role: {msg['role']}")
    print(f"  Content: {msg['content']}")

print("\nCalling apply_chat_template...")
try:
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print("✓ SUCCESS!")
    print(f"\nPrompt:\n{prompt}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
