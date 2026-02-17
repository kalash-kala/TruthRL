import requests
import json

# Configuration
API_URL = "http://35.198.251.55:8000/v1/chat/completions"
API_KEY = "token-abc123"
MODEL_ID = "google/gemma-3-27b-it"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# The prompt/question you want to ask
messages = [
    {"role": "user", "content": "what is the capital of India?"}
]

payload = {
    "model": MODEL_ID,
    "messages": messages,
    "temperature": 0.7,
    "max_tokens": 512
}

try:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    
    # Extract and print the answer
    content = result['choices'][0]['message']['content']
    print("\n--- Response ---\n")
    print(content)
    print("\n----------------\n")

except requests.exceptions.RequestException as e:
    print(f"Error calling API: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(e.response.text)
