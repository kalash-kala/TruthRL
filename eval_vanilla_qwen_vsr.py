
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = "/data/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
TEST_DATA_PATH = "/data/visual-spatial-reasoning-final/truthrl-sample/parquet/test.parquet"
OUTPUT_FILE = "vanilla_qwen_vsr_results.csv"
BATCH_SIZE = 1  # Keeping it safe for inference

# -----------------------------------------------------------------------------
# Load Model & Processor
# -----------------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
print(f"Loading test data from {TEST_DATA_PATH}...")
df = pd.read_parquet(TEST_DATA_PATH)

# Ensure the dataframe has the expected columns
# Based on previous analysis, we expect 'images' (list of paths), 'prompt' (list of messages), and 'label' (True/False)
# If 'label' is missing, we might need to parse it from somewhere else, but let's assume it's there or in the prompt.
# Actually, the parquet file had 'prompt' which was a list of dicts (messages).
# We need to extract the ground truth if it's not a separate column.
# Let's inspect the first row structure briefly to be sure.
# For now, we'll assume standard VSR structure where we can extract the ground truth.

def extract_ground_truth(row):
    # If there is a dedicated label column, use it.
    if 'label' in row:
        return str(row['label'])
    # Otherwise, try to find it in the reward_model metadata if it exists
    if 'reward_model' in row and 'ground_truth' in row['reward_model']:
        return str(row['reward_model']['ground_truth'])
    return None

# -----------------------------------------------------------------------------
# Inference Loop
# -----------------------------------------------------------------------------
results = []
print("Starting inference...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    # 1. Prepare Inputs
    # The 'prompt' column is already a list of messages [system, user, assistant...]
    # We need to ensure the image path is correct.
    
    messages = list(row['prompt']) # Make a copy
    
    # We need to ensure the image path is absolute or correct relative to execution.
    # The 'images' column contains a list of paths.
    image_paths = row['images']
    
    # Check if we need to fix image paths (e.g. if they are just filenames)
    # Assuming paths in parquet are correct for now, or we might need a base path.
    # Based on previous scripts, they might be absolute or relative. 
    # Let's assume they work as is for now, but we'll print one to debug if it fails.

    # 2. Extract Ground Truth (for reference)
    ground_truth = extract_ground_truth(row)

    # 3. Processing
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to("cuda")

    # 4. Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 5. Store Result
    results.append({
        "index": index,
        "ground_truth": ground_truth,
        "prediction": output_text.strip(),
        "prompt": text,
        "image_path": image_paths[0] if len(image_paths) > 0 else None
    })

# -----------------------------------------------------------------------------
# Save Results & Calculate Accuracy
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}")

# Simple accuracy check
# VSR is usually True/False. We need to normalize.
def normalize_text(text):
    text = str(text).lower().strip()
    if "true" in text: return "true"
    if "false" in text: return "false"
    return text

correct_count = 0
valid_count = 0

for _, row in results_df.iterrows():
    gt = normalize_text(row['ground_truth'])
    pred = normalize_text(row['prediction'])
    
    if gt in ["true", "false"]: # Only count if we have a valid GT
        valid_count += 1
        if gt == pred:
            correct_count += 1

if valid_count > 0:
    accuracy = correct_count / valid_count
    print(f"\nVanilla Model Accuracy: {accuracy:.4f} ({correct_count}/{valid_count})")
else:
    print("\nCould not calculate accuracy (no valid ground truth found). Check the output CSV.")
