
import os
import torch
from transformers import AutoTokenizer, AutoProcessor
from verl.utils.dataset.rl_dataset import RLHFDataset
from omegaconf import DictConfig
import numpy as np

def dry_run_dataloader():
    model_name = "google/gemma-3-4b-it"
    parquet_path = "/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"
    
    print(f"Loading tokenizer and processor for {model_name}...")
    # Using trust_remote_code=True for Gemma 3
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Verl RLHFDataset expected config keys
    data_config = DictConfig({
        "train_files": parquet_path,
        "max_prompt_length": 1024,
        "max_response_length": 512,
        "truncation": "right",
        "reward_fn_key": "ground_truth",
        "image_key": "images",      # Column we just added
        "prompt_key": "prompt",
        "return_multi_modal_inputs": True
    })

    print(f"Initializing RLHFDataset with {parquet_path}...")
    dataset = RLHFDataset(
        data_files=[parquet_path],
        tokenizer=tokenizer,
        config=data_config,
        processor=processor
    )

    print(f"Dataset size: {len(dataset)}")
    
    # Load first sample
    print("\nProcessing first sample...")
    item = dataset[0]
    
    print("\n--- Verification Results ---")
    
    # 1. Check for Pads
    input_ids = item['input_ids']
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    
    # Strip left padding for cleaner display
    non_pad_indices = np.where(input_ids != tokenizer.pad_token_id)[0]
    if len(non_pad_indices) > 0:
        actual_content_ids = input_ids[non_pad_indices[0]:]
        decoded_text = tokenizer.decode(actual_content_ids, skip_special_tokens=False)
        print(f"\nDecoded Text (Pads Stripped):\n{decoded_text}")
    else:
        print("\nWARNING: Entire prompt consists of pad tokens!")

    # 2. Check Multimodal Inputs
    if "multi_modal_inputs" in item:
        mm_inputs = item["multi_modal_inputs"]
        print(f"\nMultimodal Inputs detected!")
        for k, v in mm_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Check for pixel_values (the images)
        if 'pixel_values' in mm_inputs:
             print("\nSUCCESS: Pixel values (images) are present in the batch.")
        else:
             print("\nWARNING: pixel_values missing from multi_modal_inputs.")
             print(f"Available keys: {list(mm_inputs.keys())}")
    else:
        print("\nERROR: multi_modal_inputs key missing from sample.")

    # 3. Check extra metadata
    print(f"\nReward Target: {item.get('reward_model')}")

if __name__ == "__main__":
    if not os.path.exists("/home/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/train.parquet"):
        print("Error: Parquet file not found. Run preprocessing first.")
    else:
        dry_run_dataloader()
