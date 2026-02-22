#!/usr/bin/env python3
import json
import os
import urllib.request
import hashlib
import pandas as pd
import requests
from tqdm import tqdm

# --- Configuration ---
METADATA_PATH = "/home/debarpanb1/kalashkala/vsr_dataset/dataset_infos.json"
BASE_DIR = "/home/debarpanb1/kalashkala/visual-spatial-reasoning"
SAMPLE_DIR = os.path.join(BASE_DIR, "truthrl-sample/data")
IMAGE_DIR = os.path.join(BASE_DIR, "truthrl-sample/images")
SEED = 42

def get_checksum(file_path):
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def download_jsonl_from_metadata():
    """Step 1: Download JSONL files from GitHub using the HF metadata."""
    print("--- STEP 1: Downloading JSONL files ---")
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return False

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    for config_name, config_data in metadata.items():
        if not isinstance(config_data, dict) or "download_checksums" not in config_data:
            continue
            
        print(f"\nProcessing Configuration: {config_name}")
        config_dir = os.path.join(BASE_DIR, config_name)
        os.makedirs(config_dir, exist_ok=True)
            
        checksums = config_data.get("download_checksums", {})
        for url, meta in checksums.items():
            filename = url.split('/')[-1]
            output_file = os.path.join(config_dir, filename)
            expected_checksum = meta.get("checksum")
            
            if os.path.exists(output_file):
                if get_checksum(output_file) == expected_checksum:
                    print(f"  [Skipping] {filename} (Already exists and verified)")
                    continue
                
            print(f"  [Downloading] {url} -> {output_file}")
            try:
                urllib.request.urlretrieve(url, output_file)
                if expected_checksum and get_checksum(output_file) != expected_checksum:
                    print(f"    WARNING: Checksum mismatch for {filename}")
            except Exception as e:
                print(f"    Failed to download: {e}")
    return True

def sample_dataset():
    """Step 2: Sample and balance the dataset using pandas."""
    print("\n--- STEP 2: Sampling and balancing dataset ---")
    paths = {
        'test': os.path.join(BASE_DIR, 'random/test.jsonl'),
        'train': os.path.join(BASE_DIR, 'random/train.jsonl')
    }

    for p in paths.values():
        if not os.path.exists(p):
            print(f"Error: Required file missing: {p}")
            return False

    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Process Train
    df_train = pd.read_json(paths['train'], lines=True)
    df_train_0 = df_train[df_train['label'] == 0].sample(n=375, random_state=SEED)
    df_train_1 = df_train[df_train['label'] == 1].sample(n=375, random_state=SEED)
    df_train_sampled = pd.concat([df_train_0, df_train_1]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_out = os.path.join(SAMPLE_DIR, 'train_sampled.jsonl')
    df_train_sampled.to_json(train_out, orient='records', lines=True)
    print(f"Saved balanced train sample to: {train_out}")

    # Process Test
    df_test = pd.read_json(paths['test'], lines=True)
    df_test_0 = df_test[df_test['label'] == 0].sample(n=375, random_state=SEED)
    df_test_1 = df_test[df_test['label'] == 1].sample(n=375, random_state=SEED)
    df_test_sampled = pd.concat([df_test_0, df_test_1]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_out = os.path.join(SAMPLE_DIR, 'test_sampled.jsonl')
    df_test_sampled.to_json(test_out, orient='records', lines=True)
    print(f"Saved balanced test sample to: {test_out}")
    
    return True

def download_sampled_images():
    """Step 3: Download images referred to in the sampled JSONL files."""
    print("\n--- STEP 3: Downloading images ---")
    jsonl_files = [
        os.path.join(SAMPLE_DIR, "test_sampled.jsonl"),
        os.path.join(SAMPLE_DIR, "train_sampled.jsonl")
    ]
    
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    image_tasks = {}
    for file_path in jsonl_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                name, link = data.get('image'), data.get('image_link')
                if name and link:
                    image_tasks[name] = link

    print(f"Found {len(image_tasks)} unique images to download.")
    
    for name, link in tqdm(image_tasks.items(), desc="Downloading"):
        target = os.path.join(IMAGE_DIR, name)
        if os.path.exists(target):
            continue
            
        try:
            response = requests.get(link, timeout=15)
            if response.status_code == 200:
                with open(target, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Error {name}: {e}")

if __name__ == "__main__":
    if download_jsonl_from_metadata():
        if sample_dataset():
            download_sampled_images()
            print("\nWorkflow Complete!")