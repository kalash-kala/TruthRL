#!/usr/bin/env python3
"""
Standalone script to download the Visual Spatial Reasoning dataset.
This script reads the metadata from your local Hugging Face cache but downloads the raw data files
directly from the source URLs, bypassing the need for the 'datasets' library or missing cache blobs.
"""

import json
import os
import urllib.request
import hashlib
import glob

def get_checksum(file_path):
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return None

def main():
    # Base path where huggingface stores the repo in your cache
    base_path = "/home/kalashkala/.cache/huggingface/hub/datasets--juletxara--visual-spatial-reasoning"
    
    print(f"Scanning for metadata in: {base_path}")
    
    # Locate snapshots directory
    snapshots_path = os.path.join(base_path, "snapshots")
    info_path = None
    
    # Find dataset_infos.json in any snapshot subdirectory
    if os.path.isdir(snapshots_path):
        # Sort to ensure consistent order
        for snap in sorted(glob.glob(os.path.join(snapshots_path, "*"))):
            if os.path.isdir(snap):
                candidate = os.path.join(snap, "dataset_infos.json")
                if os.path.exists(candidate):
                    info_path = candidate
                    break
    
    if not info_path:
        print(f"Error: Could not find 'dataset_infos.json' in {snapshots_path}")
        print("This file is required to find the download URLs.")
        return

    print(f"Found metadata file: {info_path}")
    
    try:
        with open(info_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading metadata JSON: {e}")
        return

    # Output directory
    output_base = "/home/kalashkala/visual-spatial-reasoning"
    if not os.path.exists(output_base):
        os.makedirs(output_base)
        print(f"Created output directory: {output_base}")
    
    files_processed = 0
    
    # Process each configuration (e.g., 'random', 'zeroshot')
    for config_name, config_data in data.items():
        # Only process entries that look like configurations (have download_checksums)
        if not isinstance(config_data, dict) or "download_checksums" not in config_data:
            continue
            
        print(f"\nConfiguration: {config_name}")
        config_dir = os.path.join(output_base, config_name)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        checksums = config_data.get("download_checksums", {})
        
        for url, meta in checksums.items():
            # Extract filename from URL (e.g., train.jsonl)
            filename = url.split('/')[-1]
            output_file = os.path.join(config_dir, filename)
            expected_checksum = meta.get("checksum")
            
            # Check if file already exists with correct checksum
            if os.path.exists(output_file):
                current_checksum = get_checksum(output_file)
                if expected_checksum and current_checksum == expected_checksum:
                    print(f"  [Skipping] {filename} (Already exists and verified)")
                    files_processed += 1
                    continue
                else:
                    print(f"  [Updating] {filename} (Checksum mismatch)")
            else:
                print(f"  [Downloading] {filename}...")
                
            try:
                urllib.request.urlretrieve(url, output_file)
                files_processed += 1
                
                # Verify checksum after download
                if expected_checksum:
                    new_checksum = get_checksum(output_file)
                    if new_checksum == expected_checksum:
                        print(f"    Verified checksum: OK")
                    else:
                        print(f"    WARNING: Checksum mismatch for {filename}!")
                        print(f"      Expected: {expected_checksum}")
                        print(f"      Got:      {new_checksum}")
            except Exception as e:
                print(f"    Failed to download {url}: {e}")

    print(f"\nOperation complete.")
    if files_processed > 0:
        print(f"Data has been saved to: {output_base}")
    else:
        print("No files were processed.")


if __name__ == "__main__":
    main()