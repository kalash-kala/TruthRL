# #!/usr/bin/env python3
# import json
# import os
# import urllib.request
# import hashlib

# def get_checksum(file_path):
#     """Calculate SHA256 checksum of a file."""
#     sha256_hash = hashlib.sha256()
#     try:
#         with open(file_path, "rb") as f:
#             for byte_block in iter(lambda: f.read(4096), b""):
#                 sha256_hash.update(byte_block)
#         return sha256_hash.hexdigest()
#     except Exception:
#         return None

# def main():
#     # 1. INPUT: Path to the metadata file downloaded via 'hf download'
#     # This is currently on your root partition
#     info_path = "/root/kalashkala/vsr-dataset/dataset_infos.json"
    
#     # 2. OUTPUT: Path where you want the heavy data (Images/JSONL) to land
#     # We point this to the /data partition to save space on root
#     output_base = "/data/visual-spatial-reasoning-final"

#     if not os.path.exists(info_path):
#         print(f"Error: Could not find metadata file at: {info_path}")
#         print("Please ensure you ran: hf download juletxara/visual-spatial-reasoning --local-dir ./vsr-dataset")
#         return

#     print(f"Reading metadata from: {info_path}")
    
#     try:
#         with open(info_path, 'r') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"Error reading JSON: {e}")
#         return

#     if not os.path.exists(output_base):
#         os.makedirs(output_base)
#         print(f"Created output directory on data partition: {output_base}")
    
#     files_processed = 0
    
#     # Iterate through dataset configurations (e.g., 'random', 'zeroshot')
#     for config_name, config_data in data.items():
#         if not isinstance(config_data, dict) or "download_checksums" not in config_data:
#             continue
            
#         print(f"\nProcessing Configuration: {config_name}")
#         config_dir = os.path.join(output_base, config_name)
#         os.makedirs(config_dir, exist_ok=True)
            
#         checksums = config_data.get("download_checksums", {})
        
#         for url, meta in checksums.items():
#             filename = url.split('/')[-1]
#             output_file = os.path.join(config_dir, filename)
#             expected_checksum = meta.get("checksum")
            
#             # Skip if already exists and is valid
#             if os.path.exists(output_file):
#                 current_checksum = get_checksum(output_file)
#                 if expected_checksum and current_checksum == expected_checksum:
#                     print(f"  [Skipping] {filename} (Verified)")
#                     files_processed += 1
#                     continue
                
#             print(f"  [Downloading] {url} -> {output_file}")
#             try:
#                 urllib.request.urlretrieve(url, output_file)
#                 files_processed += 1
                
#                 # Immediate verification
#                 if expected_checksum:
#                     if get_checksum(output_file) == expected_checksum:
#                         print(f"    Checksum Verified: OK")
#                     else:
#                         print(f"    WARNING: Checksum mismatch for {filename}")
#             except Exception as e:
#                 print(f"    Failed to download: {e}")

#     print(f"\nDone! Entire dataset is now located at: {output_base}")

# if __name__ == "__main__":
#     main()

# import pandas as pd

# def main():

#     # Analysis of data:
#     df_test = pd.read_json('visual-spatial-reasoning-final/random/test.jsonl', lines=True)
#     df_dev = pd.read_json('visual-spatial-reasoning-final/random/dev.jsonl', lines=True)
#     df_train = pd.read_json('visual-spatial-reasoning-final/random/train.jsonl', lines=True)

#     print("Total number of examples:", len(df_test))
#     print("Number of examples per class:", df_test['label'].value_counts())

#     print("Total number of examples:", len(df_dev))
#     print("Number of examples per class:", df_dev['label'].value_counts())

#     print("Total number of examples:", len(df_train))
#     print("Number of examples per class:", df_train['label'].value_counts())

#     # sample 750 training samples where 500 have label 0 and 500 have label 1

#     df_train_sampled_label_0 = df_train[df_train['label'] == 0].sample(n=375)
#     df_train_sampled_label_1 = df_train[df_train['label'] == 1].sample(n=375)
#     df_train_sampled = pd.concat([df_train_sampled_label_0, df_train_sampled_label_1])
#     df_train_sampled.to_json('visual-spatial-reasoning-final/truthrl-sample/data/train_sampled.jsonl', orient='records', lines=True)

#     # sample 750 test samples where 500 have label 0 and 500 have label 1
#     df_test_sampled_label_0 = df_test[df_test['label'] == 0].sample(n=375)
#     df_test_sampled_label_1 = df_test[df_test['label'] == 1].sample(n=375)
#     df_test_sampled = pd.concat([df_test_sampled_label_0, df_test_sampled_label_1])
#     df_test_sampled.to_json('visual-spatial-reasoning-final/truthrl-sample/data/test_sampled.jsonl', orient='records', lines=True)

#     # verify the data
#     print("Train data")
#     print(len(df_train_sampled[df_train_sampled['label'] == 1]))
#     print(len(df_train_sampled[df_train_sampled['label'] == 0]))

#     print("Test data")
#     print(len(df_test_sampled[df_test_sampled['label'] == 1]))
#     print(len(df_test_sampled[df_test_sampled['label'] == 0]))




# if __name__ == "__main__":
#     main()

import json
import os
import requests
from tqdm import tqdm

def download_images(jsonl_files, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all unique image links to avoid redundant downloads
    image_tasks = {}
    for file_path in jsonl_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    image_name = data.get('image')
                    image_link = data.get('image_link')
                    if image_name and image_link:
                        image_tasks[image_name] = image_link
                except json.JSONDecodeError:
                    continue

    print(f"Found {len(image_tasks)} unique images to process.")
    
    # Download images
    for image_name, image_link in tqdm(image_tasks.items(), desc="Downloading images"):
        target_path = os.path.join(output_dir, image_name)
        
        # Skip if file already exists
        if os.path.exists(target_path):
            continue
            
        try:
            response = requests.get(image_link, timeout=15)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"\nFailed: {image_name} (Status: {response.status_code})")
        except Exception as e:
            print(f"\nError downloading {image_name}: {e}")

if __name__ == "__main__":
    DATA_FILES = [
        "/root/kalashkala/visual-spatial-reasoning-final/truthrl-sample/data/test_sampled.jsonl",
        "/root/kalashkala/visual-spatial-reasoning-final/truthrl-sample/data/train_sampled.jsonl"
    ]
    IMAGE_DIR = "/root/kalashkala/visual-spatial-reasoning-final/truthrl-sample/images"
    
    download_images(DATA_FILES, IMAGE_DIR)