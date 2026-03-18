import os
import datasets
from tqdm import tqdm

def main():
    cache_dir = "/data/.cache/huggingface/datasets"
    output_dir = "/home/kalashkala/Datasets/GQA"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets from cache...")
    img_ds = datasets.load_dataset("lmms-lab/gqa", "val_all_images", cache_dir=cache_dir)
    inst_ds = datasets.load_dataset("lmms-lab/gqa", "val_all_instructions", cache_dir=cache_dir)

    # 1. Option: Save as Hugging Face Dataset format (fast, ready to be loaded via load_from_disk)
    hf_img_dir = os.path.join(output_dir, "hf_val_all_images")
    hf_inst_dir = os.path.join(output_dir, "hf_val_all_instructions")
    
    print(f"Saving HF Dataset to {hf_img_dir} and {hf_inst_dir}")
    img_ds.save_to_disk(hf_img_dir)
    inst_ds.save_to_disk(hf_inst_dir)

    # 2. Option: Extract raw files (Images as JPEGs + Metadata as Parquet)
    images_dir = os.path.join(output_dir, "val_images")
    print(f"Extracting images to {images_dir}...")
    os.makedirs(images_dir, exist_ok=True)

    for item in tqdm(img_ds["val"], desc="Saving images"):
        img_id = item["id"]
        image = item["image"]
        # Save image as JPEG
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            # Sometimes image can be None or corrupted, so use try-except
            try:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path, "JPEG")
            except Exception as e:
                print(f"Failed to save image {img_id}: {e}")

    parquet_path = os.path.join(output_dir, "val_instructions.parquet")
    print(f"Saving metadata to {parquet_path}...")
    inst_ds["val"].to_parquet(parquet_path)
    
    print("Dataset extraction is complete!")

if __name__ == "__main__":
    main()
