import datasets

try:
    print("Loading val_all_images...")
    img_ds = datasets.load_dataset("lmms-lab/gqa", "val_all_images", cache_dir="/data/.cache/huggingface/datasets")
    print(img_ds)
    print(img_ds['val'].features)

    print("Loading val_all_instructions...")
    inst_ds = datasets.load_dataset("lmms-lab/gqa", "val_all_instructions", cache_dir="/data/.cache/huggingface/datasets")
    print(inst_ds)
    print(inst_ds['val'].features)
except Exception as e:
    print("Error:", e)
