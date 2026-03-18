import datasets

try:
    ds = datasets.load_from_disk("/data/.cache/huggingface/datasets/lmms-lab___gqa/val_all_images/0.0.0/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8")
    print("Loaded from disk successfully")
    print(ds)
except Exception as e:
    print("Could not load from disk:", e)

    # try loading dataset from huggingface directly specifying cache_dir
    print("Trying load_dataset...")
    ds = datasets.load_dataset("lmms-lab/gqa", cache_dir="/data/.cache/huggingface/datasets")
    print("Loaded dataset successfully")
    print(ds)
