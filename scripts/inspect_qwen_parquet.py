
import pandas as pd
import pyarrow.parquet as pq
import io
from PIL import Image

pd.set_option('display.max_colwidth', None)

file_path = "/root/kalashkala/visual-spatial-reasoning-final/truthrl-sample/parquet/train.parquet"

try:
    df = pd.read_parquet(file_path)
    print("Columns:", df.columns.tolist())
    print("\nShape:", df.shape)
    print("\nFirst row sample:")
    row = df.iloc[0]
    for col in df.columns:
        val = row[col]
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            print(f"{col}: (List of dicts, showing first element keys)")
            print(val[0].keys())
            # Check content of messages if present
            if 'content' in val[0]:
                print(f"  Content sample: {val[0]['content']}")
        elif isinstance(val, (bytes, bytearray)):
             print(f"{col}: (Bytes, length {len(val)})")
        else:
            print(f"{col}: {val}")

    if 'images' in df.columns:
        img_data = df.iloc[0]['images']
        if isinstance(img_data, list) and len(img_data) > 0:
            print(f"\nImages column type: List of {type(img_data[0])}")
            if isinstance(img_data[0], dict) and 'bytes' in img_data[0]:
                 print("Image data structure seems to be dict with bytes.")
            elif isinstance(img_data[0], bytes):
                 print("Image data seems to be raw bytes.")

except Exception as e:
    print(f"Error reading parquet: {e}")
