import pandas as pd
import argparse
import os

def update_image_paths(input_file, output_file, old_prefix, new_prefix):
    """
    Updates the image paths in a Parquet dataset.
    Handles the format: [{'image': 'file:///path/to/image.jpg'}]
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    
    if 'images' not in df.columns:
        print(f"Error: 'images' column not found in {input_file}. Available columns: {list(df.columns)}")
        return
    
    import numpy as np
    def replace_path(image_list):
        if image_list is None:
            return None
        if not isinstance(image_list, (list, tuple, np.ndarray)):
            # If it's a single string (not expected in Qwen2-VL verl format, but safe check)
            if isinstance(image_list, str):
                return image_list.replace(old_prefix, new_prefix)
            return image_list
        
        new_list = []
        for img_dict in image_list:
            if isinstance(img_dict, dict) and 'image' in img_dict:
                old_path = img_dict['image']
                # Perform the replacement
                new_path = old_path.replace(old_prefix, new_prefix)
                new_list.append({'image': new_path})
            elif isinstance(img_dict, str):
                 new_list.append(img_dict.replace(old_prefix, new_prefix))
            else:
                new_list.append(img_dict)
        return new_list

    print(f"Found {len(df)} rows.")
    print(f"Replacing '{old_prefix}' with '{new_prefix}'...")
    
    df['images'] = df['images'].apply(replace_path)
    
    # Preview change on first row if possible
    try:
        sample = df['images'].iloc[0]
        print(f"Sample updated path: {sample}")
    except:
        pass

    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("Update complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update image paths in a Parquet dataset.")
    parser.add_argument("--input", required=True, help="Path to input .parquet file")
    parser.add_argument("--output", help="Path to output .parquet file")
    parser.add_argument("--old_prefix", default="home/debarpanb1/kalashkala/visual-question-answering/", 
                        help="Old path prefix (e.g. home/debarpanb1/kalashkala/visual-question-answering/)")
    parser.add_argument("--new_prefix", default="home/kalashkala/Datasets/VQAv2/", 
                        help="New path prefix (e.g. home/kalashkala/Datasets/VQAv2/)")
    
    args = parser.parse_args()
    
    if not args.output:
        # Default output name: name_updated.parquet
        name, ext = os.path.splitext(args.input)
        args.output = f"{name}_updated{ext}"

    update_image_paths(args.input, args.output, args.old_prefix, args.new_prefix)
