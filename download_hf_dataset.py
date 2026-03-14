import argparse
from datasets import load_dataset
from datasets.config import HF_DATASETS_CACHE
import os

def download_and_save_dataset(dataset_name, split, output_file, format_type, config_name=None, data_files=None):
    """
    Downloads a specific split of a Hugging Face dataset and saves it to a file.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face (e.g., 'imdb', 'rotten_tomatoes').
        split (str): The split to download (e.g., 'train', 'test', 'validation').
        output_file (str): The path to the output file or directory.
        format_type (str): The format to save the dataset ('csv', 'json', 'parquet', or 'hf').
        config_name (str, optional): The configuration name of the dataset if it has subsets.
        data_files (str, optional): Specific data files to load, to avoid downloading the whole dataset.
    """
    print(f"Loading dataset '{dataset_name}' (config: {config_name}), split: '{split}'...")
    
    try:
        # Prepare kwargs for load_dataset
        kwargs = {'split': split}
        if data_files:
            kwargs['data_files'] = {split: data_files}
            
        # Load the specific split of the dataset
        if config_name:
            dataset = load_dataset(dataset_name, config_name, **kwargs)
        else:
            dataset = load_dataset(dataset_name, **kwargs)
            
        print(f"Dataset successfully loaded. Number of rows: {len(dataset)}")
        
        if output_file is None:
            config_str = f"_{config_name}" if config_name else ""
            file_extension = "" if format_type == "hf" else f".{format_type}"
            filename = f"{dataset_name.replace('/', '_')}{config_str}_{split}{file_extension}"
            output_file = os.path.join(str(HF_DATASETS_CACHE), filename)
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"Saving dataset to {output_file} in '{format_type}' format...")
        
        # Save in the specified format
        if format_type == 'json':
            dataset.to_json(output_file)
        elif format_type == 'csv':
            dataset.to_csv(output_file)
        elif format_type == 'parquet':
            dataset.to_parquet(output_file)
        elif format_type == 'hf':
            # 'hf' saves as Hugging Face Dataset format to a directory
            dataset.save_to_disk(output_file)
        else:
            raise ValueError(f"Unsupported format: {format_type}. Choose from 'csv', 'json', 'parquet', 'hf'.")
            
        print("Save complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset split and save it to a file.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the Hugging Face dataset (e.g., 'imdb')")
    parser.add_argument("--config", type=str, default=None, help="Configuration/subset name (if applicable)")
    parser.add_argument("--split", type=str, required=True, help="Split to download (e.g., 'train', 'test', 'validation')")
    parser.add_argument("--output", type=str, default=None, help="Path to the output file or folder. If omitted, saves to HuggingFace cache directory.")
    parser.add_argument("--format", type=str, choices=['csv', 'json', 'parquet', 'hf'], default='json', 
                        help="Format to save as (default: json). 'hf' saves as HuggingFace dataset directory.")
    parser.add_argument("--data_files", type=str, default=None, 
                        help="Specific data files parameter to avoid downloading the whole dataset. Example: 'data/testdev-*.parquet'")
    
    args = parser.parse_args()
    
    download_and_save_dataset(
        dataset_name=args.dataset,
        split=args.split,
        output_file=args.output,
        format_type=args.format,
        config_name=args.config,
        data_files=args.data_files
    )
