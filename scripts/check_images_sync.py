import os

def check_images(dir1, dir2):
    """
    Checks if all images in dir1 exist in dir2.
    """
    print(f"Checking images from: {dir1}")
    print(f"Against directory:   {dir2}")
    
    if not os.path.exists(dir1):
        print(f"Error: {dir1} does not exist.")
        return
    if not os.path.exists(dir2):
        print(f"Error: {dir2} does not exist.")
        return
        
    images1 = set(os.listdir(dir1))
    images2 = set(os.listdir(dir2))
    
    print(f"Count in directory 1: {len(images1)}")
    print(f"Count in directory 2: {len(images2)}")
    
    missing_in_dir2 = images1 - images2
    
    if not missing_in_dir2:
        print("\nSUCCESS: All images from dir1 are present in dir2.")
    else:
        print(f"\nFAILURE: {len(missing_in_dir2)} images from dir1 are missing in dir2.")
        print("Example missing files (up to 5):")
        for img in list(missing_in_dir2)[:5]:
            print(f" - {img}")

if __name__ == "__main__":
    dir1 = "/home/kalashkala/Datasets/VQAv2/processed_for_verl_2A100/images"
    dir2 = "/home/kalashkala/Datasets/VQAv2/processed_for_verl/images"
    check_images(dir1, dir2)
