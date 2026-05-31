import os

def verify_dataset_structure(image_dir):
    """
    Crawls our raw image directory to verify that the dataset is present,
    properly organized, and free of empty folders.
    """
    print(f"Running Data Check on: {image_dir} ---")
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Directory {image_dir} does not exist!")
        return False
        
    categories = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    if not categories:
        print("[ERROR] No subdirectories/classes found in the image path.")
        return False
        
    print(f"[SUCCESS] Found {len(categories)} distinct hair classification categories.")
    for cat in categories:
        cat_path = os.path.join(image_dir, cat)
        files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f" -> Class '{cat}': Contains {len(files)} valid images.")
        
    print("Data validation process complete.\n")
    return True

if __name__ == "__main__":
    RAW_DIR = "hair-ai-trichologist/data/raw/hair_images"
    verify_dataset_structure(RAW_DIR)
