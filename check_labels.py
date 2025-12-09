import os
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path("bdd10k")

# UPDATED: Images are directly in bdd10k/train, bdd10k/val, etc.
# If they are in bdd10k/images/train, change this to BASE_DIR / "images"
IMAGES_DIR = BASE_DIR 

LABELS_DIR = BASE_DIR / "labels"      # Where your .txt files are

def check_split(split_name):
    print(f"\n--- Checking '{split_name}' split ---")
    
    # Look for images in bdd10k/train (or val/test)
    img_path = IMAGES_DIR / split_name
    # Look for labels in bdd10k/labels/train
    lbl_path = LABELS_DIR / split_name
    
    if not img_path.exists():
        print(f"Skipping {split_name}: Image directory not found at {img_path}")
        return
    if not lbl_path.exists():
        print(f"Skipping {split_name}: Label directory not found at {lbl_path}")
        return

    # Get sets of filenames (without extensions)
    img_files = {f.stem for f in img_path.glob("*.jpg")}
    lbl_files = {f.stem for f in lbl_path.glob("*.txt")}
    
    print(f"Found {len(img_files)} images in {img_path}")
    print(f"Found {len(lbl_files)} label files in {lbl_path}")
    
    # Calculate differences
    excessive_labels = lbl_files - img_files
    missing_labels = img_files - lbl_files
    
    if excessive_labels:
        print(f"EXCESSIVE LABELS: {len(excessive_labels)}")
        print(f"   (You have labels for these, but no images. Safe to delete.)")
    else:
        print("No excessive labels found.")
        
    if missing_labels:
        print(f"MISSING LABELS: {len(missing_labels)}")

def main():
    splits = ['train', 'val']
    for split in splits:
        check_split(split)

if __name__ == "__main__":
    main()