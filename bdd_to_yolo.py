import json
import os
from pathlib import Path
from tqdm import tqdm  # pip install tqdm if you don't have it

BASE_DIR = Path("bdd10k/") 
OUTPUT_DIR = BASE_DIR / "labels"

IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Class Mapping (Must match your bdd100k.yaml)
CLASS_MAP = {
    "person": 0, "rider": 1, "car": 2, "truck": 3,
    "bus": 4, "train": 5, "motor": 6, "bike": 7,
    "traffic light": 8, "traffic sign": 9
}

def convert_bbox(box):
    """Converts x1, y1, x2, y2 to normalized x_center, y_center, width, height"""
    dw = 1.0 / IMG_WIDTH
    dh = 1.0 / IMG_HEIGHT
    
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + (w / 2)
    y_center = y1 + (h / 2)
    
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    return x_center, y_center, w, h

def process_dataset_split(split_name):
    """
    Processes a whole dataset split (e.g., 'train', 'val').
    Reads from the single JSON file 'bdd10k/bdd_labels/bdd100k_labels_images_<split_name>.json'
    and writes to 'bdd10k/labels/<split_name>'.
    """
    json_file_path = BASE_DIR / "bdd_labels" / f"bdd100k_labels_images_{split_name}.json"
    output_dir = OUTPUT_DIR / split_name
    
    if not json_file_path.exists():
        print(f"Warning: Input file not found, skipping: {json_file_path}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {json_file_path}...")
    with open(json_file_path) as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} images in '{split_name}' split...")

    for image_entry in tqdm(data, desc=f"  -> Converting {split_name}"):
        
        image_name = image_entry['name']
        txt_filename = os.path.splitext(image_name)[0] + ".txt"
        txt_path = output_dir / txt_filename
        
        with open(txt_path, 'w') as f_out:

            labels = image_entry.get('labels', [])
            
            for obj in labels:
                category = obj.get('category')
                
                if category in CLASS_MAP and 'box2d' in obj:
                    cls_id = CLASS_MAP[category]
                    xc, yc, w, h = convert_bbox(obj['box2d'])
                    f_out.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def main():
    """Main function to run the conversion for all dataset splits."""
    print("Starting BDD100k to YOLO conversion...")
    print(f"Source: {BASE_DIR / 'labels'}")
    print(f"Destination: {OUTPUT_DIR}\n")

    splits_to_process = ["train", "val"]
    
    for split in splits_to_process:
        process_dataset_split(split)

    print("\nConversion Complete.")
    print(f"YOLO format labels saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()