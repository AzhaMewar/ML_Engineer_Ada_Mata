"""
bsort: Data Preprocessing and Re-labeling Script
"""

import cv2
import yaml
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
import shutil
from rich.console import Console

# --- Globals ---
console = Console()

# --- Helper Functions ---

def _load_config(config_path: Path) -> Dict[str, Any]:
    """Loads the YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _de_normalize_bbox(
    img_shape: Tuple[int, int], bbox: List[float]
) -> Tuple[int, int, int, int]:
    """Converts YOLO bbox to [xmin, ymin, xmax, ymax]"""
    h, w = img_shape
    x_center, y_center, w_norm, h_norm = bbox
    
    x_center_px = x_center * w
    y_center_px = y_center * h
    w_px = w_norm * w
    h_px = h_norm * h
    
    xmin = int(x_center_px - (w_px / 2))
    ymin = int(y_center_px - (h_px / 2))
    xmax = int(x_center_px + (w_px / 2))
    ymax = int(y_center_px + (h_px / 2))
    
    return (xmin, ymin, xmax, ymax)

def _classify_color(
    hsv_values: Tuple[float, float, float],
    thresholds: Dict[str, Any]
) -> int:
    """
    Classifies a mean HSV value into 0, 1, or 2.
    
    Returns:
        int: 0 (light_blue), 1 (dark_blue), or 2 (other)
    """
    H, S, V = hsv_values
    
    # Get thresholds from config
    light_blue_h = thresholds['light_blue_h']
    dark_blue_h = thresholds['dark_blue_h']
    min_s = thresholds['min_saturation']
    min_v = thresholds['min_value']
    
    # Check for grays/blacks/whites first
    if S < min_s or V < min_v:
        return 2  # 'other'

    # Check for light blue
    if light_blue_h[0] <= H <= light_blue_h[1]:
        return 0  # 'light_blue'
        
    # Check for dark blue
    if dark_blue_h[0] <= H <= dark_blue_h[1]:
        return 1  # 'dark_blue'

    # Everything else (green, yellow, red, etc.)
    return 2  # 'other'

def _process_file(
    img_path: Path,
    label_path: Path,
    hsv_image: np.ndarray,
    thresholds: Dict[str, Any]
) -> Tuple[List[str], int]:
    """
    Processes a single image and its label file.
    Returns a list of new YOLO label strings and a count of objects.
    """
    new_labels = []
    if not label_path.exists():
        return new_labels, 0

    img_shape = hsv_image.shape[:2] # (h, w)
    
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            # Original class is ignored
            _, x_center, y_center, w_norm, h_norm = map(float, line.split())
        except ValueError:
            continue

        bbox_coords = [x_center, y_center, w_norm, h_norm]
        xmin, ymin, xmax, ymax = _de_normalize_bbox(img_shape, bbox_coords)
        
        # --- START OF CODE FIX ---
        
        # 1. Crop the cap region (like before)
        cap_hsv_crop = hsv_image[ymin:ymax, xmin:xmax]
        
        if cap_hsv_crop.size == 0:
            continue
            
        # 2. Create a circular mask to ignore corners
        #    Ini adalah langkah KUNCI untuk mengabaikan latar hijau
        crop_h, crop_w = cap_hsv_crop.shape[:2]
        center = (crop_w // 2, crop_h // 2)
        # Buat radius lingkaran sedikit lebih kecil dari kotaknya
        radius = int(min(crop_h, crop_w) // 2 * 0.9) # 90% dari radius
        
        # Buat gambar hitam (mask) seukuran crop
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        # Gambar lingkaran putih di tengah mask
        cv2.circle(mask, center, radius, (255), -1) # -1 artinya isi lingkaran

        # 3. Hitung warna rata-rata HANYA dari area lingkaran putih
        #    Fungsi 'mask=' akan mengabaikan semua piksel di luar lingkaran
        mean_hsv = cv2.mean(cap_hsv_crop, mask=mask)[:3]
        
        # --- END OF CODE FIX ---
        
        # 4. Classify the color (same as before)
        new_class_id = _classify_color(mean_hsv, thresholds)
        
        # 5. Create new label string (same as before)
        new_labels.append(
            f"{new_class_id} {x_center} {y_center} {w_norm} {h_norm}"
        )
    
    return new_labels, len(new_labels)

# --- Main Runner ---

def run(config_path: Path):
    """
    Main function to run the preprocessing pipeline.
    """
    console.rule("[bold blue]Data Preprocessing[/bold blue]")
    
    # 1. Load Config
    config = _load_config(config_path)
    raw_path = Path(config['dataset_path'])
    processed_path = Path(config['processed_path'])
    thresholds = config['hsv_thresholds']
    class_names = list(config['class_names'].values())
    split_ratio = config['train_val_split']

    if not raw_path.exists():
        console.print(f"[bold red]Error:[/bold red] Raw dataset path not found: {raw_path}")
        return

    # 2. Clean and create output directories
    console.print(f"Cleaning output directory: {processed_path}")
    if processed_path.exists():
        shutil.rmtree(processed_path)
    
    (processed_path / "images/train").mkdir(parents=True)
    (processed_path / "images/val").mkdir(parents=True)
    (processed_path / "labels/train").mkdir(parents=True)
    (processed_path / "labels/val").mkdir(parents=True)

    # 3. Find all image files
    console.print(f"Finding images in: {raw_path}")
    image_files = list(raw_path.glob("*.jpg"))
    random.shuffle(image_files)
    split_point = int(len(image_files) * split_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]

    console.print(f"Found {len(image_files)} images.")
    console.print(f"Split: {len(train_files)} train, {len(val_files)} val.")
    
    total_objects = 0

    # 4. Process files
    for split, files in [("train", train_files), ("val", val_files)]:
        console.print(f"\nProcessing [bold]{split}[/bold] set...")
        
        for img_path in tqdm(files, desc=f"Processing {split} files"):
            label_path = img_path.with_suffix(".txt")
            
            # Load image and convert to HSV
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Get new labels
            new_labels, obj_count = _process_file(
                img_path, label_path, hsv_image, thresholds
            )
            total_objects += obj_count

            if new_labels:
                # Define new paths
                new_img_path = processed_path / "images" / split / img_path.name
                new_label_path = processed_path / "labels" / split / label_path.name
                
                # Copy image
                shutil.copy(img_path, new_img_path)
                
                # Write new labels
                with open(new_label_path, "w") as f:
                    f.write("\n".join(new_labels))

    # 5. Create data.yaml for YOLO
    data_yaml_path = processed_path / "data.yaml"
    data_yaml_content = {
        "train": str((processed_path / "images/train").resolve()),
        "val": str((processed_path / "images/val").resolve()),
        "nc": len(class_names),
        "names": class_names,
    }
    
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
        
    console.print(f"\nCreated YOLO config: {data_yaml_path}")
    console.print(f"[bold green]Preprocessing complete![/bold green]")
    console.print(f"Total objects re-labeled: {total_objects}")