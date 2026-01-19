# fix_dataset_fundamental.py
import yaml
from pathlib import Path
import numpy as np
import shutil

def validate_and_fix_dataset():
    data_path = Path("/home/root/dataset/DATASET2K/split/data.yaml")
    
    # Load data config
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("=== COMPREHENSIVE DATASET VALIDATION ===")
    
    # Check train and val splits
    for split in ['train', 'val']:
        print(f"\n=== VALIDATING {split.upper()} SPLIT ===")
        
        image_dir = Path(data_config[split]) 
        label_dir = image_dir.parent / 'labels'
        
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
        label_files = list(label_dir.glob("*.txt"))
        
        print(f"Found {len(image_files)} images and {len(label_files)} label files")
        
        # Check for image-label correspondence
        image_names = {f.stem for f in image_files}
        label_names = {f.stem for f in label_files}
        
        missing_labels = image_names - label_names
        missing_images = label_names - image_names
        
        if missing_labels:
            print(f"WARNING: {len(missing_labels)} images missing labels: {list(missing_labels)[:5]}...")
        
        if missing_images:
            print(f"WARNING: {len(missing_images)} labels missing images: {list(missing_images)[:5]}...")
        
        # Validate each label file
        problematic_files = []
        total_objects = 0
        fixed_count = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                valid_lines = []
                has_issues = False
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    
                    # Check line format
                    if len(parts) != 5:
                        print(f"INVALID FORMAT: {label_file.name} line {i}: '{line.strip()}'")
                        has_issues = True
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                    except ValueError:
                        print(f"INVALID VALUES: {label_file.name} line {i}: '{line.strip()}'")
                        has_issues = True
                        continue
                    
                    # Validate ranges
                    if not (0 <= class_id < data_config['nc']):
                        print(f"INVALID CLASS: {label_file.name} line {i}: class {class_id}")
                        has_issues = True
                        continue
                    
                    # Fix coordinate ranges
                    x_center = np.clip(x_center, 0.0, 1.0)
                    y_center = np.clip(y_center, 0.0, 1.0)
                    width = np.clip(width, 0.0, 1.0)
                    height = np.clip(height, 0.0, 1.0)
                    
                    # Ensure bbox stays within image with safety margin
                    x_min = max(0.0, x_center - width/2)
                    x_max = min(1.0, x_center + width/2)
                    y_min = max(0.0, y_center - height/2)
                    y_max = min(1.0, y_center + height/2)
                    
                    # Recalculate valid bbox
                    width_new = max(x_max - x_min, 0.01)  # Minimum 1% width
                    height_new = max(y_max - y_min, 0.01)  # Minimum 1% height
                    x_center_new = (x_min + x_max) / 2
                    y_center_new = (y_min + y_max) / 2
                    
                    # Final validation
                    if (x_center_new < 0 or x_center_new > 1 or y_center_new < 0 or y_center_new > 1 or
                        width_new <= 0 or width_new > 1 or height_new <= 0 or height_new > 1):
                        print(f"INVALID BBOX: {label_file.name} line {i} - skipping")
                        continue
                    
                    valid_line = f"{class_id} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}\n"
                    valid_lines.append(valid_line)
                    total_objects += 1
                
                # Write fixed file if there were issues
                if has_issues or len(valid_lines) != len(lines):
                    # Create backup
                    backup_path = label_file.with_suffix('.txt.backup')
                    if not backup_path.exists():
                        shutil.copy2(label_file, backup_path)
                    
                    # Write fixed file
                    with open(label_file, 'w') as f:
                        f.writelines(valid_lines)
                    
                    fixed_count += 1
                    print(f"FIXED: {label_file.name} - {len(valid_lines)}/{len(lines)} bboxes kept")
                    
                if has_issues:
                    problematic_files.append(label_file.name)
                    
            except Exception as e:
                print(f"ERROR processing {label_file}: {e}")
                problematic_files.append(label_file.name)
        
        print(f"\n{split.upper()} SUMMARY:")
        print(f"Total objects: {total_objects}")
        print(f"Problematic files: {len(problematic_files)}")
        print(f"Fixed files: {fixed_count}")
        
        if problematic_files:
            print(f"Problematic files (first 10): {problematic_files[:10]}")

def check_cache_files():
    """Check and clear problematic cache files"""
    print("\n=== CHECKING CACHE FILES ===")
    
    cache_dirs = [
        Path("/home/root/dataset/DATASET2K/split/train/labels.cache"),
        Path("/home/root/dataset/DATASET2K/split/val/labels.cache"),
    ]
    
    for cache_path in cache_dirs:
        if cache_path.exists():
            print(f"Removing cache: {cache_path}")
            cache_path.unlink()

if __name__ == "__main__":
    validate_and_fix_dataset()
    check_cache_files()
    print("\n=== DATASET FIX COMPLETE ===")