# debug_dataset.py
import yaml
from pathlib import Path
import numpy as np

def debug_dataset():
    data_path = Path("/home/root/dataset/DATASET2K/split/data.yaml")
    
    # Load data config
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("=== DATASET CONFIG ===")
    print(f"Number of classes: {data_config['nc']}")
    print(f"Class names: {data_config['names']}")
    print(f"Train images: {data_config['train']}")
    print(f"Val images: {data_config['val']}")
    
    # Check train labels
    train_label_dir = Path(data_config['train']).parent / 'labels'
    print(f"\n=== CHECKING TRAIN LABELS in {train_label_dir} ===")
    
    label_files = list(train_label_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files")
    
    total_objects = 0
    class_counts = {}
    problematic_files = []
    
    for label_file in label_files[:100]:  # Check first 100 files
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"WRONG FORMAT: {label_file} line {i}: {line}")
                    problematic_files.append(str(label_file))
                    continue
                    
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Check bounds
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    print(f"OUT OF BOUNDS: {label_file} - class:{class_id} x:{x_center:.6f} y:{y_center:.6f} w:{width:.6f} h:{height:.6f}")
                    problematic_files.append(str(label_file))
                
                # Check if bbox would go outside image
                x_min = x_center - width/2
                x_max = x_center + width/2  
                y_min = y_center - height/2
                y_max = y_center + height/2
                
                if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                    print(f"BBOX OUTSIDE: {label_file} - x_min:{x_min:.8f} x_max:{x_max:.8f} y_min:{y_min:.8f} y_max:{y_max:.8f}")
                    problematic_files.append(str(label_file))
                
                total_objects += 1
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                
        except Exception as e:
            print(f"ERROR reading {label_file}: {e}")
            problematic_files.append(str(label_file))
    
    print(f"\n=== SUMMARY ===")
    print(f"Total objects checked: {total_objects}")
    print(f"Class distribution: {dict(sorted(class_counts.items()))}")
    print(f"Problematic files: {len(set(problematic_files))}")
    
    # Check specific class 3 (green)
    print(f"\n=== GREEN TOMATOES (class 3) ===")
    green_count = class_counts.get(3, 0)
    print(f"Count: {green_count}")
    print(f"Percentage: {green_count/total_objects*100:.1f}%")

if __name__ == "__main__":
    debug_dataset()