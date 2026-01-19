# fix_labels_final.py
from pathlib import Path
import numpy as np

def fix_all_label_files():
    label_dir = Path("/home/root/dataset/DATASET2K/split/train/labels/")
    label_files = list(label_dir.glob("*.txt"))
    
    print(f"Checking and fixing all {len(label_files)} label files...")
    
    total_fixed = 0
    for filepath in label_files:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            needs_fixing = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    fixed_lines.append(line)
                    continue
                    
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # AGGRESSIVE clipping to handle floating point precision issues
                x_center = np.clip(x_center, 0.0, 1.0)
                y_center = np.clip(y_center, 0.0, 1.0)
                width = np.clip(width, 0.0, 1.0)
                height = np.clip(height, 0.0, 1.0)
                
                # Calculate boundaries with safety margin
                x_min = max(0.0, x_center - width/2)
                x_max = min(1.0, x_center + width/2)
                y_min = max(0.0, y_center - height/2)
                y_max = min(1.0, y_center + height/2)
                
                # Recalculate with precise clipping
                width_new = max(x_max - x_min, 0.001)  # Minimum width
                height_new = max(y_max - y_min, 0.001)  # Minimum height
                x_center_new = (x_min + x_max) / 2
                y_center_new = (y_min + y_max) / 2
                
                # Final validation
                if (x_center_new < 0 or x_center_new > 1 or y_center_new < 0 or y_center_new > 1 or
                    width_new <= 0 or width_new > 1 or height_new <= 0 or height_new > 1):
                    print(f"  Invalid bbox in {filepath.name}: x:{x_center_new:.6f} y:{y_center_new:.6f} w:{width_new:.6f} h:{height_new:.6f}")
                    continue
                
                fixed_line = f"{class_id} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}\n"
                fixed_lines.append(fixed_line)
                
                # Check if this line needed fixing
                original_values = list(map(float, parts[1:5]))
                fixed_values = [x_center_new, y_center_new, width_new, height_new]
                if original_values != fixed_values:
                    needs_fixing = True
            
            if needs_fixing:
                with open(filepath, 'w') as f:
                    f.writelines(fixed_lines)
                total_fixed += 1
                print(f"  Fixed {filepath.name} - {len(fixed_lines)} bboxes")
                
        except Exception as e:
            print(f"ERROR processing {filepath}: {e}")
    
    print(f"\nFixed {total_fixed} files out of {len(label_files)}")

if __name__ == "__main__":
    fix_all_label_files()