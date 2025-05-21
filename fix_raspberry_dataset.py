import os
import cv2
import numpy as np
import yaml
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_yaml(yaml_path: str) -> Dict:
    """Load YAML configuration."""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML {yaml_path}: {e}")
        raise

def validate_image(img_path: Path) -> Tuple[bool, str]:
    """Check if image is readable and valid."""
    try:
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            return False, f"Failed to load image {img_path}"
        return True, "Valid"
    except Exception as e:
        return False, f"Image load error {img_path}: {e}"

def clamp_yolo_box(x_center: float, y_center: float, width: float, height: float) -> Tuple[float, float, float, float]:
    """Clamp YOLO box coordinates to ensure valid boundaries."""
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.001), 2.0)  # Ensure positive
    height = min(max(height, 0.001), 2.0)
    # Adjust to fit within [0, 1]
    xmin = x_center - width / 2
    xmax = x_center + width / 2
    ymin = y_center - height / 2
    ymax = y_center + height / 2
    if xmin < 0:
        width = 2 * x_center
        xmin = 0
    if xmax > 1:
        width = 2 * (1 - x_center)
        xmax = 1
    if ymin < 0:
        height = 2 * y_center
        ymin = 0
    if ymax > 1:
        height = 2 * (1 - y_center)
        ymax = 1
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    return x_center, y_center, width, height

def fix_yolo_labels(label_dir: str, output_dir: str, num_classes: int = 5) -> int:
    """Fix YOLO labels by clamping boxes and removing invalid entries."""
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent exists
    output_dir.mkdir(exist_ok=True)
    fixed_count = 0
    for label_path in label_dir.glob("*.txt"):
        lines = []
        modified = False
        with open(label_path, 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    logger.warning(f"Skipping invalid line {i} in {label_path}: {line.strip()}")
                    modified = True
                    continue
                try:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    class_id = int(class_id)
                    if not (0 <= class_id < num_classes):
                        logger.warning(f"Skipping invalid class_id {class_id} in {label_path}, line {i}")
                        modified = True
                        continue
                    # Clamp box
                    x_center, y_center, width, height = clamp_yolo_box(x_center, y_center, width, height)
                    # Verify box
                    xmin = x_center - width / 2
                    xmax = x_center + width / 2
                    ymin = y_center - height / 2
                    ymax = y_center + height / 2
                    if xmin >= 0 and xmax <= 1 and ymin >= 0 and ymax <= 1 and xmin < xmax and ymin < ymax:
                        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    else:
                        logger.warning(f"Skipping invalid box in {label_path}, line {i}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                        modified = True
                except ValueError:
                    logger.warning(f"Skipping invalid format in {label_path}, line {i}: {line.strip()}")
                    modified = True
        if modified:
            fixed_count += 1
        if lines:  # Only save non-empty files
            output_path = output_dir / label_path.name
            with open(output_path, 'w') as f:
                f.write("\n".join(lines) + "\n")
        else:
            logger.warning(f"Label {label_path} is empty after fixing, skipping")
    return fixed_count

def validate_yolo_label(label_path: Path, num_classes: int = 5) -> Tuple[bool, List[Tuple[int, str]]]:
    """Validate YOLO label format and box coordinates."""
    errors = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            errors.append((0, "Empty label file"))
            return False, errors
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append((i, f"Invalid format, expected 5 values, got {len(parts)}"))
                continue
            try:
                class_id, x_center, y_center, width, height = map(float, parts)
                class_id = int(class_id)
                if not (0 <= class_id < num_classes):
                    errors.append((i, f"Invalid class_id {class_id}, expected 0-{num_classes-1}"))
                    continue
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    errors.append((i, f"Center ({x_center}, {y_center}) outside [0, 1]"))
                    continue
                if width <= 0 or height <= 0:
                    errors.append((i, f"Zero/negative dimensions ({width}, {height})"))
                    continue
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                    errors.append((i, f"Invalid box [{xmin}, {ymin}, {xmax}, {ymax}]"))
                    continue
            except ValueError:
                errors.append((i, f"Invalid number format: {line.strip()}"))
        return len(errors) == 0, errors
    except Exception as e:
        errors.append((0, f"Error reading label {label_path}: {e}"))
        return False, errors

def draw_boxes(img_path: Path, label_path: Path, class_names: List[str], output_dir: Path, max_viz: int = 20) -> Tuple[bool, int]:
    """Draw bounding boxes on image and save visualization."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Cannot visualize {img_path}: Failed to load image")
        return False, 0
    h, w = img.shape[:2]
    box_count = 0
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)
            xmin = int((x_center - width / 2) * w)
            ymin = int((y_center - height / 2) * h)
            xmax = int((x_center + width / 2) * w)
            ymax = int((y_center + height / 2) * h)
            color = (0, 255, 0) if class_id != 3 else (255, 0, 0)  # Red for Ripe Berries
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{class_names[class_id]}"
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            box_count += 1
        output_path = output_dir / f"{img_path.stem}_viz.jpg"
        cv2.imwrite(str(output_path), img)
        logger.info(f"Saved visualization: {output_path} with {box_count} boxes")
        return True, box_count
    except Exception as e:
        logger.error(f"Visualization error {img_path}: {e}")
        return False, 0

def fix_and_validate_dataset(data_yaml: str, output_dir: str = "dataset_fix"):
    """Fix and validate RaspberrySet dataset."""
    # Load YAML
    data_config = load_yaml(data_yaml)
    splits = ['train', 'val', 'test']
    class_names = data_config['names']
    num_classes = data_config['nc']
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Initialize report
    report = {
        'splits': {},
        'errors': [],
        'class_counts': {name: {'train': 0, 'val': 0, 'test': 0} for name in class_names},
        'box_counts': {'train': 0, 'val': 0, 'test': 0},
        'fixed_labels': {'train': 0, 'val': 0, 'test': 0}
    }
    
    # Step 1: Fix labels
    for split in splits:
        label_dir = Path(data_config[split]).parent / "labels"
        fixed_label_dir = output_dir / split / "labels"
        logger.info(f"Fixing labels in {split} split")
        fixed_count = fix_yolo_labels(label_dir, fixed_label_dir, num_classes)
        report['fixed_labels'][split] = fixed_count
        # Backup original labels
        backup_dir = label_dir.parent / "labels_backup"
        if not backup_dir.exists():
            shutil.move(label_dir, backup_dir)
            shutil.move(fixed_label_dir, label_dir)
        logger.info(f"Backed up original labels to {backup_dir}, replaced with fixed labels")
    
    # Step 2: Validate fixed dataset
    for split in splits:
        img_dir = Path(data_config[split])
        label_dir = img_dir.parent / "labels"
        report['splits'][split] = {
            'total_images': 0,
            'valid_pairs': 0,
            'invalid_images': [],
            'missing_labels': [],
            'invalid_labels': [],
            'total_boxes': 0
        }
        
        img_files = sorted(img_dir.glob("*.JPEG"))
        report['splits'][split]['total_images'] = len(img_files)
        viz_count = 0
        
        for img_path in img_files:
            label_path = label_dir / f"{img_path.stem}.txt"
            
            # Check image
            img_valid, img_msg = validate_image(img_path)
            if not img_valid:
                report['splits'][split]['invalid_images'].append(img_path.name)
                report['errors'].append(img_msg)
                continue
            
            # Check label
            if not label_path.exists():
                report['splits'][split]['missing_labels'].append(img_path.name)
                report['errors'].append(f"Missing label for {img_path}")
                continue
            
            # Validate label
            label_valid, label_errors = validate_yolo_label(label_path, num_classes)
            if not label_valid:
                report['splits'][split]['invalid_labels'].append((img_path.name, label_errors))
                report['errors'].extend([f"Invalid label {label_path}: Line {line}, {msg}" for line, msg in label_errors])
                continue
            
            # Count classes and boxes
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.strip().split()[0])
                    report['class_counts'][class_names[class_id]][split] += 1
                    report['splits'][split]['total_boxes'] += 1
            
            # Visualize
            if viz_count < 20:
                viz_success, box_count = draw_boxes(img_path, label_path, class_names, viz_dir)
                if viz_success:
                    viz_count += 1
            
            report['splits'][split]['valid_pairs'] += 1
        
        report['box_counts'][split] = report['splits'][split]['total_boxes']
        logger.info(f"{split.capitalize()} split: {report['splits'][split]['valid_pairs']}/{report['splits'][split]['total_images']} valid pairs, {report['splits'][split]['total_boxes']} boxes")
    
    # Save report
    report_path = output_dir / "fix_report.yaml"
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    logger.info(f"Saved fix report: {report_path}")
    
    # Print summary
    print("\nDataset Fix and Validation Summary:")
    for split in splits:
        print(f"{split.capitalize()}:")
        print(f"  Total images: {report['splits'][split]['total_images']}")
        print(f"  Valid pairs: {report['splits'][split]['valid_pairs']}")
        print(f"  Invalid images: {len(report['splits'][split]['invalid_images'])}")
        print(f"  Missing labels: {len(report['splits'][split]['missing_labels'])}")
        print(f"  Invalid labels: {len(report['splits'][split]['invalid_labels'])}")
        print(f"  Total boxes: {report['splits'][split]['total_boxes']}")
        print(f"  Fixed labels: {report['fixed_labels'][split]}")
    print("\nClass Distribution:")
    for cls in class_names:
        print(f"{cls}: Train={report['class_counts'][cls]['train']}, Val={report['class_counts'][cls]['val']}, Test={report['class_counts'][cls]['test']}")
    print(f"\nTotal Errors: {len(report['errors'])}")
    if report['errors']:
        print("Sample Errors (up to 10):")
        for err in report['errors'][:10]:
            print(f"  - {err}")

if __name__ == "__main__":
    data_yaml = "/content/RaspberrySet/split/data.yaml"
    fix_and_validate_dataset(data_yaml)