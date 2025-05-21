import os
import shutil
from pathlib import Path
import random
import logging
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def read_classes(classes_path):
    """Read and clean class names from classes.txt"""
    class_names = []
    with open(classes_path, 'r') as f:
        for line in f:
            # Remove numbers and extra spaces
            cleaned = ' '.join(line.strip().split()[1:]).strip()
            if cleaned:
                class_names.append(cleaned)
            else:
                logger.warning(f"Skipping invalid line in classes.txt: {line.strip()}")
    return class_names

def create_data_yaml(output_dir, class_names):
    """Create data.yaml file for SSD training"""
    data = {
        'path': str(output_dir),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': class_names,
        'nc': len(class_names)
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    logger.info(f"Created data.yaml at {yaml_path}")
    return yaml_path

def prettify_xml(elem):
    """Return a pretty-printed XML string"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def yolo_to_voc(yolo_label_path, image_path, output_xml_path, class_names):
    """Convert YOLO format to VOC XML with enhanced validation"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to load image {image_path}")
            return False
            
        height, width = img.shape[:2]
        annotation = ET.Element("annotation")
        
        # Add filename and size
        ET.SubElement(annotation, "filename").text = image_path.name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        
        valid_boxes = 0
        if yolo_label_path.exists():
            with open(yolo_label_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.debug(f"Skipping invalid line {line_num} in {yolo_label_path.name}: {line.strip()}")
                        continue
                        
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError:
                        logger.warning(f"Invalid values in line {line_num} of {yolo_label_path.name}")
                        continue
                    
                    # Validate class ID
                    if class_id >= len(class_names):
                        logger.warning(f"Invalid class ID {class_id} in {yolo_label_path.name}")
                        continue
                    
                    # Convert YOLO to VOC coordinates
                    x_min = max(0, int((x_center - w/2) * width))
                    y_min = max(0, int((y_center - h/2) * height))
                    x_max = min(width, int((x_center + w/2) * width))
                    y_max = min(height, int((y_center + h/2) * height))
                    
                    # Validate box dimensions
                    if x_max <= x_min or y_max <= y_min:
                        logger.warning(f"Invalid box in {yolo_label_path.name} line {line_num}")
                        continue
                    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                        logger.warning(f"Small box in {yolo_label_path.name} line {line_num}")
                        continue
                    
                    # Create XML structure
                    obj = ET.SubElement(annotation, "object")
                    ET.SubElement(obj, "name").text = class_names[class_id]
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"
                    
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(x_min)
                    ET.SubElement(bndbox, "ymin").text = str(y_min)
                    ET.SubElement(bndbox, "xmax").text = str(x_max)
                    ET.SubElement(bndbox, "ymax").text = str(y_max)
                    valid_boxes += 1
        
        if valid_boxes == 0:
            logger.warning(f"No valid boxes in {yolo_label_path.name}")
            return False
            
        # Write formatted XML
        pretty_xml = prettify_xml(annotation)
        with open(output_xml_path, 'w') as f:
            f.write(pretty_xml)
        return True
        
    except Exception as e:
        logger.error(f"Error converting {yolo_label_path.name}: {str(e)}")
        return False

def split_dataset(
    images_dir,
    labels_dir,
    output_dir,
    classes_path,
    ratios=(0.7, 0.2, 0.1),
    img_exts=('.jpeg', '.jpg', '.png')
):
    """Enhanced dataset splitting with better validation"""
    # Validate input paths
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    logger.info(f"Image directory: {images_dir.resolve()}")
    logger.info(f"Labels directory: {labels_dir.resolve()}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Read and validate class names
    class_names = read_classes(classes_path)
    if not class_names:
        raise ValueError("No valid classes found in classes.txt")
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect valid image-label pairs with multiple extensions
    valid_pairs = []
    for ext in img_exts:
        logger.info(f"Checking {ext.upper()} files...")
        for img_path in images_dir.glob(f'*{ext}'):
            label_path = labels_dir / f'{img_path.stem}.txt'
            
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
                logger.debug(f"Found pair: {img_path.name} <-> {label_path.name}")
            else:
                logger.warning(f"Missing label for {img_path.name}")
                logger.debug(f"Expected label path: {label_path}")

    if not valid_pairs:
        raise ValueError(f"No valid image-label pairs found. Check:\n"
                         f"1. Matching filenames (image.jpg <-> image.txt)\n"
                         f"2. File extensions in {img_exts}\n"
                         f"3. Files in correct directories")
    
    # Shuffle and split
    random.shuffle(valid_pairs)
    n_total = len(valid_pairs)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    n_test = n_total - n_train - n_val
    
    splits = {
        'train': valid_pairs[:n_train],
        'val': valid_pairs[n_train:n_train+n_val],
        'test': valid_pairs[n_train+n_val:]
    }
    
    # Process splits
    for split_name, pairs in splits.items():
        logger.info(f"Processing {split_name} set ({len(pairs)} images)")
        success_count = 0
        
        for img_path, label_path in pairs:
            dest_img = output_dir / split_name / 'images' / img_path.name
            dest_ann = output_dir / split_name / 'annotations' / f'{img_path.stem}.xml'
            
            try:
                # Copy image
                shutil.copy(img_path, dest_img)
                # Convert annotation
                if yolo_to_voc(label_path, img_path, dest_ann, class_names):
                    success_count += 1
                else:
                    dest_img.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed processing {img_path.name}: {str(e)}")
                dest_img.unlink(missing_ok=True)
                dest_ann.unlink(missing_ok=True)
        
        logger.info(f"Successfully processed {success_count}/{len(pairs)} {split_name} images")
    
    # Create data.yaml
    create_data_yaml(output_dir, class_names)
    
    logger.info(f"""
    Dataset Split Summary:
    - Total valid pairs: {n_total}
    - Training set:   {n_train} ({n_train/n_total:.1%})
    - Validation set: {n_val} ({n_val/n_total:.1%})
    - Test set:       {n_test} ({n_test/n_total:.1%})
    """)

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'images_dir': "/content/RaspberrySet/images",
        'labels_dir': "/content/RaspberrySet/annotations",
        'output_dir': "/content/RaspberrySet/split",
        'classes_path': "/content/RaspberrySet/classes.txt",
        'ratios': (0.7, 0.2, 0.1),
        'img_exts': ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG')
    }
    
    try:
        logger.info("Starting dataset preparation...")
        split_dataset(**CONFIG)
        logger.info("Dataset preparation completed successfully")
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        exit(1)
