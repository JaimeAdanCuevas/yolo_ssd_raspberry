# visualize_dataset_simple.py
import cv2
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

# Define class colors
CLASS_COLORS = {
    0: (255, 0, 0),    # Boton - Red
    1: (0, 0, 255),    # BrightRed C4 - Blue  
    2: (128, 0, 128),  # DarkRed C5 - Purple
    3: (0, 255, 0),    # Green - Green
    4: (0, 255, 255)   # Orange(red dot) - Yellow
}

CLASS_NAMES = ['Boton', 'BrightRed C4', 'DarkRed C5', 'Green', 'Orange(red dot)']

def draw_bboxes_pil(image, bboxes, classes):
    """Draw bounding boxes on PIL image"""
    img_with_boxes = image.copy()
    w, h = img_with_boxes.size
    
    for bbox, cls in zip(bboxes, classes):
        # YOLO format: x_center, y_center, width, height (normalized)
        x_center, y_center, bbox_w, bbox_h = bbox
        
        # Convert to pixel coordinates
        x1 = int((x_center - bbox_w/2) * w)
        y1 = int((y_center - bbox_h/2) * h)
        x2 = int((x_center + bbox_w/2) * w)
        y2 = int((y_center + bbox_h/2) * h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Get color and class name
        color = CLASS_COLORS.get(int(cls), (255, 255, 255))
        class_name = CLASS_NAMES[int(cls)]
        
        # Draw using PIL
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_with_boxes)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{class_name}"
        draw.text((x1, y1 - 20), label, fill=color)
    
    return img_with_boxes

def visualize_random_samples():
    split_dir = Path("/home/root/dataset/DATASET2K_split")
    
    print("=== VISUALIZING RANDOM SAMPLES ===")
    
    for split in ['train', 'val']:
        print(f"\n--- {split.upper()} Split ---")
        
        image_dir = split_dir / split / "images"
        label_dir = split_dir / split / "labels"
        
        # Get all image files
        image_files = list(image_dir.glob("*.jpg"))
        print(f"Total images: {len(image_files)}")
        
        # Select random samples
        random.shuffle(image_files)
        samples = image_files[:6]  # Show 6 samples
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, (ax, img_path) in enumerate(zip(axes, samples)):
            try:
                # Load image with PIL
                image = Image.open(img_path)
                
                # Load corresponding labels
                label_path = label_dir / img_path.with_suffix('.txt').name
                bboxes = []
                classes = []
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                coords = list(map(float, parts[1:5]))
                                classes.append(cls_id)
                                bboxes.append(coords)
                
                # Draw bounding boxes
                image_with_boxes = draw_bboxes_pil(image, bboxes, classes)
                
                # Display
                ax.imshow(image_with_boxes)
                ax.set_title(f"{img_path.name}\nObjects: {len(bboxes)}", fontsize=10)
                ax.axis('off')
                
                print(f"  {img_path.name}: {len(bboxes)} objects")
                for cls_id, bbox in zip(classes, bboxes):
                    print(f"    - {CLASS_NAMES[cls_id]}: {bbox}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")
                ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"{split.upper()} Split - Random Samples", fontsize=16, y=1.02)
        plt.savefig(f"{split}_visualization.png", dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {split}_visualization.png")
        plt.show()

def check_specific_images():
    """Check specific images to verify bounding boxes"""
    split_dir = Path("/home/root/dataset/DATASET2K_split")
    
    print("\n=== CHECKING SPECIFIC IMAGES ===")
    
    # Check a few specific images from each split
    test_images = {
        'train': ['IMG_3490.jpg', 'IMG_3938.jpg', 'IMG_20240913_110418.jpg'],
        'val': ['IMG_6338.jpg', 'IMG_3192.jpg', 'IMG_5923.jpg']
    }
    
    for split, images in test_images.items():
        print(f"\n--- {split.upper()} Split Specific Images ---")
        
        image_dir = split_dir / split / "images"
        label_dir = split_dir / split / "labels"
        
        for img_name in images:
            img_path = image_dir / img_name
            if not img_path.exists():
                print(f"  ❌ {img_name} not found")
                continue
                
            label_path = label_dir / img_path.with_suffix('.txt').name
            
            # Load image
            image = Image.open(img_path)
            w, h = image.size
            
            # Load labels
            bboxes = []
            classes = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            coords = list(map(float, parts[1:5]))
                            classes.append(cls_id)
                            bboxes.append(coords)
            
            print(f"  {img_name}:")
            print(f"    Image size: {w}x{h}")
            print(f"    Objects: {len(bboxes)}")
            
            for i, (cls_id, bbox) in enumerate(zip(classes, bboxes)):
                x_center, y_center, bbox_w, bbox_h = bbox
                x1 = int((x_center - bbox_w/2) * w)
                y1 = int((y_center - bbox_h/2) * h)
                x2 = int((x_center + bbox_w/2) * w)
                y2 = int((y_center + bbox_h/2) * h)
                
                print(f"    Object {i+1}: {CLASS_NAMES[cls_id]}")
                print(f"      BBox: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"      Size: {x2-x1}x{y2-y1} pixels")
                print(f"      Coords: {bbox}")

def validate_label_formats():
    """Validate that all labels have correct format"""
    split_dir = Path("/home/root/dataset/DATASET2K_split")
    
    print("\n=== VALIDATING LABEL FORMATS ===")
    
    issues = []
    
    for split in ['train', 'val']:
        label_dir = split_dir / split / "labels"
        label_files = list(label_dir.glob("*.txt"))
        
        print(f"\n{split.upper()}: Checking {len(label_files)} label files...")
        
        for label_file in label_files[:50]:  # Check first 50
            try:
                with open(label_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                for i, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"{label_file.name} line {i}: {len(parts)} parts (should be 5)")
                        continue
                    
                    try:
                        cls_id = int(parts[0])
                        coords = list(map(float, parts[1:5]))
                        
                        # Check class ID
                        if cls_id not in [0, 1, 2, 3, 4]:
                            issues.append(f"{label_file.name} line {i}: invalid class {cls_id}")
                        
                        # Check coordinate ranges
                        for j, coord in enumerate(coords):
                            if coord < 0 or coord > 1:
                                issues.append(f"{label_file.name} line {i}: coord {j} out of range: {coord}")
                                
                    except ValueError as e:
                        issues.append(f"{label_file.name} line {i}: invalid values - {e}")
                        
            except Exception as e:
                issues.append(f"{label_file.name}: {e}")
    
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  {issue}")
    else:
        print("✅ All labels have correct format!")

if __name__ == "__main__":
    # Validate label formats first
    validate_label_formats()
    
    # Check specific images
    check_specific_images()
    
    # Visualize random samples
    visualize_random_samples()