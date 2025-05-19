import os
import cv2
import numpy as np
from pathlib import Path
import logging
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import ToTensor
import albumentations as A
import xml.etree.ElementTree as ET
import shutil
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import yaml
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Convert YOLO to VOC XML format
def yolo_to_voc(yolo_label_path, image_path, output_xml_path, class_names):
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Failed to load image {image_path}")
        return
    height, width, _ = img.shape
    
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = image_path.name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    
    with open(yolo_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            
            x_min = int((x_center - w/2) * width)
            y_min = int((y_center - h/2) * height)
            x_max = int((x_center + w/2) * width)
            y_max = int((y_center + h/2) * height)
            
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = class_names[class_id]
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x_min)
            ET.SubElement(bndbox, "ymin").text = str(y_min)
            ET.SubElement(bndbox, "xmax").text = str(x_max)
            ET.SubElement(bndbox, "ymax").text = str(y_max)
    
    tree = ET.ElementTree(annotation)
    tree.write(str(output_xml_path))
    logger.info(f"Created VOC XML at {output_xml_path}")

# Oversample ripe_berries images
def oversample_ripe_berries(train_images_dir, train_labels_dir, train_xml_dir, class_names):
    train_images_dir = Path(train_images_dir)
    train_labels_dir = Path(train_labels_dir)
    train_xml_dir = Path(train_xml_dir)
    train_xml_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing copies
    for copy_file in train_images_dir.glob("*_copy[1-2].JPEG"):
        copy_file.unlink()
        logger.info(f"Removed {copy_file}")
    for copy_label in train_labels_dir.glob("*_copy[1-2].txt"):
        copy_label.unlink()
        logger.info(f"Removed {copy_label}")
    for copy_xml in train_xml_dir.glob("*_copy[1-2].xml"):
        copy_xml.unlink()
        logger.info(f"Removed {copy_xml}")
    
    ripe_berries_files = []
    for label_file in train_labels_dir.glob("*.txt"):
        if "_copy" in label_file.name:
            continue
        with open(label_file, "r") as f:
            for line in f:
                if line.startswith("3 "):  # Class ID 3 = Ripe Berries
                    ripe_berries_files.append(label_file)
                    break
    
    for label_file in ripe_berries_files:
        image_file = train_images_dir / label_file.name.replace(".txt", ".JPEG")
        if not image_file.exists():
            logger.warning(f"Image {image_file} not found for label {label_file}")
            continue
        # Generate VOC XML for original file
        xml_file = train_xml_dir / label_file.name.replace(".txt", ".xml")
        yolo_to_voc(label_file, image_file, xml_file, class_names)
        for i in range(1, 3):
            copy_label = label_file.parent / f"{label_file.stem}_copy{i}{label_file.suffix}"
            copy_image = image_file.parent / f"{image_file.stem}_copy{i}{image_file.suffix}"
            copy_xml = train_xml_dir / f"{label_file.stem}_copy{i}.xml"
            shutil.copy(label_file, copy_label)
            shutil.copy(image_file, copy_image)
            yolo_to_voc(copy_label, copy_image, copy_xml, class_names)
            logger.info(f"Copied {image_file} to {copy_image}, {label_file} to {copy_label}, created {copy_xml}")
    
    new_count = len(list(train_images_dir.glob("*.JPEG")))
    logger.info(f"New training set size: {new_count} images")
    return new_count

# Augmentation pipeline
def get_augmentation():
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))

# Custom dataset for SSD
class RaspberryDataset(Dataset):
    def __init__(self, images_dir, xml_dir, class_names, transform=None):
        self.images_dir = Path(images_dir)
        self.xml_dir = Path(xml_dir)
        self.class_names = class_names
        self.transform = transform
        self.image_files = sorted(list(self.images_dir.glob("*.JPEG")))
        self.ripe_berries_images = self._find_ripe_berries_images()

    def _find_ripe_berries_images(self):
        ripe_berries_images = []
        for xml_file in self.xml_dir.glob("*.xml"):
            tree = ET.parse(xml_file)
            for obj in tree.findall("object"):
                if obj.find("name").text == "Ripe Berries":  # Match data.yaml
                    ripe_berries_images.append(xml_file.name.replace(".xml", ".JPEG"))
                    break
        return set(ripe_berries_images)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        xml_path = self.xml_dir / img_path.name.replace(".JPEG", ".xml")
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        if xml_path.exists():
            tree = ET.parse(xml_path)
            for obj in tree.findall("object"):
                name = obj.find("name").text
                bbox = obj.find("bndbox")
                x_min = float(bbox.find("xmin").text)
                y_min = float(bbox.find("ymin").text)
                x_max = float(bbox.find("xmax").text)
                y_max = float(bbox.find("ymax").text)
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(self.class_names.index(name) + 1)  # +1 for background class
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        if img_path.name in self.ripe_berries_images and self.transform:
            augmented = self.transform(image=img, bboxes=boxes, category_id=labels)
            img = augmented["image"]
            boxes = np.array(augmented["bboxes"], dtype=np.float32)
            labels = np.array(augmented["category_id"], dtype=np.int64)
        
        img = ToTensor()(img)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }
        
        return img, target

# Evaluation function
def evaluate_ssd(model, data_loader, class_names, device):
    model.eval()
    coco_results = []
    image_id = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.5:
                        continue
                    x_min, y_min, x_max, y_max = box
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": float(score)
                    })
                image_id += 1
    
    # Create COCO-format ground truth
    coco_gt = {
        "images": [{"id": i} for i in range(image_id)],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    }
    ann_id = 1
    image_id = 0
    for _, targets in data_loader:
        for target in targets:
            boxes = target["boxes"].numpy()
            labels = target["labels"].numpy()
            for box, label in zip(boxes, labels):
                x_min, y_min, x_max, y_max = box
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                })
                ann_id += 1
            image_id += 1
    
    # Save COCO ground truth and predictions
    import json
    with open("coco_gt.json", "w") as f:
        json.dump(coco_gt, f)
    with open("coco_pred.json", "w") as f:
        json.dump(coco_results, f)
    
    coco_gt = COCO("coco_gt.json")
    coco_dt = coco_gt.loadRes("coco_pred.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    ap50 = coco_eval.stats[1]  # mAP@50
    per_class_ap = coco_eval.params.catIds
    per_class_results = {}
    for i, cat_id in enumerate(per_class_ap):
        per_class_results[class_names[cat_id - 1]] = coco_eval.stats[8 + i]
    
    return ap50, per_class_results

# Visualize prediction
def visualize_prediction(model, image_path, output_dir, class_names, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)[0]
    
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    
    ripe_berries_count = 0
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.5:
            continue
        if label - 1 == class_names.index("Ripe Berries"):  # Adjust for background class
            ripe_berries_count += 1
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{class_names[label - 1]} {score:.2f}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {ripe_berries_count} Ripe Berries")
    output_path = output_dir / f"prediction_{Path(image_path).name}"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved prediction visualization at {output_path} with {ripe_berries_count} Ripe Berries")
    
    return output_path

# Training function
def train_ssd(data_yaml, output_dir, test_image, epochs=100, batch_size=32, img_size=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names from data.yaml
    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config["names"]
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")
    
    data_dir = Path(data_yaml).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert YOLO to VOC for all splits
    for split in ["train", "val", "test"]:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        xml_dir = data_dir / split / "annotations"
        xml_dir.mkdir(parents=True, exist_ok=True)
        for label_file in labels_dir.glob("*.txt"):
            image_file = images_dir / label_file.name.replace(".txt", ".JPEG")
            xml_file = xml_dir / label_file.name.replace(".txt", ".xml")
            if image_file.exists():
                yolo_to_voc(label_file, image_file, xml_file, class_names)
    
    # Oversample train set
    train_images_dir = data_dir / "train" / "images"
    train_labels_dir = data_dir / "train" / "labels"
    train_xml_dir = data_dir / "train" / "annotations"
    oversample_ripe_berries(train_images_dir, train_labels_dir, train_xml_dir, class_names)
    
    # Datasets and loaders
    train_dataset = RaspberryDataset(
        images_dir=train_images_dir,
        xml_dir=train_xml_dir,
        class_names=class_names,
        transform=get_augmentation()
    )
    val_dataset = RaspberryDataset(
        images_dir=data_dir / "val" / "images",
        xml_dir=data_dir / "val" / "annotations",
        class_names=class_names,
        transform=None
    )
    test_dataset = RaspberryDataset(
        images_dir=data_dir / "test" / "images",
        xml_dir=data_dir / "test" / "annotations",
        class_names=class_names,
        transform=None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    # Load SSD300 model
    model = ssd300_vgg16(weights="DEFAULT", num_classes=len(class_names) + 1)  # +1 for background
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Evaluate on test set
    ap50, per_class_ap = evaluate_ssd(model, test_loader, class_names, device)
    logger.info(f"Test mAP@50: {ap50:.4f}")
    for cls, ap in per_class_ap.items():
        logger.info(f"Class {cls} AP@50: {ap:.4f}")
    
    # Save model
    model_path = output_dir / "ssd_raspberry.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved SSD model at {model_path}")
    
    # Visualize test image
    if Path(test_image).exists():
        output_path = visualize_prediction(model, test_image, output_dir, class_names, device)
        logger.info(f"Visualized prediction saved at {output_path}")
    else:
        logger.warning(f"Test image {test_image} not found")
    
    return model

# Main execution
try:
    data_yaml = "/content/RaspberrySet/split/data.yaml"
    output_dir = "/content/drive/MyDrive/RaspberrySet/runs"
    test_image = "/content/RaspberrySet/split/test/images/IMG_3532.JPEG"
    
    model = train_ssd(data_yaml, output_dir, test_image, epochs=100, batch_size=32, img_size=300)
except Exception as e:
    logger.error(f"Error during SSD training: {e}")