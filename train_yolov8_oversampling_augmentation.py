import os
import cv2
import numpy as np
from pathlib import Path
import logging
from ultralytics import YOLO
import matplotlib.pyplot as plt
from IPython.display import Image
import albumentations as A
import shutil
import torch
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Oversample ripe_berries images
def oversample_ripe_berries(train_images_dir, train_labels_dir):
    train_images_dir = Path(train_images_dir)
    train_labels_dir = Path(train_labels_dir)
    
    # Remove existing copies
    for copy_file in train_images_dir.glob("*_copy[1-2].JPEG"):
        copy_file.unlink()
        logger.info(f"Removed {copy_file}")
    for copy_label in train_labels_dir.glob("*_copy[1-2].txt"):
        copy_label.unlink()
        logger.info(f"Removed {copy_label}")
    
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
        for i in range(1, 3):
            copy_label = label_file.parent / f"{label_file.stem}_copy{i}{label_file.suffix}"
            copy_image = image_file.parent / f"{image_file.stem}_copy{i}{image_file.suffix}"
            shutil.copy(label_file, copy_label)
            shutil.copy(image_file, copy_image)
            logger.info(f"Copied {image_file} to {copy_image} and {label_file} to {copy_label}")
    
    new_count = len(list(train_images_dir.glob("*.JPEG")))
    logger.info(f"New training set size: {new_count} images")
    return new_count

# Define augmentation pipeline for ripe_berries images
def get_ripe_berries_augmentation():
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))

# Custom dataset loader for selective augmentation
class RipeBerriesDataset:
    def __init__(self, images_dir, labels_dir, augmentation=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.augmentation = augmentation
        self.image_files = sorted(list(self.images_dir.glob("*.JPEG")))
        self.ripe_berries_images = self._find_ripe_berries_images()

    def _find_ripe_berries_images(self):
        ripe_berries_images = []
        for label_file in self.labels_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    if line.startswith("3 "):  # Class ID 3 = Ripe Berries
                        img_name = label_file.name.replace(".txt", ".JPEG")
                        ripe_berries_images.append(img_name)
                        break
        return set(ripe_berries_images)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / img_path.name.replace(".JPEG", ".txt")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        class_labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    boxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        if img_path.name in self.ripe_berries_images and self.augmentation:
            augmented = self.augmentation(image=img, bboxes=boxes, class_labels=class_labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            class_labels = augmented["class_labels"]
        
        return img, boxes, class_labels

def train_yolov8_with_augmentation(data_yaml, output_dir, epochs=100, batch_size=64, img_size=640):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data.yaml to get class names
    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config["names"]
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")

    try:
        model = YOLO("yolov8m.pt")
        logger.info("Loaded YOLOv8m model")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8m model: {e}")
        raise

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=0 if torch.cuda.is_available() else "cpu",
            project=str(output_dir),
            name="yolov8m_raspberry_oversampled_augmented",
            exist_ok=True,
            optimizer="SGD",
            lr0=0.01,
            patience=20,
            verbose=True,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.5  # Reduced to balance albumentations
        )
        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    best_model_path = output_dir / "yolov8m_raspberry_oversampled_augmented" / "weights" / "best.pt"
    results_csv = output_dir / "yolov8m_raspberry_oversampled_augmented" / "results.csv"
    logger.info(f"Best model saved at {best_model_path}")
    logger.info(f"Training metrics saved at {results_csv}")

    try:
        metrics = model.val(data=data_yaml, split="test")
        logger.info(f"Test mAP@50: {metrics.box.map50:.4f}, mAP@50:95: {metrics.box.map:.4f}")
        for i, ap in enumerate(metrics.box.maps):
            logger.info(f"Class {class_names[i]} AP@50: {ap:.4f}")
    except Exception as e:
        logger.error(f"Test validation failed: {e}")

    return model

def visualize_prediction(model, image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = model.predict(image_path, conf=0.5, save=False)
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ripe_berries_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                if result.names[cls_id] == "Ripe Berries":  # Match data.yaml
                    ripe_berries_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf.item()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{result.names[cls_id]} {conf:.2f}",
                    (x1, y1 - 10),
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
    except Exception as e:
        logger.error(f"Prediction visualization failed: {e}")
        return None

# Main execution
try:
    # Define paths
    data_yaml = "/content/RaspberrySet/split/data.yaml"
    output_dir = "/content/drive/MyDrive/RaspberrySet/runs"
    test_image = "/content/RaspberrySet/split/test/images/IMG_3532.JPEG"
    train_images_dir = "/content/RaspberrySet/split/train/images"
    train_labels_dir = "/content/RaspberrySet/split/train/labels"

    # Verify test image exists
    if not Path(test_image).exists():
        logger.error(f"Test image {test_image} not found")
        test_images = list(Path("/content/RaspberrySet/split/test/images").glob("*.JPEG"))
        if test_images:
            test_image = str(test_images[0])
            logger.info(f"Using fallback test image: {test_image}")
        else:
            raise FileNotFoundError("No test images found")

    # Oversample ripe_berries images
    logger.info("Starting oversampling of Ripe Berries images")
    new_train_size = oversample_ripe_berries(train_images_dir, train_labels_dir)

    # Train with augmentations
    logger.info("Starting training with augmentations")
    model = train_yolov8_with_augmentation(data_yaml, output_dir, epochs=100, batch_size=64, img_size=640)

    # Visualize a test image
    output_path = visualize_prediction(model, test_image, output_dir)
    if output_path:
        logger.info(f"Visualized prediction saved at {output_path}")
except Exception as e:
    logger.error(f"Error during oversampling, training, or prediction: {e}")
