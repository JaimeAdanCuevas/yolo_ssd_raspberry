import os
import cv2
import numpy as np
from pathlib import Path
import logging
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import shutil
import matplotlib.pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
import random
from torch.optim.lr_scheduler import StepLR


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- Data Validation --------------------
def validate_yolo_label(label_path):
    """Validate a YOLO format label file with detailed checks"""
    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    return False, f"Line {line_num}: Expected 5 values, got {len(parts)}"
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError as e:
                    return False, f"Line {line_num}: Invalid number format - {str(e)}"
                
                # Validate coordinate ranges
                coord_issues = []
                if not (0 <= x_center <= 1):
                    coord_issues.append(f"x_center={x_center}")
                if not (0 <= y_center <= 1):
                    coord_issues.append(f"y_center={y_center}")
                if not (0 <= width <= 1):
                    coord_issues.append(f"width={width}")
                if not (0 <= height <= 1):
                    coord_issues.append(f"height={height}")
                
                if coord_issues:
                    return False, f"Line {line_num}: Invalid coordinates - {', '.join(coord_issues)}"
                
                # Validate box dimensions
                if width <= 0 or height <= 0:
                    return False, f"Line {line_num}: Zero/negative box dimensions"
                
        return True, "Valid"
    except Exception as e:
        return False, f"File read error: {str(e)}"

# -------------------- Oversampling --------------------
def oversample_ripe_berries(train_images_dir, train_labels_dir, class_id=3):
    """Safe oversampling with validation and detailed logging"""
    train_images_dir = Path(train_images_dir)
    train_labels_dir = Path(train_labels_dir)
    
    logger.info("Starting oversampling process...")
    logger.debug(f"Image directory: {train_images_dir.resolve()}")
    logger.debug(f"Label directory: {train_labels_dir.resolve()}")

    # Remove existing copies
    for copy_file in train_images_dir.glob("*_copy[1-2].JPEG"):
        copy_file.unlink()
        logger.debug(f"Removed old image copy: {copy_file.name}")
    for copy_label in train_labels_dir.glob("*_copy[1-2].txt"):
        copy_label.unlink()
        logger.debug(f"Removed old label copy: {copy_label.name}")

    # Find valid ripe berry files
    valid_files = []
    for label_file in train_labels_dir.glob("*.txt"):
        if "_copy" in label_file.name:
            continue
        
        logger.debug(f"Checking {label_file.name} for class {class_id}")
        has_ripe = False
        is_valid = True
        
        # Validate label file
        valid_status, valid_msg = validate_yolo_label(label_file)
        if not valid_status:
            logger.warning(f"Skipping invalid {label_file.name}: {valid_msg}")
            continue
        
        with open(label_file, "r") as f:
            for line in f:
                if line.startswith(f"{class_id} "):
                    has_ripe = True
                    break
        
        if has_ripe:
            image_file = train_images_dir / f"{label_file.stem}.JPEG"
            if image_file.exists():
                valid_files.append((image_file, label_file))
                logger.debug(f"Found valid ripe berry pair: {image_file.name}")
            else:
                logger.warning(f"Image missing for {label_file.name}")

    logger.info(f"Found {len(valid_files)} valid ripe berry samples to oversample")

    # Create copies with validation
    for img_path, label_path in valid_files:
        for i in range(1, 3):
            copy_img = img_path.parent / f"{img_path.stem}_copy{i}.JPEG"
            copy_label = label_path.parent / f"{label_path.stem}_copy{i}.txt"
            
            try:
                # Validate original files
                if not img_path.exists():
                    logger.warning(f"Original image missing: {img_path.name}")
                    continue
                if not label_path.exists():
                    logger.warning(f"Original label missing: {label_path.name}")
                    continue
                
                # Create copies
                shutil.copy(img_path, copy_img)
                shutil.copy(label_path, copy_label)
                logger.debug(f"Created copy pair: {copy_img.name} / {copy_label.name}")
                
                # Validate new copies
                if not copy_img.exists():
                    logger.error(f"Failed to create image copy: {copy_img.name}")
                if not copy_label.exists():
                    logger.error(f"Failed to create label copy: {copy_label.name}")
                
            except Exception as e:
                logger.error(f"Copy failed for {img_path.name}: {str(e)}")

    new_count = len(list(train_images_dir.glob("*.JPEG")))
    logger.info(f"New training set size: {new_count} images")
    return new_count

# -------------------- Augmentation --------------------
def get_ripe_berries_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Blur(blur_limit=3, p=0.3)
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.4,
        min_area=0.0025  # 0.25% of image area
    ))
# -------------------- SSD Dataset --------------------
class SSDDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, augmentation=None, img_size=300):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.augmentation = augmentation
        self.img_size = img_size

        # Add base transform definition
        self.base_transform = A.Compose([
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                         border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        logger.info("Initializing SSD Dataset")
        logger.debug(f"Images directory: {self.images_dir.resolve()}")
        logger.debug(f"Labels directory: {self.labels_dir.resolve()}")
        logger.debug(f"Class names: {class_names}")
        logger.debug(f"Number of classes: {len(class_names)}")

        # Validate directory existence
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Collect valid image-label pairs
        self.image_files = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        logger.debug("Scanning for valid image files...")
        for ext in valid_extensions:
            for img_path in self.images_dir.glob(f"*{ext}"):
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                
                logger.debug(f"Checking pair: {img_path.name} -> {label_path.name}")
                
                # Check label existence
                if not label_path.exists():
                    logger.debug(f"Label missing for {img_path.name}")
                    continue
                
                # Validate label format
                is_valid, valid_msg = validate_yolo_label(label_path)
                if not is_valid:
                    logger.warning(f"Invalid label {label_path.name}: {valid_msg}")
                    continue
                
                # Check image validity
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Invalid image file: {img_path.name}")
                        continue
                    h, w = img.shape[:2]
                    if h == 0 or w == 0:
                        logger.warning(f"Empty image dimensions: {img_path.name}")
                        continue
                except Exception as e:
                    logger.error(f"Image load error {img_path.name}: {str(e)}")
                    continue
                
                self.image_files.append(img_path)
        
        logger.info(f"Found {len(self.image_files)} valid image-label pairs")
        
        if len(self.image_files) == 0:
            logger.error("No valid training samples found!")
            logger.error("Possible reasons:")
            logger.error("1. Mismatched image/label filenames")
            logger.error("2. Invalid YOLO format in label files")
            logger.error("3. Corrupted image files")
            logger.error("4. Incorrect directory structure")
            raise RuntimeError("No valid training samples available")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        # Load image and validate
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Enhanced YOLO label parsing with strict validation
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                
                # Structure validation
                if len(parts) != 5:
                    logger.warning(f"Invalid format {label_path.name}:L{line_num}")
                    continue
                
                # Value validation
                try:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    logger.warning(f"Invalid values {label_path.name}:L{line_num}")
                    continue
                
                # Coordinate validation
                if not all(0 < c <= 1 for c in coords):
                    logger.warning(f"Invalid coords {label_path.name}:L{line_num}")
                    continue
                
                # Dimension validation
                if coords[2] <= 0 or coords[3] <= 0:
                    logger.warning(f"Zero-size box {label_path.name}:L{line_num}")
                    continue
                
                boxes.append(coords)
                labels.append(class_id)

        # Augmentation pipeline with YOLO format
        if self.augmentation:
            transformed = self.augmentation(
                image=img,
                bboxes=boxes,
                class_labels=labels
             )
            if not transformed["bboxes"]:
                logger.warning(f"Augmentation remove all boxes for {img_path.name}")
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # Post-augmentation processing
        new_h, new_w = img.shape[:2]
        valid_boxes = []
        valid_labels = []
        
        for (x_center, y_center, width, height), label in zip(boxes, labels):
            # Convert to absolute coordinates
            xmin = (x_center - width/2) * new_w
            ymin = (y_center - height/2) * new_h
            xmax = (x_center + width/2) * new_w
            ymax = (y_center + height/2) * new_h
            
            # Post-augmentation validation
            if (xmax - xmin) < 2 or (ymax - ymin) < 2:
                continue
            if xmin >= xmax or ymin >= ymax:
                continue
            if any(val < 0 for val in [xmin, ymin]) or any(val > new_w for val in [xmax, ymax]):
                continue
                
            valid_boxes.append([xmin, ymin, xmax, ymax])
            valid_labels.append(label)

        # Final transforms and tensor conversion
        img = self.base_transform(image=img)["image"]
        boxes_tensor = torch.as_tensor(valid_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(valid_labels, dtype=torch.int64)
        
        # Fallback for empty boxes
        if boxes_tensor.shape[0] == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
            logger.warning(f"No valid boxes: {img_path.name}")

        return img, {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0]),
            "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64)
        }

# -------------------- SSD Model --------------------
def create_ssd_model(num_classes):
    logger.info("Creating SSD model")
    model = ssd300_vgg16(weights='DEFAULT')
    
    # Modify classification head
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    logger.debug(f"SSD model configuration:")
    logger.debug(f"- Input channels: {in_channels}")
    logger.debug(f"- Anchors per location: {num_anchors}")
    logger.debug(f"- Number of classes: {num_classes}")
    
    return model

# -------------------- Training --------------------
def train_ssd(data_yaml, output_dir, epochs=100, batch_size=64, img_size=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    # Load data config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    logger.debug("Data config loaded:")
    logger.debug(f"- Train: {data_config['train']}")
    logger.debug(f"- Val: {data_config['val']}")
    logger.debug(f"- Test: {data_config['test']}")
    logger.debug(f"- Classes: {data_config['names']}")

    # Create datasets
    def get_paths(split_key):
        images_dir = Path(data_config[split_key])
        labels_dir = images_dir.parent.parent / images_dir.parent.name / 'labels'
        logger.debug(f"Resolved paths for {split_key}:")
        logger.debug(f"- Images: {images_dir}")
        logger.debug(f"- Labels: {labels_dir}")
        return images_dir, labels_dir

    train_images_dir, train_labels_dir = get_paths('train')
    
    train_dataset = SSDDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        class_names=data_config['names'],
        augmentation=get_ripe_berries_augmentation(),
        img_size=img_size
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    logger.info(f"Training loader created with {len(train_dataset)} samples")

    # Model setup
    model = create_ssd_model(len(data_config['names']) + 1)  # +1 for background
    device = torch.device("cpu")
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    accum_iter = 4
    max_grad_norm = 5.0
    
    # Optimizer
    optimizer = SGD(params, lr=0.004, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    
    logger.info("Starting training...")
    
    # Training loop with stability checks
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        if epoch < 5:
            warmup_scheduler.step()

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Skip batches with invalid data
            if any(len(t["boxes"]) == 0 for t in targets):
                logger.warning("Skipping batch with empty targets")
                continue
                
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            
            # Check for NaN in losses
            losses = sum(loss for loss in loss_dict.values())
            if torch.isnan(losses):
                logger.error("NaN loss detected, skipping batch")
                continue
            
            skipped_batches = 0
            if any(len(t["boxes"]) == 0 for t in targets):
                skipped_batches += 1
                logger.warning(f"Skipping batch {batch_idx} with empty targets (total: {skipped_batches})")
                continue
            
            logger.info(f"Epoch {epoch+1}: Skipped {skipped_batches} batches")

            # Gradient accumulation
            losses = losses / accum_iter
            losses.backward()

            if (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.error(f"NaN gradients in {name}")
                        raise RuntimeError("NaN gradients in model parameters")

                optimizer.step()
                optimizer.zero_grad()

            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #optimizer.step()
            #optimizer.zero_grad()
            
            # Stability checks
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.error(f"NaN gradients in {name}")
                        raise RuntimeError("NaN gradients detected")
            
            current_loss = losses.item() * accum_iter
            epoch_loss += losses.item()
            
            # Batch logging with numerical sanity checks
            if batch_idx % 10 == 0:
                logger.debug(f"Epoch {epoch+1} Batch {batch_idx} Loss: {losses.item():.4f}")
                logger.debug(f"Max gradient: {max(p.grad.abs().max() for p in model.parameters() if p.grad is not None):.4f}")
                logger.debug(f"Weight norms: {[p.data.norm().item() for p in model.parameters()][:3]}")
        
        # Validation and learning rate adjustment
        val_metrics = evaluate_ssd(model, data_yaml, img_size, split='val')
        scheduler.step(val_metrics['map'])
        
        avg_loss = epoch_loss / len(train_loader)

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train loss: {avg_loss:.4f} | "
            f"Val mAP@50: {val_metrics['map50']:.4f} | "
            f"Current LR: {current_lr:.2e}"
            )
        
        # Early stopping check
        if current_lr < 1e-6:
            logger.warning("Learning rate too low, stopping early")
            break

        if val_metrics['map50'] == scheduler.best:
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            logger.info(f"Saved best model with mAP@50: {val_metrics['map50']:.4f}")

    return model

# -------------------- Evaluation --------------------
def evaluate_ssd(model, data_yaml, img_size=300, split='val'):
    model.eval()
    device = next(model.parameters()).device
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    test_images_dir = Path(data_config[split])
    test_labels_dir = test_images_dir.parent.parent / test_images_dir.parent.name / 'labels'
    
    test_dataset = SSDDataset(
        images_dir=test_images_dir,
        labels_dir=test_labels_dir,
        class_names=data_config['names'],
        img_size=img_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    metric = MeanAveragePrecision()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            preds = []
            for pred in predictions:
                preds.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
            
            formatted_targets = []
            for target in targets:
                formatted_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
            
            metric.update(preds, formatted_targets)
    
    results = metric.compute()
    return {
        'map': results['map'].item(),
        'map50': results['map_50'].item(),
        'map75': results['map_75'].item()
    }

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    try:
        # Configuration
        DATA_YAML = "/content/RaspberrySet/split/data.yaml"
        OUTPUT_DIR = "/content/drive/MyDrive/RaspberrySet/ssd_runs"
        
        # Oversample ripe berries
        logger.info("Starting oversampling process")
        oversample_ripe_berries(
            train_images_dir="/content/RaspberrySet/split/train/images",
            train_labels_dir="/content/RaspberrySet/split/train/labels",
            class_id=3  # Ripe Berries class ID
        )
        
        # Training
        logger.info("Starting SSD training pipeline")
        ssd_model = train_ssd(
            data_yaml=DATA_YAML,
            output_dir=OUTPUT_DIR,
            epochs=100,
            batch_size=64,
            img_size=300
        )
        
        # Final evaluation
        logger.info("Running final evaluation")
        metrics = evaluate_ssd(ssd_model, DATA_YAML, split='test')
        logger.info(f"Final mAP@50: {metrics['map50']:.4f}")
        logger.info(f"Final mAP@75: {metrics['map75']:.4f}")
        logger.info(f"Final mAP: {metrics['map']:.4f}")
        
    except Exception as e:
        logger.error(f"Critical error in main pipeline: {str(e)}")
        raise
