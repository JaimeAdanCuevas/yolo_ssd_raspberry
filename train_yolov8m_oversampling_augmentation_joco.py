# train_yolov8m_oversampling_augmentation_joco.py - BULLETPROOF VERSION
import torch
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import colorstr
import albumentations as A

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

GREEN_ID = 3

def green_augs():
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.3),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.5))  # Higher min_visibility


def ultra_safe_clip_bboxes(bboxes, cls):
    """Ultra-safe bbox clipping that guarantees same counts"""
    if bboxes.numel() == 0:
        return bboxes, cls

    device = bboxes.device
    bboxes_np = bboxes.cpu().numpy()
    cls_np = cls.cpu().numpy().flatten()
    
    # CRITICAL: Ensure counts match from the start
    if bboxes_np.shape[0] != cls_np.shape[0]:
        log.error(f"CRITICAL: Initial bbox/cls count mismatch: {bboxes_np.shape[0]} vs {cls_np.shape[0]}")
        # Return empty to avoid crash
        return torch.empty((0, 4), device=device), torch.empty((0, 1), device=device)
    
    # Handle empty case
    if bboxes_np.shape[0] == 0:
        return torch.empty((0, 4), device=device), torch.empty((0, 1), device=device)
    
    # Clip coordinates
    x_center, y_center, w, h = bboxes_np.T
    
    x_min = np.clip(x_center - w / 2, 0.0, 1.0)
    x_max = np.clip(x_center + w / 2, 0.0, 1.0)
    y_min = np.clip(y_center - h / 2, 0.0, 1.0)
    y_max = np.clip(y_center + h / 2, 0.0, 1.0)
    
    w_new = np.maximum(x_max - x_min, 0.001)
    h_new = np.maximum(y_max - y_min, 0.001)
    
    x_center_new = x_min + w_new / 2
    y_center_new = y_min + h_new / 2
    
    bboxes_np = np.stack([x_center_new, y_center_new, w_new, h_new], axis=1)
    
    # FINAL VALIDATION: This should never fail
    if bboxes_np.shape[0] != cls_np.shape[0]:
        log.error(f"CRITICAL: Clipping caused count mismatch! This should never happen.")
        return torch.empty((0, 4), device=device), torch.empty((0, 1), device=device)
    
    # Convert back to torch
    bboxes_new = torch.from_numpy(bboxes_np).float().to(device)
    cls_new = torch.from_numpy(cls_np).unsqueeze(1).float().to(device)
    
    return bboxes_new, cls_new


class GreenDataset(YOLODataset):
    def __init__(self, augmentation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = augmentation
        # Don't pre-find green indices to avoid loading issues

    def __getitem__(self, idx):
        # Get original item
        item = super().__getitem__(idx)
        
        # Always clip bboxes first (for ALL images)
        item["bboxes"], item["cls"] = ultra_safe_clip_bboxes(item["bboxes"], item["cls"])
        
        # Skip augmentation if no bboxes or no augmentation
        if item["bboxes"].shape[0] == 0 or not self.aug:
            return item
        
        # Check if this image has green tomatoes
        has_green = False
        try:
            cls_np = item["cls"].cpu().numpy().flatten().astype(int)
            has_green = any(cls_id == GREEN_ID for cls_id in cls_np)
        except:
            has_green = False
        
        # Only apply augmentation to images with green tomatoes (20% chance)
        if not has_green or torch.rand(1) < 0.8:  # 20% chance of augmentation
            return item

        try:
            # Convert to Albumentations format
            img = (item["img"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            bboxes = item["bboxes"].cpu().numpy()
            cls = item["cls"].cpu().numpy().flatten().astype(int)
            
            # Store original for fallback
            original_bboxes = bboxes.copy()
            original_cls = cls.copy()
            
            # Apply augmentation
            aug = self.aug(image=img, bboxes=bboxes, class_labels=cls)
            img = aug["image"]
            
            # SAFEST POSSIBLE HANDLING
            aug_bboxes_list = aug["bboxes"]
            aug_cls_list = aug["class_labels"]
            
            # Convert safely
            if aug_bboxes_list is None or aug_cls_list is None:
                return item  # Use original
            
            aug_bboxes = np.array(aug_bboxes_list)
            aug_cls = np.array(aug_cls_list)
            
            # CRITICAL: If counts don't match, use original
            if aug_bboxes.shape[0] != aug_cls.shape[0]:
                log.debug(f"Augmentation caused count mismatch, using original")
                return item
            
            # If augmentation removed all bboxes, use original
            if aug_bboxes.shape[0] == 0:
                return item
            
            # Clip augmented bboxes
            aug_bboxes_torch = torch.from_numpy(aug_bboxes).float()
            aug_cls_torch = torch.from_numpy(aug_cls).unsqueeze(1).float()
            aug_bboxes_torch, aug_cls_torch = ultra_safe_clip_bboxes(aug_bboxes_torch, aug_cls_torch)
            
            # FINAL VALIDATION
            if aug_bboxes_torch.shape[0] != aug_cls_torch.shape[0]:
                log.warning(f"Final validation failed, using original")
                return item
            
            # Update item with augmented data
            item["img"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            item["bboxes"] = aug_bboxes_torch
            item["cls"] = aug_cls_torch

        except Exception as e:
            log.debug(f"Augmentation failed for image {idx}: {e}")
            # Return original item (already clipped)

        return item


class GreenTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        aug = green_augs() if mode == "train" else None
        return GreenDataset(
            augmentation=aug,
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=False,
            cache=self.args.cache,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )


def train(yaml_path, out_dir="runs", epochs=150, batch=64, img_sz=640):
    out = Path(out_dir) / "yolov8m_dataset2k"
    last_pt = out / "weights" / "last.pt"
    resume = last_pt.exists()

    overrides = {
        "data": yaml_path,
        "epochs": epochs,
        "batch": batch,
        "imgsz": img_sz,
        "model": "yolov8m.pt",
        "device": "cpu",
        "project": str(out.parent),
        "name": out.name,
        "exist_ok": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "patience": 30,
        "augment": True,
        "mosaic": 0.7,
        "resume": resume,
        "workers": 8,
        "cache": "disk",
        "amp": True,
    }
    if resume:
        overrides["resume"] = str(last_pt)

    trainer = GreenTrainer(overrides=overrides)
    trainer.train()
    best = Path(trainer.best)
    log.info(f"Training finished – best model: {best}")
    return YOLO(best)


if __name__ == "__main__":
    model = train("/home/root/dataset/DATASET2K_split/data.yaml")