# train_optimized_cpu.py
from ultralytics import YOLO
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def train_optimized_cpu():
    data_path = "/home/root/dataset/DATASET2K/DATASET2K_split/data.yaml"
    
    log.info("=== OPTIMIZED CPU TRAINING ===")
    
    # Use smaller model for CPU training
    model = YOLO("yolov8s.pt")  # Small model is much faster on CPU
    
    results = model.train(
        data="/home/root/dataset/DATASET2K/DATASET2K_split/data.yaml",
        epochs=80,
        batch=16,
        imgsz=640,
        device="cpu",
        project="runs",
        name="yolov8s_base_final",
        exist_ok=True,

        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,

        warmup_epochs=3,
        patience=15,
        cos_lr=True,

        # Augmentación ligera (fondo verde)
        augment=True,
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,

        fliplr=0.5,
        flipud=0.0,

        degrees=3.0,
        translate=0.03,
        scale=0.15,
        shear=0.0,
        perspective=0.0,

        workers=8,
        cache=False,
        amp=False,
        plots=True,
        save_period=20,
        val=True,
    )
    
    log.info("✅ Training completed!")
    return model

if __name__ == "__main__":
    model = train_optimized_cpu()