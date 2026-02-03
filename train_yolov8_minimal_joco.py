from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# Dataset
# -----------------------------
DATASET_PATH = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
DATA_YAML = DATASET_PATH / "data_final.yaml"  # 4 clases merged

# -----------------------------
# Entrenamiento YOLOv8
# -----------------------------
def train_yolo(model_type="m", epochs=100, batch=12, imgsz=640):

    model = YOLO(f"yolov8{model_type}.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device="cpu",
        project="runs",
        name="yolov8m_merged_boton-green_darkred-focus_oversampling_augmentation",
        exist_ok=True,

        optimizer="SGD",
        lr0=0.008,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        patience=20,
        warmup_epochs=3,

        augment=True,
        mosaic=0.15,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.05,

        fliplr=0.5,
        flipud=0.0,
        degrees=10,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,

        workers=8,
        amp=False,
        verbose=True,
        plots=True,
        save_period=25,
        val=True,
    )

    return results


# -----------------------------
# Ejecutar
# -----------------------------
if __name__ == "__main__":
    train_yolo(model_type="m", epochs=100)
