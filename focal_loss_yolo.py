from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# Dataset
# -----------------------------
DATASET_PATH = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
DATA_YAML = DATASET_PATH / "data_final.yaml"

# -----------------------------
# Train
# -----------------------------
def train_yolo_m_big_machine():
    model = YOLO("yolov8m.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=120,
        batch=64,
        imgsz=640,
        device="cpu",

        project="runs",
        name="yolov8m_merged_noOS_bigCPU",
        exist_ok=True,

        # Optimización
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # 🔥 CLAVE: más peso a clasificación
        cls=1.5,          # default ~0.5
        box=7.5,
        dfl=1.5,

        # Augmentaciones suaves
        augment=True,
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,
        degrees=5,
        translate=0.05,
        scale=0.1,

        # Rendimiento (TU máquina brilla aquí)
        workers=96,
        cache="ram",
        amp=False,

        # Estabilidad
        warmup_epochs=5,
        patience=25,
        verbose=True,
        plots=True,
        val=True,
        save_period=20,
    )


if __name__ == "__main__":
    train_yolo_m_big_machine()
