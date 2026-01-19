from pathlib import Path
from ultralytics import YOLO
import random
import shutil
from PIL import Image
import os

# -----------------------------
# Configuración del dataset
# -----------------------------
SRC_DATASET = Path("/home/root/dataset/DATASET2K/DATASET2K_split/DATASET2K_tiled")
DATA_CONFIG = {
    "train": SRC_DATASET / "train",
    "val":   SRC_DATASET / "val",
    "test":  SRC_DATASET / "test"
}

IMG_EXTS = [".jpg", ".jpeg", ".JPG", ".png"]

# -----------------------------
# Oversampling de clases minoritarias
# -----------------------------
def oversample_class(cls_name, target_count):
    """
    Duplica imágenes de una clase hasta target_count.
    cls_name debe ser exactamente como aparece en los labels YOLO.
    """
    train_dir = DATA_CONFIG["train"]
    labels_dir = train_dir / "labels"
    images_dir = train_dir / "images"

    # Buscar etiquetas con la clase
    lbl_files = [f for f in labels_dir.glob("*.txt") if any(line.split()[0] == cls_name for line in open(f))]
    current_count = len(lbl_files)
    if current_count >= target_count:
        print(f"[INFO] La clase {cls_name} ya tiene {current_count} ≥ {target_count}, no se duplicará.")
        return

    needed = target_count - current_count
    print(f"[INFO] Oversampling clase {cls_name}: {current_count} → {target_count} ({needed} copias)")

    for i in range(needed):
        src_lbl = random.choice(lbl_files)
        img_name = src_lbl.stem
        # Buscar imagen con alguna extensión
        src_img = None
        for ext in IMG_EXTS:
            candidate = images_dir / f"{img_name}{ext}"
            if candidate.exists():
                src_img = candidate
                break
        if src_img is None:
            continue

        # Copiar con nuevo nombre
        new_name = f"{img_name}_dup{i}"
        shutil.copy(src_lbl, labels_dir / f"{new_name}.txt")
        shutil.copy(src_img, images_dir / f"{new_name}{src_img.suffix}")

# -----------------------------
# Entrenamiento YOLOv8
# -----------------------------
def train_yolo(model_type="s", epochs=80, batch=16, imgsz=640):
    """
    Entrena YOLOv8 S o M en CPU con augmentations.
    """
    model = YOLO(f"yolov8{model_type}.pt")  # pre-trained S o M

    results = model.train(
        data="/home/root/dataset/DATASET2K/DATASET2K_split/data_tiled.yaml",  # <--- aquí el YAML
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device="cpu",
        project="runs",
        name=f"yolov8{model_type}_tiled_augmented",
        exist_ok=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        patience=20,
        warmup_epochs=3,
        augment=True,        # Habilita augmentations
        mosaic=0.3,          # Mosaico reducido (CPU)
        mixup=0.1,
        copy_paste=0.1,
        flipud=0.0,
        fliplr=0.5,
        degrees=10,
        translate=0.05,
        scale=0.2,
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
# Ejecutar pipeline
# -----------------------------
if __name__ == "__main__":
    # Configuración de oversampling según tus clases minoritarias
    # IDs según data.yaml: ['Boton', 'BrightRed C4', 'DarkRed C5', 'Green', 'Orange(red dot)']
    oversample_config = {
        "0": 5000,   # Boton
        "2": 5000,   # DarkRed C5
        "4": 3000    # Orange(red dot)
    }

    # 1️⃣ Oversampling
    for cls_id, target in oversample_config.items():
        oversample_class(cls_id, target)

    # 2️⃣ Entrenar YOLOv8 S
    train_yolo(model_type="s", epochs=80)
