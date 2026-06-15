#!/usr/bin/env python3
from pathlib import Path
import sys

# -----------------------------
# Configuración del dataset
# -----------------------------
DATASET_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
SPLITS = ["train", "val", "test"]

# Clases según data.yaml actual
CLASSES = ['Bud', 'BrightRed C4', 'DarkRed C5', 'Green', 'Orange(red dot)']

# -----------------------------
# Funciones
# -----------------------------
def check_missing_labels(split_dir):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    missing_labels = []
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_labels.append(lbl_path)
    return missing_labels

def count_classes(split_dir):
    labels_dir = split_dir / "labels"
    class_counts = {i:0 for i in range(len(CLASSES))}
    for lbl_path in labels_dir.glob("*.txt"):
        with open(lbl_path, "r") as f:
            for line in f:
                try:
                    cls_id = int(line.split()[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                    else:
                        print(f"[WARNING] Clase desconocida {cls_id} en {lbl_path}")
                except Exception as e:
                    print(f"[ERROR] No se pudo leer línea en {lbl_path}: {line}")
    return class_counts

# -----------------------------
# Script principal
# -----------------------------
if __name__ == "__main__":
    print("🔹 Verificando dataset en:", DATASET_DIR, "\n")
    for split in SPLITS:
        split_dir = DATASET_DIR / split
        print(f"📂 Split: {split}")

        # 1️⃣ Missing labels
        missing = check_missing_labels(split_dir)
        if missing:
            print(f"⚠️ Imágenes sin label ({len(missing)}):")
            for lbl in missing:
                print("  ", lbl)
        else:
            print("✅ Todas las imágenes tienen su label.")

        # 2️⃣ Conteo de instancias por clase
        counts = count_classes(split_dir)
        print("📊 Conteo de instancias por clase:")
        for cls_id, cnt in counts.items():
            print(f"  {cls_id} ({CLASSES[cls_id]}): {cnt}")
        print("-"*40)
