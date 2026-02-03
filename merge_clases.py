#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
DATASET_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split")

SPLITS = ["train", "val", "test"]

# Mapa de clases final: merge Boton+Green -> Immature (0), conservar Orange como 3
CLASS_MAP = {
    0: 0,  # Boton -> Immature
    3: 0,  # Green -> Immature
    1: 1,  # BrightRed C4
    2: 2,  # DarkRed C5
    4: 3   # Orange(red dot)
}

CLASS_NAMES = ["Immature", "BrightRed C4", "DarkRed C5", "Orange(red dot)"]

# -----------------------------
# FUNCIONES
# -----------------------------
def remap_labels():
    for split in SPLITS:
        lbl_dir = DATASET_DIR / split / "labels"
        print(f"🔹 Procesando {split} ({lbl_dir})")
        for lbl_file in lbl_dir.glob("*.txt"):
            new_lines = []
            with open(lbl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # saltar líneas vacías
                    parts = line.split()
                    try:
                        old_cls = int(parts[0])
                        if old_cls not in CLASS_MAP:
                            print(f"⚠️  Clase {old_cls} en {lbl_file} no está en CLASS_MAP, ignorando línea")
                            continue
                        new_cls = CLASS_MAP[old_cls]
                        new_line = " ".join([str(new_cls)] + parts[1:])
                        new_lines.append(new_line)
                    except ValueError:
                        print(f"⚠️  Línea corrupta en {lbl_file}: {line}")
                        continue
            # Sobrescribir archivo
            with open(lbl_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")
    print("✅ Labels remapeados correctamente.")

def verify_counts():
    print("\n📊 Conteo de instancias por clase después del remapeo:")
    for split in SPLITS:
        counts = [0] * len(CLASS_NAMES)
        lbl_dir = DATASET_DIR / split / "labels"
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cls = int(line.strip().split()[0])
                    counts[cls] += 1
        print(f"{split}: ", {i: counts[i] for i in range(len(CLASS_NAMES))})

def create_yaml():
    yaml_path = DATASET_DIR / "data_final.yaml"
    content = f"""train: {DATASET_DIR}/train/images
val: {DATASET_DIR}/val/images
test: {DATASET_DIR}/test/images
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\n✅ data_final.yaml generado en {yaml_path}")

# -----------------------------
# SCRIPT PRINCIPAL
# -----------------------------
if __name__ == "__main__":
    remap_labels()
    verify_counts()
    create_yaml()
