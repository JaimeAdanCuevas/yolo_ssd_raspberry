#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
DATASET_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
SPLITS = ["train", "val", "test"]

# Dataset actual (4 clases)
# 0 Boton
# 1 BrightRed
# 2 Green
# 3 Orange

# Nuevo mapeo (3 clases)
CLASS_MAP = {
    0: 0,  # Boton -> Immature
    2: 0,  # Green -> Immature
    1: 1,  # BrightRed C4
    3: 2   # Orange(red dot)
}

CLASS_NAMES = ["Immature", "BrightRed C4", "Orange(red dot)"]

# -----------------------------
# REMAP
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
                        continue

                    parts = line.split()
                    old_cls = int(parts[0])

                    if old_cls not in CLASS_MAP:
                        print(f"⚠ Clase desconocida {old_cls} en {lbl_file}")
                        continue

                    new_cls = CLASS_MAP[old_cls]
                    new_line = " ".join([str(new_cls)] + parts[1:])
                    new_lines.append(new_line)

            with open(lbl_file, "w") as f:
                if new_lines:
                    f.write("\n".join(new_lines) + "\n")
                else:
                    f.write("")

    print("✅ Merge completado correctamente.")


# -----------------------------
# VERIFICACIÓN
# -----------------------------
def verify_counts():
    print("\n📊 Conteo final por clase:")

    for split in SPLITS:
        counts = [0] * len(CLASS_NAMES)
        lbl_dir = DATASET_DIR / split / "labels"

        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    cls = int(line.split()[0])
                    counts[cls] += 1

        print(f"{split}: ", {i: counts[i] for i in range(len(CLASS_NAMES))})


# -----------------------------
# YAML
# -----------------------------
def create_yaml():
    yaml_path = DATASET_DIR / "data_3classes.yaml"

    content = f"""train: {DATASET_DIR}/train/images
val: {DATASET_DIR}/val/images
test: {DATASET_DIR}/test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"\n✅ YAML generado: {yaml_path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    remap_labels()
    verify_counts()
    create_yaml()
