#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
DATASET_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
SPLITS = ["train", "val", "test"]

# Clase a eliminar
REMOVE_CLASS = 2  # DarkRed C5

# Nuevas clases después de eliminar DarkRed
CLASS_NAMES = [
    "Boton",
    "BrightRed C4",
    "Green",
    "Orange(red dot)"
]

# -----------------------------
# FUNCIONES
# -----------------------------
def remove_and_remap_labels():
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
                    cls = int(parts[0])

                    # Eliminar DarkRed
                    if cls == REMOVE_CLASS:
                        continue

                    # Reindexar clases mayores
                    if cls > REMOVE_CLASS:
                        cls -= 1

                    new_line = " ".join([str(cls)] + parts[1:])
                    new_lines.append(new_line)

            with open(lbl_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")

    print("✅ DarkRed eliminado y labels reindexados correctamente.")

def verify_counts():
    print("\n📊 Conteo de instancias por clase:")
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

def create_yaml():
    yaml_path = DATASET_DIR / "data_no_darkred.yaml"
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
    remove_and_remap_labels()
    verify_counts()
    create_yaml()
