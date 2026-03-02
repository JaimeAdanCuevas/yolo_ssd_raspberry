#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# 🔧 Ruta dataset ORIGINAL
DATASET_DIR = Path("/home/root/dataset/DATASET2K")

IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"

def segment_to_box(coords):
    xs = coords[0::2]
    ys = coords[1::2]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

def clean_labels():
    total = 0
    segments_found = 0
    segments_converted = 0
    corrupt_lines = 0
    empty_labels = 0

    print("🔹 Limpiando labels originales...\n")

    for lbl_file in LBL_DIR.glob("*.txt"):
        total += 1
        new_lines = []

        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    # bounding box normal
                    if len(coords) == 4:
                        new_lines.append(line.strip())

                    # segment → convertir
                    elif len(coords) > 4:
                        segments_found += 1
                        box = segment_to_box(coords)
                        new_line = " ".join([str(cls)] + [f"{v:.6f}" for v in box])
                        new_lines.append(new_line)
                        segments_converted += 1

                except:
                    corrupt_lines += 1

        if not new_lines:
            empty_labels += 1

        with open(lbl_file, "w") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")

    print("✅ LIMPIEZA COMPLETADA\n")
    print("Archivos procesados:", total)
    print("Segmentos encontrados:", segments_found)
    print("Segmentos convertidos:", segments_converted)
    print("Líneas corruptas ignoradas:", corrupt_lines)
    print("Labels vacíos:", empty_labels)

if __name__ == "__main__":
    clean_labels()
