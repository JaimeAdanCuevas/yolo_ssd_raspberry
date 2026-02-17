#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

DATASET_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
SPLITS = ["train", "val", "test"]

# Fix final de clases
CLASS_FIX = {
    2: 1,  # DarkRed -> Red (por si queda alguno)
    4: 2   # Orange -> Orange (nuevo índice)
}

def fix_labels():
    for split in SPLITS:
        lbl_dir = DATASET_DIR / split / "labels"
        print(f"🔧 Fixing {split}")

        for lbl_file in lbl_dir.glob("*.txt"):
            new_lines = []

            with open(lbl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    old_cls = int(parts[0])
                    new_cls = CLASS_FIX.get(old_cls, old_cls)

                    new_lines.append(
                        " ".join([str(new_cls)] + parts[1:])
                    )

            with open(lbl_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")

    print("✅ Fix de clases completado")

if __name__ == "__main__":
    fix_labels()
