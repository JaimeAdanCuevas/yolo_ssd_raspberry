from pathlib import Path
import random
import shutil

# ---------------- CONFIG ----------------
DATASET = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
TARGET_COUNT = 3000
DARKRED_ID = 2
IMG_EXTS = [".jpg", ".JPG", ".jpeg", ".png"]

# ----------------------------------------

def find_image(images_dir, stem):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def oversample_split(split):
    labels_dir = DATASET / split / "labels"
    images_dir = DATASET / split / "images"

    darkred_labels = []
    for lbl in labels_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                if line.strip() and int(line.split()[0]) == DARKRED_ID:
                    darkred_labels.append(lbl)
                    break

    current = len(darkred_labels)
    print(f"🔹 {split}: DarkRed samples = {current}")

    if current >= TARGET_COUNT:
        print(f"✅ {split}: ya cumple target")
        return

    needed = TARGET_COUNT - current
    print(f"➕ Oversampling {needed} samples...")

    for i in range(needed):
        src_lbl = random.choice(darkred_labels)
        src_img = find_image(images_dir, src_lbl.stem)
        if src_img is None:
            continue

        new_name = f"{src_lbl.stem}_os{i}"
        shutil.copy(src_lbl, labels_dir / f"{new_name}.txt")
        shutil.copy(src_img, images_dir / f"{new_name}{src_img.suffix}")


if __name__ == "__main__":
    for split in ["train"]:
        oversample_split(split)
