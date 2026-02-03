import cv2
import random
from pathlib import Path
import shutil

# ---------------- CONFIG ----------------
DATASET = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
DARKRED_ID = 2
AUG_PER_IMAGE = 2
IMG_EXTS = [".jpg", ".JPG", ".jpeg", ".png"]

# ----------------------------------------

def find_image(images_dir, stem):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def augment_image(img):
    img = img.astype("float32")

    # brightness / contrast
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    img = img * alpha + beta

    # HSV
    hsv = cv2.cvtColor(img.clip(0,255).astype("uint8"), cv2.COLOR_BGR2HSV)
    hsv[...,1] = hsv[...,1] * random.uniform(0.9, 1.1)
    hsv[...,2] = hsv[...,2] * random.uniform(0.9, 1.1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # blur (light)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3,3), 0)

    return img.clip(0,255).astype("uint8")


def augment_split(split):
    labels_dir = DATASET / split / "labels"
    images_dir = DATASET / split / "images"

    darkred_labels = []
    for lbl in labels_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                if line.strip() and int(line.split()[0]) == DARKRED_ID:
                    darkred_labels.append(lbl)
                    break

    total = len(darkred_labels)
    print(f"🔹 {split}: augmenting {total} DarkRed images")

    for idx, lbl in enumerate(darkred_labels, 1):
        print(f"[{idx}/{total}] {lbl.stem}")

        img_path = find_image(images_dir, lbl.stem)
        if img_path is None:
            print("  ⚠ image not found")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print("  ⚠ failed to read image")
            continue

        for i in range(AUG_PER_IMAGE):
            aug = augment_image(img)
            new_name = f"{lbl.stem}_aug{i}"
            cv2.imwrite(str(images_dir / f"{new_name}{img_path.suffix}"), aug)
            shutil.copy(lbl, labels_dir / f"{new_name}.txt")

if __name__ == "__main__":
    augment_split("train")
