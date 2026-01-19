import cv2
from pathlib import Path

IMG_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split/train/images")
LBL_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split/train/labels")

OUT_DIR = Path("./debug_vis")
OUT_DIR.mkdir(exist_ok=True)

for img_path in list(IMG_DIR.glob("*.jpg"))[:10]:
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w, _ = img.shape

    with open(lbl_path) as f:
        for line in f:
            parts = line.split()
            cls = parts[0]
            x, y, bw, bh = map(float, parts[1:5])

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)

    cv2.imwrite(str(OUT_DIR / img_path.name), img)

print("✅ Imágenes con cajas guardadas en ./debug_vis")
