import cv2
from pathlib import Path

# ===============================
# CONFIGURACIÓN
# ===============================

SRC_DATASET = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
DST_DATASET = Path("/home/root/dataset/DATASET2K/DATASET2K_split/DATASET2K_tiled")

IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".png")

TILE_ROWS = 2
TILE_COLS = 2
MIN_BOX_AREA_RATIO = 0.001  # filtra cajas basura

# ===============================
# FUNCIONES
# ===============================

def yolo_to_xyxy(box, w, h):
    x, y, bw, bh = box
    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    x = (x1 + x2) / 2 / w
    y = (y1 + y2) / 2 / h
    return x, y, bw, bh


def tile_image(img_path, lbl_path, dst_img_dir, dst_lbl_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w, _ = img.shape
    tile_w = w // TILE_COLS
    tile_h = h // TILE_ROWS

    labels = []
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                # ✅ FIX CRÍTICO:
                # usar solo: class + x y w h
                if len(parts) < 5:
                    continue

                cls = int(parts[0])
                box = list(map(float, parts[1:5]))
                labels.append((cls, box))

    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            x0 = c * tile_w
            y0 = r * tile_h
            x1 = x0 + tile_w
            y1 = y0 + tile_h

            tile = img[y0:y1, x0:x1]
            tile_name = f"{img_path.stem}_r{r}_c{c}{img_path.suffix}"

            out_labels = []

            for cls, box in labels:
                bx1, by1, bx2, by2 = yolo_to_xyxy(box, w, h)

                ix1 = max(bx1, x0)
                iy1 = max(by1, y0)
                ix2 = min(bx2, x1)
                iy2 = min(by2, y1)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                area = (ix2 - ix1) * (iy2 - iy1)
                tile_area = tile_w * tile_h
                if area / tile_area < MIN_BOX_AREA_RATIO:
                    continue

                nx1 = ix1 - x0
                ny1 = iy1 - y0
                nx2 = ix2 - x0
                ny2 = iy2 - y0

                x, y, bw, bh = xyxy_to_yolo(nx1, ny1, nx2, ny2, tile_w, tile_h)
                out_labels.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            cv2.imwrite(str(dst_img_dir / tile_name), tile)

            if out_labels:
                with open(dst_lbl_dir / tile_name.replace(img_path.suffix, ".txt"), "w") as f:
                    f.write("\n".join(out_labels))


def process_split(split):
    src_img_dir = SRC_DATASET / split / "images"
    src_lbl_dir = SRC_DATASET / split / "labels"

    dst_img_dir = DST_DATASET / split / "images"
    dst_lbl_dir = DST_DATASET / split / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in src_img_dir.iterdir() if p.suffix in IMG_EXTS]

    print(f"\n🔹 Procesando split: {split}")
    print(f"   Imágenes encontradas: {len(images)}")

    for img_path in images:
        lbl_path = src_lbl_dir / img_path.with_suffix(".txt").name
        tile_image(img_path, lbl_path, dst_img_dir, dst_lbl_dir)


def main():
    print("🚀 Iniciando división de imágenes (2x2)...")
    for split in ["train", "val", "test"]:
        process_split(split)
    print("\n✅ División completada correctamente")


if __name__ == "__main__":
    main()
