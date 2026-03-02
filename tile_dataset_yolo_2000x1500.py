#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2

# =========================
# CONFIGURACIÓN
# =========================

SRC_DIR = Path("/home/root/dataset/DATASET2K")
DST_DIR = Path("/home/root/dataset/DATASET2K_tiled_2000x1500")

TILE_W = 2000
TILE_H = 1500
OVERLAP = 0.10          # 10% overlap
MIN_AREA_RATIO = 0.20   # elimina cajas muy cortadas
IGNORE_EMPTY = True     # no guardar tiles sin objetos

IMG_DIR = SRC_DIR / "images"
LBL_DIR = SRC_DIR / "labels"

OUT_IMG = DST_DIR / "images"
OUT_LBL = DST_DIR / "labels"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# =========================

def yolo_to_xyxy(box, w, h):
    x, y, bw, bh = box
    x1 = (x - bw/2) * w
    y1 = (y - bh/2) * h
    x2 = (x + bw/2) * w
    y2 = (y + bh/2) * h
    return x1, y1, x2, y2

def xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    x = x1 + bw/2
    y = y1 + bh/2
    return x/w, y/h, bw/w, bh/h

# =========================

step_x = int(TILE_W * (1 - OVERLAP))
step_y = int(TILE_H * (1 - OVERLAP))

total_tiles = 0
total_boxes = 0

print("\n🔹 Generando tiles...\n")

for img_path in IMG_DIR.glob("*"):
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    h, w = image.shape[:2]

    label_path = LBL_DIR / (img_path.stem + ".txt")
    boxes = []

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())
                boxes.append((int(cls), *yolo_to_xyxy((x,y,bw,bh), w, h)))

    for y0 in range(0, h, step_y):
        for x0 in range(0, w, step_x):

            x1 = min(x0 + TILE_W, w)
            y1 = min(y0 + TILE_H, h)

            tile = image[y0:y1, x0:x1]

            # saltar bordes incompletos
            if tile.shape[0] < TILE_H or tile.shape[1] < TILE_W:
                continue

            tile_boxes = []

            for cls, bx1, by1, bx2, by2 in boxes:

                ix1 = max(bx1, x0)
                iy1 = max(by1, y0)
                ix2 = min(bx2, x1)
                iy2 = min(by2, y1)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)

                if iw <= 0 or ih <= 0:
                    continue

                inter_area = iw * ih
                box_area = (bx2 - bx1) * (by2 - by1)

                if inter_area / box_area < MIN_AREA_RATIO:
                    continue

                nx1 = ix1 - x0
                ny1 = iy1 - y0
                nx2 = ix2 - x0
                ny2 = iy2 - y0

                yolo_box = xyxy_to_yolo(
                    (nx1, ny1, nx2, ny2),
                    TILE_W,
                    TILE_H
                )

                tile_boxes.append((cls, *yolo_box))

            if IGNORE_EMPTY and not tile_boxes:
                continue

            tile_name = f"{img_path.stem}_{x0}_{y0}"

            cv2.imwrite(str(OUT_IMG / f"{tile_name}.jpg"), tile)

            with open(OUT_LBL / f"{tile_name}.txt", "w") as f:
                for b in tile_boxes:
                    f.write(" ".join(map(str,b)) + "\n")

            total_tiles += 1
            total_boxes += len(tile_boxes)

print("\n✅ Tiling completado")
print("Tiles generados:", total_tiles)
print("Boxes totales:", total_boxes)
