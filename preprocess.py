# preprocess.py
import shutil, random, logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

UNRIPE_ID = "3 "                     # line that starts with "3 " → Green

def split_dataset(src_img, src_lbl, dst_root, ratios=(0.7, 0.2, 0.1)):
    dst = Path(dst_root)
    if dst.exists():
        log.info(f"Split already exists at {dst} → skipping")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)

    # keep only images that have a label file
    imgs = [p for p in Path(src_img).glob("*.jpg")
            if (Path(src_lbl) / p.with_suffix(".txt").name).exists()]
    random.shuffle(imgs)

    n = len(imgs)
    t1 = int(n * ratios[0])
    t2 = t1 + int(n * ratios[1])

    def copy(split, files):
        for f in files:
            lbl = Path(src_lbl) / f.with_suffix(".txt").name
            shutil.copy(f, dst / split / "images" / f.name)
            shutil.copy(lbl, dst / split / "labels" / lbl.name)

    copy("train", imgs[:t1])
    copy("val",   imgs[t1:t2])
    copy("test",  imgs[t2:])

    log.info(f"Split → train:{t1}  val:{t2-t1}  test:{n-t2}")
    return dst


def write_data_yaml(split_root, classes_txt, yaml_path):
    if Path(yaml_path).exists():
        log.info(f"data.yaml already at {yaml_path} → skipping")
        return

    names = [l.strip() for l in Path(classes_txt).read_text().splitlines() if l.strip()]
    data = {
        "train": str(split_root / "train" / "images"),
        "val":   str(split_root / "val"   / "images"),
        "test":  str(split_root / "test"  / "images"),
        "nc":    len(names),
        "names": names,
    }
    Path(yaml_path).write_text(yaml.dump(data, default_flow_style=False))
    log.info(f"data.yaml written → {yaml_path}")


def oversample_green(train_img_dir, train_lbl_dir):
    img_dir = Path(train_img_dir)
    lbl_dir = Path(train_lbl_dir)

    # ---- clean old copies -------------------------------------------------
    for p in img_dir.glob("*_copy[1-2].jpg"): p.unlink()
    for p in lbl_dir.glob("*_copy[1-2].txt"): p.unlink()

    # ---- find files that contain a Green berry ---------------------------
    green_lbls = []
    for txt in lbl_dir.glob("*.txt"):
        if "_copy" in txt.name: continue
        if any(l.startswith(UNRIPE_ID) for l in txt.read_text().splitlines()):
            green_lbls.append(txt)

    # ---- duplicate twice -------------------------------------------------
    for txt in green_lbls:
        jpg = img_dir / txt.with_suffix(".jpg").name
        if not jpg.exists(): continue
        for i in (1, 2):
            shutil.copy(txt, lbl_dir / f"{txt.stem}_copy{i}.txt")
            shutil.copy(jpg, img_dir / f"{jpg.stem}_copy{i}.jpg")

    log.info(f"Oversampled {len(green_lbls)} Green images → {len(list(img_dir.glob('*.jpg')))} total")


if __name__ == "__main__":
    BASE      = Path("/home/root/dataset/DATASET2K")
    SPLIT     = BASE / "split"
    YML       = SPLIT / "data.yaml"

    split_dataset(BASE/"images", BASE/"labels", SPLIT)
    write_data_yaml(SPLIT, BASE/"classes.txt", YML)
    oversample_green(SPLIT/"train"/"images", SPLIT/"train"/"labels")