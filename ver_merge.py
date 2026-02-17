from pathlib import Path

DATASET_DIR = Path("/home/root/dataset/DATASET2K/DATASET2K_split")
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    counts = {}
    for f in (DATASET_DIR / split / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
    print(split, counts)
