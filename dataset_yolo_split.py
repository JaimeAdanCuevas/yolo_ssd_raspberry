import os
import shutil
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def split_dataset(image_dir: str, annotation_dir: str, output_dir: str, classes_file: str, train_ratio: float = 0.7, val_ratio: float = 0.2):
    """
    Split RaspberrySet dataset into train/val/test sets and create data.yaml for YOLOv8.

    Args:
        image_dir (str): Path to directory containing images.
        annotation_dir (str): Path to directory containing YOLO annotations.
        output_dir (str): Path to output directory for split datasets.
        classes_file (str): Path to classes.txt file.
        train_ratio (float): Proportion of data for training (default: 0.7).
        val_ratio (float): Proportion of data for validation (default: 0.2).
    """
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    classes_file = Path(classes_file)

    # Validate input directories
    if not image_dir.exists():
        logger.error(f"Image directory {image_dir} does not exist.")
        raise FileNotFoundError(f"Image directory {image_dir} not found.")
    if not annotation_dir.exists():
        logger.error(f"Annotation directory {annotation_dir} does not exist.")
        raise FileNotFoundError(f"Annotation directory {annotation_dir} not found.")
    if not classes_file.exists():
        logger.error(f"Classes file {classes_file} does not exist.")
        raise FileNotFoundError(f"Classes file {classes_file} not found.")

    # Load class names, stripping IDs
    with open(classes_file, "r") as f:
        class_names = [line.strip().split(maxsplit=1)[1] if line.strip().split(maxsplit=1)[0].isdigit() else line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG")])
    label_files = [annotation_dir / f.name.replace(f.suffix, ".txt") for f in image_files]
    missing_annotations = sum(1 for lf in label_files if not lf.exists())
    logger.info(f"Found {len(image_files)} images, {missing_annotations} missing annotations")

    # Split: 70% train (1,427), 20% val (408), 10% test (204)
    train_imgs, temp_imgs, train_labs, temp_labs = train_test_split(
        image_files, label_files, test_size=0.3, random_state=42
    )
    val_imgs, test_imgs, val_labs, test_labs = train_test_split(
        temp_imgs, temp_labs, test_size=0.333, random_state=42
    )

    # Copy files
    splits = {"train": (train_imgs, train_labs), "val": (val_imgs, val_labs), "test": (test_imgs, test_labs)}
    for split, (imgs, labs) in splits.items():
        for img, lab in zip(imgs, labs):
            shutil.copy(img, output_dir / split / "images" / img.name)
            if lab.exists():
                shutil.copy(lab, output_dir / split / "labels" / lab.name)
            else:
                logger.warning(f"Annotation {lab} not found for {img.name}")

    # Verify splits
    split_counts = {}
    for split in splits:
        img_count = len([
            f for f in (output_dir / split / "images").iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        lab_count = len(list((output_dir / split / "labels").glob("*.txt")))
        ripe_berries_count = len([f for f in (output_dir / split / "labels").glob("*.txt") if any(line.startswith("3 ") for line in open(f))])
        split_counts[split] = {"images": img_count, "labels": lab_count, "ripe_berries": ripe_berries_count}
        logger.info(f"{split.capitalize()} split: {img_count} images, {lab_count} labels, {ripe_berries_count} Ripe Berries files")

    # Create data.yaml
    data_yaml = f"""
train: {output_dir}/train/images
val: {output_dir}/val/images
test: {output_dir}/test/images
nc: {len(class_names)}
names: {class_names}
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(data_yaml)
    logger.info(f"Created data.yaml at {output_dir}/data.yaml")

    return split_counts

if __name__ == "__main__":
    image_dir = "/home/root/dataset/DATASET2K/images"
    annotation_dir = "/home/root/dataset/DATASET2K/labels"
    output_dir = "/home/root/dataset/DATASET2K/DATASET2K_split"
    classes_file = "/home/root/dataset/DATASET2K/classes.txt"

    try:
        split_counts = split_dataset(image_dir, annotation_dir, output_dir, classes_file)
        logger.info(f"Split summary: {split_counts}")
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
