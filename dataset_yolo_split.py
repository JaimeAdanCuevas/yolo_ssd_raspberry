import os
import shutil
from pathlib import Path
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def split_dataset(image_dir: str, annotation_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2):
    """
    Split RaspberrySet dataset into train/val/test sets and create data.yaml for YOLOv8.

    Args:
        image_dir (str): Path to directory containing images.
        annotation_dir (str): Path to directory containing YOLO annotations.
        output_dir (str): Path to output directory for split datasets.
        train_ratio (float): Proportion of data for training (default: 0.7).
        val_ratio (float): Proportion of data for validation (default: 0.2).
    """
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    # Validate input directories
    if not image_dir.exists():
        logger.error(f"Image directory {image_dir} does not exist.")
        raise FileNotFoundError(f"Image directory {image_dir} not found.")
    if not annotation_dir.exists():
        logger.error(f"Annotation directory {annotation_dir} does not exist.")
        raise FileNotFoundError(f"Annotation directory {annotation_dir} not found.")

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    random.shuffle(image_files)

    # Calculate split sizes
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Split files
    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:]
    }

    # Copy files
    for split, files in splits.items():
        for img in files:
            # Copy image
            shutil.copy(img, output_dir / split / "images" / img.name)
            # Copy annotation
            ann = annotation_dir / img.name.replace(img.suffix, ".txt")
            if ann.exists():
                shutil.copy(ann, output_dir / split / "labels" / ann.name)
            else:
                logger.warning(f"Annotation {ann} not found for {img.name}")

    logger.info(f"Split dataset: {n_train} train, {n_val} val, {n_test} test")

    # Create data.yaml
    data_yaml = f"""
train: {output_dir}/train/images
val: {output_dir}/val/images
test: {output_dir}/test/images
nc: 5
names: ['buds', 'damaged_buds', 'flowers', 'unripe_berries', 'ripe_berries']
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(data_yaml)
    logger.info(f"Created data.yaml at {output_dir}/data.yaml")

if __name__ == "__main__":
    # Your dataset paths
    image_dir = "/content/RaspberrySet/images"
    annotation_dir = "/content/RaspberrySet/annotations"
    output_dir = "/content/RaspberrySet/split"

    try:
        split_dataset(image_dir, annotation_dir, output_dir)
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
