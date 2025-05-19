import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RaspberryDatasetLoader:
    def __init__(self, image_dir: str, annotation_dir: str, classes_file: str):
        """
        Initialize dataset loader for RaspberrySet dataset (YOLO format).

        Args:
            image_dir (str): Path to image directory.
            annotation_dir (str): Path to YOLO annotation directory.
            classes_file (str): Path to classes.txt file with class names.
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.classes_file = Path(classes_file)
        self.classes = self._load_classes()
        self.validate_paths()

    def validate_paths(self) -> None:
        """Validate that image, annotation directories, and classes file exist."""
        if not self.image_dir.exists():
            logger.error(f"Image directory {self.image_dir} does not exist.")
            raise FileNotFoundError(f"Image directory {self.image_dir} not found.")
        if not self.annotation_dir.exists():
            logger.error(f"Annotation directory {self.annotation_dir} does not exist.")
            raise FileNotFoundError(f"Annotation directory {self.annotation_dir} not found.")
        if not self.classes_file.exists():
            logger.error(f"Classes file {self.classes_file} does not exist.")
            raise FileNotFoundError(f"Classes file {self.classes_file} not found.")

    def _load_classes(self) -> List[str]:
        """Load class names from classes.txt."""
        try:
            with open(self.classes_file, "r") as f:
                classes = [line.strip() for line in f if line.strip()]
            if not classes:
                raise ValueError("Classes file is empty.")
            logger.info(f"Loaded {len(classes)} classes from {self.classes_file}: {classes}")
            return classes
        except Exception as e:
            logger.error(f"Failed to load classes from {self.classes_file}: {e}")
            raise

    def parse_yolo_annotation(self, txt_file: Path, image_shape: Tuple[int, int]) -> List[Dict]:
        """Parse YOLO annotation file (normalized coordinates)."""
        try:
            boxes = []
            img_height, img_width = image_shape
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.error(f"Invalid YOLO annotation in {txt_file}: {line}")
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    if not (0 <= class_id < len(self.classes)):
                        logger.error(f"Invalid class_id {class_id} in {txt_file} (valid: 0-{len(self.classes)-1})")
                        raise ValueError(f"Invalid class_id {class_id} in {txt_file}")
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    xmin = int(x_center - width / 2)
                    ymin = int(y_center - height / 2)
                    xmax = int(x_center + width / 2)
                    ymax = int(x_center + height / 2)
                    boxes.append({
                        "name": self.classes[int(class_id)],
                        "xmin": max(0, xmin),
                        "ymin": max(0, ymin),
                        "xmax": min(img_width, xmax),
                        "ymax": min(img_height, ymax)
                    })
            return boxes
        except Exception as e:
            logger.error(f"Failed to parse YOLO file {txt_file}: {e}")
            return []

    def load_sample(self, image_name: str) -> Tuple[np.ndarray, List[Dict]]:
        """Load an image and its annotations."""
        image_path = self.image_dir / image_name
        if not image_path.exists():
            logger.error(f"Image {image_path} not found.")
            return None, []

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Image loading failed.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None, []

        annotation_path = self.annotation_dir / image_name.replace(image_path.suffix, ".txt")
        boxes = []
        if annotation_path.exists():
            boxes = self.parse_yolo_annotation(annotation_path, image.shape[:2])
        else:
            logger.warning(f"YOLO annotation {annotation_path} not found.")

        return image, boxes

    def validate_dataset(self, max_samples: int = 100) -> Dict[str, int]:
        """Validate dataset by checking images and annotations."""
        validation_results = {"images_found": 0, "annotations_found": 0, "missing_annotations": 0, "errors": 0}
        image_files = [f for f in self.image_dir.iterdir() if f.suffix.lower() == ".jpeg"]

        for image_file in image_files[:max_samples]:
            image_name = image_file.name
            image, boxes = self.load_sample(image_name)
            if image is not None:
                validation_results["images_found"] += 1
                if boxes:
                    validation_results["annotations_found"] += 1
                else:
                    validation_results["missing_annotations"] += 1
            else:
                validation_results["errors"] += 1
                logger.warning(f"Validation error for image {image_name}")

        logger.info("Dataset validation results: %s", validation_results)
        return validation_results

    def visualize_sample(self, image_name: str = None, prefer_ripe_berries: bool = False) -> None:
        """Visualize an image with annotated bounding boxes."""
        if prefer_ripe_berries:
            for txt_file in self.annotation_dir.glob("*.txt"):
                with open(txt_file, "r") as f:
                    if any(line.startswith("3 ") for line in f):
                        image_name = txt_file.name.replace(".txt", ".JPEG")
                        break
            if not image_name:
                logger.warning("No image with Ripe Berries found for visualization.")

        if not image_name:
            image_name = next(
                (f.name for f in self.image_dir.iterdir() if f.suffix.lower() == ".jpeg"), None
            )

        if not image_name:
            logger.error("No JPEG images found in image directory.")
            return

        image, boxes = self.load_sample(image_name)
        if image is None:
            logger.error("Cannot visualize: Image loading failed.")
            return

        for box in boxes:
            cv2.rectangle(
                image,
                (box["xmin"], box["ymin"]),
                (box["xmax"], box["ymax"]),
                (0, 255, 0),
                2
            )
            cv2.putText(
                image,
                box["name"],
                (box["xmin"], box["ymin"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"RaspberrySet Sample: {image_name}")
        plt.show()
        logger.info(f"Visualized image {image_name} with {len(boxes)} annotations.")

if __name__ == "__main__":
    image_dir = "/content/RaspberrySet/images"
    annotation_dir = "/content/RaspberrySet/annotations"
    classes_file = "/content/RaspberrySet/classes.txt"
    print("\nLooking in:", image_dir, "\n")

    try:
        loader = RaspberryDatasetLoader(image_dir, annotation_dir, classes_file)
        validation_results = loader.validate_dataset(max_samples=50)
        logger.info("Validation summary: %s", validation_results)
        loader.visualize_sample(prefer_ripe_berries=True)
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
