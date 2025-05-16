import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

# Configure logging for robustness
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RaspberryDatasetLoader:
    def __init__(self, image_dir: str, annotation_dir: str, classes_file: str = None):
        """
        Initialize dataset loader for RaspberrySet dataset (YOLO format).

        Args:
            image_dir (str): Path to image directory.
            annotation_dir (str): Path to YOLO annotation directory.
            classes_file (str, optional): Path to classes.txt file with class names.
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.classes_file = Path(classes_file) if classes_file else None
        self.classes = self._load_classes()  # Load class names
        self.validate_paths()

    def validate_paths(self) -> None:
        """Validate that image and annotation directories exist."""
        if not self.image_dir.exists():
            logger.error(f"Image directory {self.image_dir} does not exist.")
            raise FileNotFoundError(f"Image directory {self.image_dir} not found.")
        if not self.annotation_dir.exists():
            logger.error(f"Annotation directory {self.annotation_dir} does not exist.")
            raise FileNotFoundError(f"Annotation directory {self.annotation_dir} not found.")
        if self.classes_file and not self.classes_file.exists():
            logger.warning(f"Classes file {self.classes_file} not found. Using default class IDs.")

    def _load_classes(self) -> List[str]:
        """Load class names from classes.txt or use defaults."""
        default_classes = ["buds", "damaged_buds", "flowers", "unripe_berries", "ripe_berries"]
        if self.classes_file and self.classes_file.exists():
            try:
                with open(self.classes_file, "r") as f:
                    classes = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(classes)} classes from {self.classes_file}")
                return classes
            except Exception as e:
                logger.error(f"Failed to load classes from {self.classes_file}: {e}")
        logger.info("Using default classes: %s", default_classes)
        return default_classes

    def parse_yolo_annotation(self, txt_file: Path, image_shape: Tuple[int, int]) -> List[Dict]:
        """Parse YOLO annotation file (normalized coordinates)."""
        try:
            boxes = []
            img_height, img_width = image_shape
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.warning(f"Invalid YOLO annotation in {txt_file}: {line}")
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # Validate class_id
                    if not (0 <= class_id < len(self.classes)):
                        logger.warning(f"Invalid class_id {class_id} in {txt_file}")
                        continue
                    # Convert normalized to pixel coordinates
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    xmin = int(x_center - width / 2)
                    ymin = int(y_center - height / 2)
                    xmax = int(x_center + width / 2)
                    ymax = int(y_center + height / 2)
                    # Ensure coordinates are within image bounds
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

        # Load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Image loading failed.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None, []

        # Load annotations
        annotation_path = self.annotation_dir / image_name.replace(image_path.suffix, ".txt")
        boxes = []
        if annotation_path.exists():
            boxes = self.parse_yolo_annotation(annotation_path, image.shape[:2])
        else:
            logger.warning(f"YOLO annotation {annotation_path} not found.")

        return image, boxes

    def validate_dataset(self, max_samples: int = 100) -> Dict[str, int]:
        """Validate dataset by checking images and annotations."""
        validation_results = {"images_found": 0, "annotations_found": 0, "errors": 0}
        image_files = [f for f in self.image_dir.iterdir() if f.suffix.lower() in (".jpeg", ".png", ".jpg")]

        for i, image_file in enumerate(image_files[:max_samples]):
            image_name = image_file.name
            image, boxes = self.load_sample(image_name)
            if image is not None:
                validation_results["images_found"] += 1
                if boxes:
                    validation_results["annotations_found"] += 1
            else:
                validation_results["errors"] += 1
                logger.warning(f"Validation error for image {image_name}")

        logger.info("Dataset validation results: %s", validation_results)
        return validation_results

    def visualize_sample(self, image_name: str) -> None:
        """Visualize an image with annotated bounding boxes."""
        image, boxes = self.load_sample(image_name)
        if image is None:
            logger.error("Cannot visualize: Image loading failed.")
            return

        # Draw bounding boxes
        for box in boxes:
            cv2.rectangle(
                image,
                (box["xmin"], box["ymin"]),
                (box["xmax"], box["ymax"]),
                (0, 255, 0),  # Green color
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

        # Display image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"RaspberrySet Sample: {image_name}")
        plt.show()
        logger.info(f"Visualized image {image_name} with {len(boxes)} annotations.")

# Usage example
if __name__ == "__main__":
    # Update paths based on your RaspberrySet dataset
    samples=50
    image_dir = "/content/RaspberrySet/images"
    annotation_dir = "/content/RaspberrySet/annotations"
    classes_file = "/content/RaspberrySet/classes.txt"  # Optional
    print("\nLooking in:", image_dir ,"\n")


    try:
        loader = RaspberryDatasetLoader(image_dir, annotation_dir, classes_file)
        # Validate dataset
        validation_results = loader.validate_dataset(max_samples=samples)
        logger.info("Validation summary: %s", validation_results)

        # Visualize a sample image
        sample_image = next(
            (f for f in os.listdir(image_dir) if f.lower().endswith((".jpeg", ".png", ".jpg"))), None
        )
        if sample_image:
            loader.visualize_sample(sample_image)
        else:
            logger.error("No JPEG/PNG images found in image directory.")
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
