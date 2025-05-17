# Import libraries
from ultralytics import YOLO
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from IPython.display import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_yolov8(data_yaml: str, output_dir: str, epochs: int = 100, batch_size: int = 64, img_size: int = 640):
    """
    Train YOLOv8 on the RaspberrySet dataset.

    Args:
        data_yaml (str): Path to data.yaml file.
        output_dir (str): Directory to save model weights and results.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        img_size (int): Input image size.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8n model
    try:
        model = YOLO("yolov8s.pt")  # Pre-trained weights
        logger.info("Loaded YOLOv8n model")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8n model: {e}")
        raise

    # Train model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device='cpu',  # GPU
            project=str(output_dir),
            name="yolov8_raspberry",
            exist_ok=True,
            optimizer="SGD",
            lr0=0.01,
            patience=20,
            verbose=True
        )
        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save best model
    best_model_path = output_dir / "yolov8_raspberry" / "weights" / "best.pt"
    logger.info(f"Best model saved at {best_model_path}")

    # Evaluate on validation set
    try:
        metrics = model.val()
        logger.info(f"Validation metrics: mAP@50: {metrics.box.map50:.4f}, mAP@50:95: {metrics.box.map:.4f}")
        for i, ap in enumerate(metrics.box.maps):
            logger.info(f"Class {model.names[i]} AP@50: {ap:.4f}")
    except Exception as e:
        logger.error(f"Validation failed: {e}")

    return model

def visualize_prediction(model, image_path: str, output_dir: str):
    """
    Visualize a sample prediction, focusing on ripe_berries.

    Args:
        model: Trained YOLOv8 model.
        image_path (str): Path to a test image.
        output_dir (str): Directory to save visualization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = model.predict(image_path, conf=0.5, save=False)
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ripe_berries_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())  # Convert tensor to int
                if result.names[cls_id] == "ripe_berries":
                    ripe_berries_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Convert tensor to numpy
                conf = box.conf.item()  # Convert tensor to float
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{result.names[cls_id]} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Prediction: {ripe_berries_count} ripe_berries")
        output_path = output_dir / f"prediction_{Path(image_path).name}"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved prediction visualization at {output_path} with {ripe_berries_count} ripe_berries")

        return output_path
    except Exception as e:
        logger.error(f"Prediction visualization failed: {e}")
        return None

# Define paths
data_yaml = "/content/RaspberrySet/split/data.yaml"
output_dir = "/content/RaspberrySet/runs"
test_image = "/content/RaspberrySet/split/test/images/IMG_3552.JPEG"

try:
    # Train model
    model = train_yolov8(data_yaml, output_dir, epochs=50, batch_size=16, img_size=640)

    # Visualize a test image
    output_path = visualize_prediction(model, test_image, output_dir)

    # Display visualization
    if output_path:
        display(Image(str(output_path)))
except Exception as e:
    logger.error(f"Error during training or prediction: {e}")
