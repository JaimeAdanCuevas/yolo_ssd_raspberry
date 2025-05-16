from ultralytics import YOLO
import pandas as pd
from pathlib import Path

# Load model
model = YOLO("/content/RaspberrySet/runs/yolov8_raspberry/weights/best.pt")
test_dir = "/content/RaspberrySet/split/test/images"
output_dir = "/content/drive/MyDrive/RaspberrySet/runs/test_results"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Evaluate on test set
metrics = model.val(data="/content/RaspberrySet/split/data.yaml", split="test")
print(f"Test metrics: mAP@50: {metrics.box.map50:.4f}, mAP@50:95: {metrics.box.map:.4f}")
for i, ap in enumerate(metrics.box.maps):
    print(f"Class {model.names[i]} AP@50: {ap:.4f}")

# Count ripe_berries per image
results = []
for img_path in Path(test_dir).glob("*.JPEG"):
    result = model.predict(img_path, conf=0.5, save=False)
    ripe_count = sum(1 for box in result[0].boxes if result[0].names[int(box.cls.item())] == "ripe_berries")
    results.append({"image": img_path.name, "ripe_berries": ripe_count})
df = pd.DataFrame(results)
df.to_csv(f"{output_dir}/ripe_berries_counts.csv", index=False)
print(f"Saved ripe_berries counts to {output_dir}/ripe_berries_counts.csv")
print(f"Average ripe_berries per image: {df['ripe_berries'].mean():.2f}")
