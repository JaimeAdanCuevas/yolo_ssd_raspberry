# yolo_ssd_raspberry

# Experimental Pipeline – YOLOv8 Training & Dataset Variants

This repository documents the complete experimental workflow used to generate the results reported in the paper.

---

## 1. Dataset Preparation

### 1.1 Tiling

Script used:

tile_dataset_yolo_2000x1500.py

Although the dataset was already tiled to **2000x1500**, this script ensures:

- Correct YOLO annotation format
- Proper image/label pairing
- Consistent directory structure

Final dataset path:

/home/root/dataset/DATASET2K_tiled_2000x1500/DATASET2K_split


Dataset structure:

DATASET2K_split/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/


---

## 2. Dataset Variants

To evaluate class-merging strategies, multiple dataset variants were generated.

### 2.1 Removing DarkRed Class

Script:

remove_darkred.py


Purpose:

- Removes all samples belonging to the **darkred (raspberry fruit)** class
- Updates annotations
- Used to analyze class imbalance and confusion impact

---

### 2.2 Merging Green and Boton Classes

Script:

merge_boton_green_after_darkred.py


Purpose:

- Merges:
  - `green`
  - `boton`
- Executed after removing `darkred`
- Produces a reduced 5-class dataset

This merged dataset is used for the final reported results.

---

## 3. Training Configuration

Training script:

train_yolov8_minimal_joco.py


Framework:

- Ultralytics YOLOv8

Base model:

n,s,m,l,xl
yolov8<x>.pt


---

### 3.1 Dataset Configuration

DATASET_PATH = Path("/home/root/dataset/DATASET2K_tiled_2000x1500/DATASET2K_split")
DATA_YAML = DATASET_PATH / "data.yaml"

The data.yaml reflects the active class configuration (original or merged).

4. Training Hyperparameters
4.1 General Settings
model_type="l"
epochs=100
batch=64
imgsz=640
device="cpu"

4.2 Optimizer
optimizer="SGD"
lr0=0.008
lrf=0.01
momentum=0.937
weight_decay=0.0005

4.3 Training Strategy
patience=20
warmup_epochs=3
val=True


Early stopping after 20 epochs without improvement

3 warmup epochs

Validation enabled

4.4 Data Augmentation
augment=True
mosaic=0.15
close_mosaic=10
mixup=0.0
copy_paste=0.05
fliplr=0.5
flipud=0.0
degrees=10
translate=0.05
scale=0.15
shear=0.0
perspective=0.0


Key decisions:

Low mosaic to reduce excessive synthetic composition

Copy-paste augmentation enabled

Mild geometric transformations

No vertical flipping

4.5 System Configuration
workers=48
amp=False
cache=True
plots=True
save_period=25
exist_ok=True


Mixed precision disabled

Dataset cached

Plots automatically generated

Checkpoints saved every 25 epochs

5. Output

Training results are saved under:

runs/yolov8l_final_5classes/


Important artifacts:

weights/best.pt

weights/last.pt

Confusion matrix

Precision–Recall curves

mAP@0.5

mAP@0.5:0.95

6. Experimental Design Summary

Experiments performed:

Baseline dataset (original classes)

Dataset without darkred

Dataset with:

darkred removed

green and boton merged

All experiments were trained with identical hyperparameters to ensure fair comparison.
