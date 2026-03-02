# yolo_ssd_raspberry

Experimental Pipeline – YOLOv8 Training & Dataset Variants
1. Dataset Preparation
1.1 Tiling

Script used:

tile_dataset_yolo_2000x1500.py


Although the dataset was already tiled to 2000x1500, this script ensures:

Correct YOLO format

Consistent image-label pairing

Clean directory structure

Final dataset location:

/home/root/dataset/DATASET2K_tiled_2000x1500/DATASET2K_split


Dataset split structure:

DATASET2K_split/
├── train/
├── val/
└── test/

2. Dataset Variants for Experiments

To evaluate class-merge strategies, the following preprocessing scripts were used.

2.1 Remove DarkRed Class

Script:

remove_darkred.py


Purpose:

Removes all darkred (raspberry fruit) samples

Updates labels accordingly

Used to analyze class imbalance and confusion effects

2.2 Merge Green and Boton Classes

Script:

merge_boton_green_after_darkred.py


Purpose:

Merges:

green

boton

Executed after removing darkred

Reduces number of classes for experimental comparison

This produces the final 5-class merged dataset used in training.

3. Training Configuration

Training script:

train_yolov8_minimal_joco.py


Framework:

Ultralytics YOLOv8

Base model:

YOLOv8 yolov8l.pt

3.1 Dataset Configuration
DATASET_PATH = Path("/home/root/dataset/DATASET2K_tiled_2000x1500/DATASET2K_split")
DATA_YAML = DATASET_PATH / "data.yaml"


The data.yaml reflects the merged 5-class configuration.

4. Training Hyperparameters
Model
model = YOLO("yolov8l.pt")


Model size: Large (l)

Image size: 640

Epochs: 100

Batch size: 64

Device: CPU

4.1 Optimizer Settings
optimizer="SGD"
lr0=0.008
lrf=0.01
momentum=0.937
weight_decay=0.0005

4.2 Training Strategy
patience=20
warmup_epochs=3


Early stopping after 20 epochs without improvement

3 warmup epochs

4.3 Data Augmentation
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

Low mosaic (0.15) to reduce over-artificial samples

Copy-paste augmentation (0.05)

Mild geometric transforms

No vertical flip

4.4 System Configuration
workers=48
amp=False
cache=True
plots=True
save_period=25
val=True


Mixed precision disabled

Dataset cached

Validation enabled

Checkpoints saved every 25 epochs

5. Output

Training results are stored in:

runs/yolov8l_final_5classes/


Main outputs:

weights/best.pt

weights/last.pt

Confusion matrix

mAP metrics

Precision / Recall curves

6. Experimental Design for Paper

The reported results correspond to:

Original dataset (baseline)

Dataset without darkred

Dataset with:

darkred removed

green and boton merged

All experiments were trained under identical hyperparameter settings to ensure comparability.
