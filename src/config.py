"""
Configuration file for Brand Logo Detection with YOLOv8.
All project-wide settings: paths, hyperparameters, class names.
"""

import os

# ─────────────────────────────────────────
# 1. PROJECT PATHS
# ─────────────────────────────────────────
PROJECT_ROOT  = "/content/brand-logo-detection"
DATA_RAW      = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
IMAGES_DIR    = os.path.join(DATA_PROCESSED, "images")
LABELS_DIR    = os.path.join(DATA_PROCESSED, "labels")
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR   = os.path.join(OUTPUTS_DIR, "figures")
MODELS_DIR    = os.path.join(OUTPUTS_DIR, "models")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
RUNS_DIR      = os.path.join(PROJECT_ROOT, "runs")

# Dataset YAML path (required by YOLO)
DATASET_YAML  = os.path.join(PROJECT_ROOT, "dataset.yaml")

# ─────────────────────────────────────────
# 2. DATASET CONFIG
# ─────────────────────────────────────────
KAGGLE_DATASET = "sushovansaha9/flickr-logos-27-dataset"

# 27 brand logo classes
CLASS_NAMES = [
    "Adidas", "Apple", "BMW", "Citroen", "Coca-Cola",
    "DHL", "FedEx", "Ferrari", "Ford", "Google",
    "Heineken", "HP", "McDonalds", "Mini", "NBC",
    "Nike", "Pepsi", "Porsche", "Puma", "Red Bull",
    "Sprite", "Starbucks", "Intel", "Texaco", "Unicef",
    "Vodafone", "Yahoo"
]
NUM_CLASSES = len(CLASS_NAMES)

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─────────────────────────────────────────
# 3. TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────
# YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=xlarge)
MODEL_VARIANT = "yolov8s.pt"

# Training settings
IMG_SIZE    = 640
BATCH_SIZE  = 16
EPOCHS      = 100
PATIENCE    = 15        # Early stopping patience
OPTIMIZER   = "AdamW"
LR0         = 0.001     # Initial learning rate
LRF         = 0.01      # Final learning rate (fraction of lr0)
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3

# Augmentation (YOLOv8 built-in)
MOSAIC      = 1.0       # Mosaic augmentation probability
MIXUP       = 0.1       # Mixup augmentation probability
DEGREES     = 10.0      # Rotation degrees
TRANSLATE   = 0.1       # Translation fraction
SCALE       = 0.5       # Scale gain
FLIPLR      = 0.5       # Horizontal flip probability
FLIPUD      = 0.0       # Vertical flip probability (off for logos)

# ─────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────
CONF_THRESHOLD = 0.25   # Confidence threshold for predictions
IOU_THRESHOLD  = 0.45   # IoU threshold for NMS

# ─────────────────────────────────────────
# 5. RANDOM SEED
# ─────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────
# 6. DISPLAY CONFIG
# ─────────────────────────────────────────
print(f"✅ Config loaded!")
print(f"   Project   : Brand Logo Detection with YOLOv8")
print(f"   Classes   : {NUM_CLASSES} brands")
print(f"   Model     : {MODEL_VARIANT}")
print(f"   Image Size: {IMG_SIZE}px")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs    : {EPOCHS}")
print(f"   Patience  : {PATIENCE}")
