"""
train.py — Fine-tune YOLOv8 on the Brand Logo Detection dataset.
Uses Ultralytics YOLO API with custom hyperparameters from config.
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    PROJECT_ROOT, DATASET_YAML, RUNS_DIR, MODELS_DIR,
    MODEL_VARIANT, IMG_SIZE, BATCH_SIZE, EPOCHS, PATIENCE,
    OPTIMIZER, LR0, LRF, WEIGHT_DECAY, WARMUP_EPOCHS,
    MOSAIC, MIXUP, DEGREES, TRANSLATE, SCALE,
    FLIPLR, FLIPUD, SEED
)

from ultralytics import YOLO


def main():
    print("=" * 60)
    print("  BRAND LOGO DETECTION — YOLOv8 TRAINING")
    print("=" * 60)

    # ── Load pretrained YOLOv8 ────────────────────────────
    print(f"\n[1/3] Loading pretrained model: {MODEL_VARIANT}")
    model = YOLO(MODEL_VARIANT)

    # ── Train ─────────────────────────────────────────────
    print(f"\n[2/3] Starting training...")
    print(f"   Dataset  : {DATASET_YAML}")
    print(f"   Epochs   : {EPOCHS}")
    print(f"   Img Size : {IMG_SIZE}")
    print(f"   Batch    : {BATCH_SIZE}")
    print(f"   Optimizer: {OPTIMIZER}")
    print(f"   LR0      : {LR0}")
    print(f"   Patience : {PATIENCE}")

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        seed=SEED,

        # Augmentation
        mosaic=MOSAIC,
        mixup=MIXUP,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        fliplr=FLIPLR,
        flipud=FLIPUD,

        # Output
        project=RUNS_DIR,
        name="train",
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # ── Copy best weights ─────────────────────────────────
    print(f"\n[3/3] Saving best model weights...")
    best_pt = os.path.join(RUNS_DIR, "train", "weights", "best.pt")
    last_pt = os.path.join(RUNS_DIR, "train", "weights", "last.pt")

    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(best_pt):
        shutil.copy2(best_pt, os.path.join(MODELS_DIR, "best.pt"))
        print(f"   ✅ best.pt saved to {MODELS_DIR}")
    if os.path.exists(last_pt):
        shutil.copy2(last_pt, os.path.join(MODELS_DIR, "last.pt"))
        print(f"   ✅ last.pt saved to {MODELS_DIR}")

    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
