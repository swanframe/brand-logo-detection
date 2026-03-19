"""
predict.py — Run inference on new images using the trained YOLOv8 model.
Supports single image, batch inference, and visualization.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR,
    CONF_THRESHOLD, IOU_THRESHOLD, SEED
)

from ultralytics import YOLO
import yaml

random.seed(SEED)
np.random.seed(SEED)


def load_model(model_path=None):
    """Load the trained YOLOv8 model."""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best.pt")
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    return model


# ─────────────────────────────────────────
# 1. SINGLE IMAGE PREDICTION
# ─────────────────────────────────────────
def predict_single(model, image_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, save=True):
    """Run prediction on a single image and return results."""
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    # Extract detections
    detections = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls.cpu().numpy()[0])
        conf_val = float(box.conf.cpu().numpy()[0])
        class_name = model.names[cls_id]
        detections.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf_val,
            "bbox": xyxy.tolist(),  # [x1, y1, x2, y2]
        })

    print(f"\n🔍 {os.path.basename(image_path)}: {len(detections)} logo(s) detected")
    for det in detections:
        print(f"   • {det['class_name']} ({det['confidence']:.2f})")

    return results, detections


# ─────────────────────────────────────────
# 2. VISUALIZE SINGLE PREDICTION
# ─────────────────────────────────────────
def visualize_prediction(image_path, detections, save_path=None):
    """Draw bounding boxes on image and display/save."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img)

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, 27))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        color = colors[cls_id % len(colors)]

        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor=color,
                                 facecolor="none")
        ax.add_patch(rect)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        ax.text(x1, y1 - 8, label, color="white", fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                         facecolor=color, alpha=0.85))

    ax.axis("off")
    ax.set_title(f"Detected {len(detections)} logo(s)", fontsize=16, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   💾 Saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────
# 3. BATCH PREDICTION
# ─────────────────────────────────────────
def predict_batch(model, image_dir, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, max_images=12):
    """Run prediction on all images in a directory."""
    img_files = sorted([f for f in os.listdir(image_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if max_images:
        img_files = random.sample(img_files, min(max_images, len(img_files)))

    all_results = []
    for img_file in img_files:
        img_path = os.path.join(image_dir, img_file)
        results, detections = predict_single(model, img_path, conf=conf, iou=iou, save=False)
        all_results.append((img_path, detections))

    return all_results


# ─────────────────────────────────────────
# 4. VISUALIZE BATCH (GRID)
# ─────────────────────────────────────────
def visualize_batch(batch_results, save_path=None, max_show=12):
    """Visualize batch results in a grid."""
    n = min(len(batch_results), max_show)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 6))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, 27))

    for idx in range(n):
        img_path, detections = batch_results[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(img)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            color = colors[cls_id % len(colors)]

            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor=color,
                                     facecolor="none")
            axes[idx].add_patch(rect)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            axes[idx].text(x1, y1 - 5, label, color="white", fontsize=8,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor=color, alpha=0.85))

        logo_names = [d["class_name"] for d in detections]
        title = ", ".join(logo_names) if logo_names else "No logos"
        axes[idx].set_title(title, fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Batch Inference Results", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n💾 Saved batch visualization to {save_path}")
    plt.show()


# ─────────────────────────────────────────
# 5. INFERENCE SPEED BENCHMARK
# ─────────────────────────────────────────
def benchmark_speed(model, image_dir, n_runs=50):
    """Benchmark inference speed."""
    import time

    img_files = [f for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not img_files:
        print("⚠️  No images found for benchmarking")
        return

    img_path = os.path.join(image_dir, img_files[0])

    # Warmup
    for _ in range(5):
        model.predict(source=img_path, imgsz=640, verbose=False)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(source=img_path, imgsz=640, verbose=False)
        times.append((time.perf_counter() - start) * 1000)  # ms

    avg = np.mean(times)
    std = np.std(times)
    fps = 1000 / avg

    print(f"\n⚡ Inference Speed Benchmark ({n_runs} runs):")
    print(f"   Average : {avg:.1f} ± {std:.1f} ms")
    print(f"   FPS     : {fps:.1f}")
    print(f"   Device  : CUDA" if next(model.model.parameters()).is_cuda else "   Device  : CPU")

    return {"avg_ms": avg, "std_ms": std, "fps": fps}


# ─────────────────────────────────────────
# MAIN — Demo on test set
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BRAND LOGO DETECTION — INFERENCE DEMO")
    print("=" * 60)

    model = load_model()
    class_names = model.names

    # Test image directory
    test_img_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "processed", "images", "test"
    )

    # 1. Batch prediction on test images
    print("\n[1/3] Running batch predictions on test set...")
    batch_results = predict_batch(model, test_img_dir, max_images=12)
    visualize_batch(
        batch_results,
        save_path=os.path.join(FIGURES_DIR, "inference_batch_demo.png")
    )

    # 2. Single image detailed prediction
    print("\n[2/3] Detailed single image prediction...")
    sample_img = batch_results[0][0]
    results, detections = predict_single(model, sample_img)
    visualize_prediction(
        sample_img, detections,
        save_path=os.path.join(FIGURES_DIR, "inference_single_demo.png")
    )

    # 3. Speed benchmark
    print("\n[3/3] Speed benchmark...")
    speed = benchmark_speed(model, test_img_dir)

    print("\n" + "=" * 60)
    print("  ✅ INFERENCE DEMO COMPLETE!")
    print("=" * 60)

    return batch_results, speed


if __name__ == "__main__":
    main()
