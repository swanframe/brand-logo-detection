"""
evaluate.py — Evaluate the trained YOLOv8 model on the test set.
Generates per-class metrics, confusion matrix, and error analysis.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    PROJECT_ROOT, DATA_PROCESSED, IMAGES_DIR, LABELS_DIR,
    DATASET_YAML, MODELS_DIR, FIGURES_DIR, PREDICTIONS_DIR,
    CONF_THRESHOLD, IOU_THRESHOLD, SEED
)

from ultralytics import YOLO
import yaml

random.seed(SEED)
np.random.seed(SEED)


def load_class_names():
    with open(DATASET_YAML, "r") as fp:
        cfg = yaml.safe_load(fp)
    return cfg["names"]


# ─────────────────────────────────────────
# 1. RUN TEST SET EVALUATION
# ─────────────────────────────────────────
def evaluate_test_set(model):
    """Run official YOLO validation on the test set."""
    print("\n[1/4] Running evaluation on TEST set...")

    results = model.val(
        data=DATASET_YAML,
        split="test",
        imgsz=640,
        batch=16,
        conf=0.001,        # Low conf for mAP calculation
        iou=0.6,
        plots=True,
        save_json=False,
        project=os.path.join(PROJECT_ROOT, "runs"),
        name="test_eval",
        exist_ok=True,
    )

    return results


# ─────────────────────────────────────────
# 2. PRINT PER-CLASS METRICS
# ─────────────────────────────────────────
def print_per_class_metrics(results, class_names):
    """Print detailed per-class metrics table."""
    print("\n[2/4] Per-Class Test Metrics:")
    print("=" * 85)
    print(f"  {'Class':<15} {'Images':>7} {'Instances':>10} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("=" * 85)

    # Access per-class metrics
    box = results.box
    ap50 = box.ap50
    ap = box.ap
    p = box.p
    r = box.r

    metrics_data = []

    for i, name in enumerate(class_names):
        if i < len(ap50):
            row = {
                "class": name,
                "precision": float(p[i]) if i < len(p) else 0,
                "recall": float(r[i]) if i < len(r) else 0,
                "mAP50": float(ap50[i]),
                "mAP50_95": float(ap[i]) if i < len(ap) else 0,
            }
            metrics_data.append(row)
            print(f"  {name:<15} {'':>7} {'':>10} {row['precision']:>8.3f} {row['recall']:>8.3f} {row['mAP50']:>8.3f} {row['mAP50_95']:>10.3f}")

    # Overall
    print("-" * 85)
    print(f"  {'OVERALL':<15} {'':>7} {'':>10} {float(box.mp):>8.3f} {float(box.mr):>8.3f} {float(box.map50):>8.3f} {float(box.map):>10.3f}")
    print("=" * 85)

    return metrics_data


# ─────────────────────────────────────────
# 3. PLOT PER-CLASS AP BAR CHART
# ─────────────────────────────────────────
def plot_per_class_ap(metrics_data):
    """Horizontal bar chart of per-class AP@50."""
    print("\n[3/4] Generating per-class AP chart...")

    # Sort by mAP50
    sorted_data = sorted(metrics_data, key=lambda x: x["mAP50"], reverse=True)
    names = [d["class"] for d in sorted_data]
    ap50_vals = [d["mAP50"] for d in sorted_data]

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = []
    for v in ap50_vals:
        if v >= 0.8:
            colors.append("#2ecc71")   # green
        elif v >= 0.6:
            colors.append("#f39c12")   # orange
        else:
            colors.append("#e74c3c")   # red

    bars = ax.barh(names, ap50_vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("AP@50", fontsize=12)
    ax.set_title("Per-Class AP@50 on Test Set", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 1.05)

    # Add value labels
    for bar, val in zip(bars, ap50_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Good (≥0.80)"),
        Patch(facecolor="#f39c12", label="Fair (0.60-0.80)"),
        Patch(facecolor="#e74c3c", label="Poor (<0.60)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Mean line
    mean_ap = np.mean(ap50_vals)
    ax.axvline(x=mean_ap, color="blue", linestyle="--", alpha=0.7, label=f"Mean: {mean_ap:.3f}")
    ax.text(mean_ap + 0.01, len(names) - 0.5, f"Mean: {mean_ap:.3f}", color="blue", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "test_per_class_ap50.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved test_per_class_ap50.png")


# ─────────────────────────────────────────
# 4. ERROR ANALYSIS: WORST PREDICTIONS
# ─────────────────────────────────────────
def error_analysis(model, class_names):
    """Run predictions on test set and visualize failures."""
    print("\n[4/4] Running error analysis on test set...")

    test_img_dir = os.path.join(IMAGES_DIR, "test")
    test_lbl_dir = os.path.join(LABELS_DIR, "test")

    all_imgs = sorted([f for f in os.listdir(test_img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # Run predictions
    pred_results = model.predict(
        source=test_img_dir,
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=False,
        verbose=False,
    )

    # Compare predictions vs ground truth
    analysis = []
    for result in pred_results:
        img_name = os.path.basename(result.path)
        base = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(test_lbl_dir, base + ".txt")

        # Count GT boxes
        gt_count = 0
        gt_classes = set()
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_count += 1
                        gt_classes.add(int(parts[0]))

        # Count predictions
        pred_count = len(result.boxes)
        pred_classes = set(result.boxes.cls.cpu().numpy().astype(int).tolist()) if pred_count > 0 else set()

        # Missed = GT classes not in predictions
        missed = gt_classes - pred_classes
        # False positives = predicted classes not in GT
        false_pos = pred_classes - gt_classes

        analysis.append({
            "image": img_name,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "gt_classes": gt_classes,
            "pred_classes": pred_classes,
            "missed": missed,
            "false_pos": false_pos,
            "n_missed": len(missed),
            "n_false_pos": len(false_pos),
        })

    # Show worst cases (most missed)
    worst = sorted(analysis, key=lambda x: x["n_missed"] + x["n_false_pos"], reverse=True)

    print("\n📋 Error Analysis Summary:")
    perfect = sum(1 for a in analysis if a["n_missed"] == 0 and a["n_false_pos"] == 0)
    some_miss = sum(1 for a in analysis if a["n_missed"] > 0)
    some_fp = sum(1 for a in analysis if a["n_false_pos"] > 0)
    print(f"   Total test images : {len(analysis)}")
    print(f"   Perfect matches   : {perfect} ({100*perfect/len(analysis):.1f}%)")
    print(f"   Images with misses: {some_miss} ({100*some_miss/len(analysis):.1f}%)")
    print(f"   Images with FP    : {some_fp} ({100*some_fp/len(analysis):.1f}%)")

    # Visualize top 6 worst predictions
    n_show = min(6, len(worst))
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    random.seed(SEED)
    colors_map = {i: [random.random() for _ in range(3)] for i in range(len(class_names))}

    for idx in range(n_show):
        entry = worst[idx]
        img_path = os.path.join(test_img_dir, entry["image"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        axes[idx].imshow(img)

        # Draw GT boxes (green)
        lbl_path = os.path.join(test_lbl_dir, os.path.splitext(entry["image"])[0] + ".txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        box_w = int(bw * w)
                        box_h = int(bh * h)
                        rect = patches.Rectangle((x1, y1), box_w, box_h,
                                                 linewidth=2, edgecolor="lime",
                                                 facecolor="none", linestyle="--")
                        axes[idx].add_patch(rect)
                        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
                        axes[idx].text(x1, y1 - 8, f"GT: {name}", color="lime",
                                      fontsize=8, fontweight="bold",
                                      bbox=dict(boxstyle="round,pad=0.2",
                                               facecolor="black", alpha=0.7))

        # Draw predicted boxes (red)
        result = model.predict(source=img_path, imgsz=640, conf=CONF_THRESHOLD,
                              iou=IOU_THRESHOLD, verbose=False)[0]
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = xyxy
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor="red", facecolor="none")
            axes[idx].add_patch(rect)
            name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
            axes[idx].text(x2, y1, f"{name} {conf:.2f}", color="red",
                          fontsize=8, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor="white", alpha=0.7))

        missed_names = [class_names[c] for c in entry["missed"] if c < len(class_names)]
        fp_names = [class_names[c] for c in entry["false_pos"] if c < len(class_names)]
        title = f"{entry['image']}\n"
        if missed_names:
            title += f"MISSED: {', '.join(missed_names)}  "
        if fp_names:
            title += f"FP: {', '.join(fp_names)}"
        if not missed_names and not fp_names:
            title += "PERFECT"
        axes[idx].set_title(title, fontsize=9, color="red" if missed_names or fp_names else "green")
        axes[idx].axis("off")

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Error Analysis — Worst Predictions on Test Set\n(Green dashed = GT, Red solid = Prediction)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "test_error_analysis.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved test_error_analysis.png")

    return analysis


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BRAND LOGO DETECTION — TEST SET EVALUATION")
    print("=" * 60)

    class_names = load_class_names()

    # Load best model
    best_pt = os.path.join(MODELS_DIR, "best.pt")
    print(f"\n📦 Loading model: {best_pt}")
    model = YOLO(best_pt)

    # 1. Official evaluation
    results = evaluate_test_set(model)

    # 2. Per-class metrics
    metrics_data = print_per_class_metrics(results, class_names)

    # 3. Per-class AP chart
    plot_per_class_ap(metrics_data)

    # 4. Error analysis
    analysis = error_analysis(model, class_names)

    # Save metrics as JSON
    metrics_path = os.path.join(PREDICTIONS_DIR, "test_metrics.json")
    summary = {
        "overall": {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        },
        "per_class": metrics_data,
    }
    with open(metrics_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n📄 Metrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("  ✅ EVALUATION COMPLETE!")
    print("=" * 60)

    return results, metrics_data, analysis


if __name__ == "__main__":
    main()
