"""
eda.py — Exploratory Data Analysis for Brand Logo Detection.
Generates visualizations: class distribution, bbox size analysis,
aspect ratios, split comparison, and sample annotations.
"""

import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    DATA_PROCESSED, IMAGES_DIR, LABELS_DIR,
    FIGURES_DIR, DATASET_YAML, SEED
)

random.seed(SEED)
np.random.seed(SEED)

# Load class names
import yaml
with open(DATASET_YAML, "r") as fp:
    ds_cfg = yaml.safe_load(fp)
CLASS_NAMES = ds_cfg["names"]
NUM_CLASSES = ds_cfg["nc"]


# ─────────────────────────────────────────
# UTILITY: Parse all labels
# ─────────────────────────────────────────
def load_all_labels():
    """Load all YOLO labels across splits. Returns dict of split -> list of (file, boxes)."""
    data = {}
    for split in ["train", "val", "test"]:
        lbl_dir = os.path.join(LABELS_DIR, split)
        img_dir = os.path.join(IMAGES_DIR, split)
        entries = []
        for lbl_file in sorted(os.listdir(lbl_dir)):
            if not lbl_file.endswith(".txt"):
                continue
            lbl_path = os.path.join(lbl_dir, lbl_file)
            base = os.path.splitext(lbl_file)[0]

            # Find image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(img_dir, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            boxes = []
            with open(lbl_path, "r") as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        boxes.append((cls_id, xc, yc, w, h))

            entries.append({"label_file": lbl_file, "img_path": img_path, "boxes": boxes})
        data[split] = entries
    return data


# ─────────────────────────────────────────
# 1. CLASS DISTRIBUTION (per split)
# ─────────────────────────────────────────
def plot_class_distribution(data):
    """Bar chart of class frequency across train/val/test."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for idx, split in enumerate(["train", "val", "test"]):
        counts = defaultdict(int)
        for entry in data[split]:
            for (cls_id, *_) in entry["boxes"]:
                counts[cls_id] += 1

        classes = list(range(NUM_CLASSES))
        freqs = [counts.get(c, 0) for c in classes]
        names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls_{c}" for c in classes]

        colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
        bars = axes[idx].barh(names, freqs, color=colors)
        axes[idx].set_xlabel("Number of Instances")
        axes[idx].set_title(f"{split.upper()} ({sum(freqs)} boxes)", fontsize=14, fontweight="bold")
        axes[idx].invert_yaxis()

        for bar, freq in zip(bars, freqs):
            if freq > 0:
                axes[idx].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                              str(freq), va="center", fontsize=8)

    plt.suptitle("Class Distribution per Split", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved class_distribution.png")


# ─────────────────────────────────────────
# 2. OVERALL CLASS BALANCE
# ─────────────────────────────────────────
def plot_overall_balance(data):
    """Single bar chart with total instances per class."""
    counts = defaultdict(int)
    for split in ["train", "val", "test"]:
        for entry in data[split]:
            for (cls_id, *_) in entry["boxes"]:
                counts[cls_id] += 1

    classes = list(range(NUM_CLASSES))
    freqs = [counts.get(c, 0) for c in classes]
    names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls_{c}" for c in classes]

    # Sort by frequency
    sorted_pairs = sorted(zip(names, freqs), key=lambda x: x[1], reverse=True)
    names_sorted, freqs_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names_sorted)))
    bars = ax.barh(list(names_sorted), list(freqs_sorted), color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Total Instances", fontsize=12)
    ax.set_title("Overall Class Balance (All Splits)", fontsize=16, fontweight="bold")

    for bar, freq in zip(bars, freqs_sorted):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(freq), va="center", fontsize=9)

    mean_freq = np.mean(freqs_sorted)
    ax.axvline(x=mean_freq, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_freq:.1f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "overall_class_balance.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved overall_class_balance.png")


# ─────────────────────────────────────────
# 3. BOUNDING BOX SIZE DISTRIBUTION
# ─────────────────────────────────────────
def plot_bbox_sizes(data):
    """Scatter plot of bbox width vs height (normalized) + histogram."""
    widths, heights, areas = [], [], []

    for split in ["train", "val", "test"]:
        for entry in data[split]:
            for (cls_id, xc, yc, w, h) in entry["boxes"]:
                widths.append(w)
                heights.append(h)
                areas.append(w * h)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Scatter: width vs height
    axes[0].scatter(widths, heights, alpha=0.3, s=10, c="steelblue")
    axes[0].set_xlabel("Normalized Width")
    axes[0].set_ylabel("Normalized Height")
    axes[0].set_title("Bbox Width vs Height")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Histogram: area
    axes[1].hist(areas, bins=50, color="coral", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Normalized Area (w × h)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Bbox Area Distribution")
    axes[1].axvline(x=np.median(areas), color="red", linestyle="--",
                    label=f"Median: {np.median(areas):.3f}")
    axes[1].legend()

    # Box categories: small/medium/large
    small = sum(1 for a in areas if a < 0.04)
    medium = sum(1 for a in areas if 0.04 <= a < 0.25)
    large = sum(1 for a in areas if a >= 0.25)
    labels = ["Small\n(< 4%)", "Medium\n(4-25%)", "Large\n(> 25%)"]
    sizes = [small, medium, large]
    colors_pie = ["#ff9999", "#66b3ff", "#99ff99"]
    axes[2].pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 11})
    axes[2].set_title("Bbox Size Categories")

    plt.suptitle("Bounding Box Size Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "bbox_sizes.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved bbox_sizes.png")


# ─────────────────────────────────────────
# 4. ASPECT RATIO DISTRIBUTION
# ─────────────────────────────────────────
def plot_aspect_ratios(data):
    """Histogram of bbox aspect ratios."""
    ratios = []
    for split in ["train", "val", "test"]:
        for entry in data[split]:
            for (cls_id, xc, yc, w, h) in entry["boxes"]:
                if h > 0:
                    ratios.append(w / h)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ratios, bins=50, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Aspect Ratio (width / height)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Bounding Box Aspect Ratios", fontsize=16, fontweight="bold")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Square (1:1)")
    ax.axvline(x=np.median(ratios), color="green", linestyle="--", alpha=0.7,
              label=f"Median: {np.median(ratios):.2f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "aspect_ratios.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved aspect_ratios.png")


# ─────────────────────────────────────────
# 5. OBJECTS PER IMAGE
# ─────────────────────────────────────────
def plot_objects_per_image(data):
    """Histogram of how many objects per image."""
    counts = []
    for split in ["train", "val", "test"]:
        for entry in data[split]:
            counts.append(len(entry["boxes"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    max_count = max(counts)
    bins = range(0, max_count + 2)
    ax.hist(counts, bins=bins, color="teal", edgecolor="white", alpha=0.8, align="left")
    ax.set_xlabel("Number of Objects per Image", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Objects per Image Distribution", fontsize=16, fontweight="bold")
    ax.set_xticks(range(0, max_count + 1))

    mean_val = np.mean(counts)
    ax.axvline(x=mean_val, color="red", linestyle="--", alpha=0.7,
              label=f"Mean: {mean_val:.2f}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "objects_per_image.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved objects_per_image.png")


# ─────────────────────────────────────────
# 6. IMAGE SIZE DISTRIBUTION
# ─────────────────────────────────────────
def plot_image_sizes(data):
    """Scatter plot of image widths vs heights."""
    img_widths, img_heights = [], []

    for split in ["train", "val", "test"]:
        for entry in data[split]:
            if entry["img_path"] and os.path.exists(entry["img_path"]):
                img = cv2.imread(entry["img_path"])
                if img is not None:
                    h, w = img.shape[:2]
                    img_widths.append(w)
                    img_heights.append(h)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(img_widths, img_heights, alpha=0.3, s=15, c="darkorange")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Image Dimensions")
    axes[0].axline((0, 0), slope=1, color="gray", linestyle="--", alpha=0.5, label="1:1")
    axes[0].legend()

    axes[1].hist(img_widths, bins=30, alpha=0.6, color="steelblue", label="Width", edgecolor="white")
    axes[1].hist(img_heights, bins=30, alpha=0.6, color="coral", label="Height", edgecolor="white")
    axes[1].set_xlabel("Pixels")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Width & Height Distributions")
    axes[1].legend()

    plt.suptitle("Image Size Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "image_sizes.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved image_sizes.png")


# ─────────────────────────────────────────
# 7. SPLIT SUMMARY TABLE
# ─────────────────────────────────────────
def print_split_summary(data):
    """Print a summary table of the dataset splits."""
    print("\n" + "=" * 65)
    print(f"  {'Split':<8} {'Images':>8} {'Boxes':>8} {'Avg/Img':>10} {'Classes':>10}")
    print("=" * 65)

    for split in ["train", "val", "test"]:
        n_images = len(data[split])
        n_boxes = sum(len(e["boxes"]) for e in data[split])
        avg = n_boxes / n_images if n_images > 0 else 0
        classes_seen = set()
        for e in data[split]:
            for (cls_id, *_) in e["boxes"]:
                classes_seen.add(cls_id)
        print(f"  {split:<8} {n_images:>8} {n_boxes:>8} {avg:>10.2f} {len(classes_seen):>10}")

    total_imgs = sum(len(data[s]) for s in data)
    total_boxes = sum(sum(len(e["boxes"]) for e in data[s]) for s in data)
    print("-" * 65)
    print(f"  {'TOTAL':<8} {total_imgs:>8} {total_boxes:>8}")
    print("=" * 65)


# ─────────────────────────────────────────
# 8. SAMPLE ANNOTATED IMAGES (per class)
# ─────────────────────────────────────────
def plot_samples_per_class(data, n_classes=27):
    """Show one sample image per class with bounding boxes."""
    # Collect one sample per class from train
    class_samples = {}
    for entry in data["train"]:
        for (cls_id, *_) in entry["boxes"]:
            if cls_id not in class_samples and entry["img_path"]:
                class_samples[cls_id] = entry
            if len(class_samples) >= n_classes:
                break
        if len(class_samples) >= n_classes:
            break

    n = len(class_samples)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 4))
    axes = axes.flatten()

    random.seed(SEED)
    colors = {i: [random.random() for _ in range(3)] for i in range(NUM_CLASSES)}

    for idx, (cls_id, entry) in enumerate(sorted(class_samples.items())):
        img = cv2.imread(entry["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        axes[idx].imshow(img)
        axes[idx].set_title(CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}",
                           fontsize=10, fontweight="bold")
        axes[idx].axis("off")

        for (c, xc, yc, bw, bh) in entry["boxes"]:
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            box_w = int(bw * w)
            box_h = int(bh * h)
            color = colors.get(c, [1, 0, 0])
            rect = patches.Rectangle((x1, y1), box_w, box_h,
                                     linewidth=2, edgecolor=color, facecolor="none")
            axes[idx].add_patch(rect)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Image per Class (Training Set)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "samples_per_class.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved samples_per_class.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BRAND LOGO DETECTION — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n📂 Loading all labels...")
    data = load_all_labels()

    print("\n[1/7] Dataset split summary...")
    print_split_summary(data)

    print("\n[2/7] Class distribution per split...")
    plot_class_distribution(data)

    print("\n[3/7] Overall class balance...")
    plot_overall_balance(data)

    print("\n[4/7] Bounding box sizes...")
    plot_bbox_sizes(data)

    print("\n[5/7] Aspect ratios...")
    plot_aspect_ratios(data)

    print("\n[6/7] Objects per image...")
    plot_objects_per_image(data)

    print("\n[7/7] Image sizes...")
    plot_image_sizes(data)

    print("\n[BONUS] Sample images per class...")
    plot_samples_per_class(data)

    print("\n" + "=" * 60)
    print("  ✅ EDA COMPLETE! All figures saved to outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
