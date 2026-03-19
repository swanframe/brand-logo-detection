"""
prepare_data.py — Download, parse, convert, and split the Flickr Logos 27
dataset into YOLO format (train/val/test).

Annotation format (tab-separated):
    filename  class_name  subset_id  x1 y1 x2 y2

YOLO label format (space-separated, per .txt file):
    class_id  x_center_norm  y_center_norm  width_norm  height_norm
"""

import os
import sys
import glob
import shutil
import random
from collections import defaultdict

import cv2

# ── project imports ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    PROJECT_ROOT, DATA_RAW, DATA_PROCESSED,
    IMAGES_DIR, LABELS_DIR, DATASET_YAML,
    CLASS_NAMES, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
)


# ─────────────────────────────────────────
# 1. FIND RAW DATA FILES
# ─────────────────────────────────────────
def find_raw_files(raw_dir):
    """Locate annotation files and image directories in raw data."""
    annotations = {}
    image_dirs = []

    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            fpath = os.path.join(root, f)
            # Find annotation txt files (not readme/license)
            if f.endswith(".txt") and os.path.getsize(fpath) > 500:
                with open(fpath, "r") as fp:
                    first_line = fp.readline().strip()
                # Check if it looks like annotation (has coordinates)
                parts = first_line.split("\t") if "\t" in first_line else first_line.split()
                if len(parts) >= 4:
                    annotations[f] = fpath
                    print(f"  📄 Found annotation: {fpath}")

        img_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if img_files:
            image_dirs.append((root, len(img_files)))

    for d, count in image_dirs:
        print(f"  📂 Image dir: {d} ({count} images)")

    return annotations, image_dirs


# ─────────────────────────────────────────
# 2. PARSE ANNOTATIONS
# ─────────────────────────────────────────
def parse_annotations(raw_dir):
    """
    Parse all annotation files and return a dict:
        {image_filename: [(class_name, x1, y1, x2, y2), ...]}
    Handles both training set annotations (with bboxes)
    and query set annotations (class name only — skipped if no bbox).
    """
    annotations = defaultdict(list)
    class_set = set()
    skipped = 0

    # Find all txt files
    txt_files = glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True)

    for txt_file in txt_files:
        with open(txt_file, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Try tab-separated first, then space-separated
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = line.split()

                if len(parts) < 2:
                    continue

                filename = parts[0].strip()
                class_name = parts[1].strip()

                # Skip "no-logo" or empty classes
                if class_name.lower() in ("none", "no-logo", ""):
                    continue

                # Need at least coordinates for object detection
                # Format: filename class subset x1 y1 x2 y2
                # We need to find 4 numbers for bbox
                numbers = []
                for p in parts[2:]:
                    try:
                        numbers.append(int(p.strip()))
                    except ValueError:
                        try:
                            numbers.append(int(float(p.strip())))
                        except ValueError:
                            continue

                if len(numbers) >= 4:
                    x1, y1, x2, y2 = numbers[-4], numbers[-3], numbers[-2], numbers[-1]
                    # Basic validation
                    if x2 > x1 and y2 > y1:
                        annotations[filename].append((class_name, x1, y1, x2, y2))
                        class_set.add(class_name)
                    else:
                        skipped += 1
                else:
                    skipped += 1

    print(f"\n📊 Parsed annotations:")
    print(f"   Images with bboxes: {len(annotations)}")
    print(f"   Total bboxes: {sum(len(v) for v in annotations.values())}")
    print(f"   Classes found: {sorted(class_set)}")
    print(f"   Skipped lines: {skipped}")

    return dict(annotations), sorted(class_set)


# ─────────────────────────────────────────
# 3. BUILD CLASS MAPPING
# ─────────────────────────────────────────
def build_class_mapping(detected_classes):
    """
    Map class names to integer IDs.
    Uses CLASS_NAMES from config if all match, otherwise builds from detected.
    """
    # Normalize detected class names for matching
    name_map = {}
    for dc in detected_classes:
        # Try to match with config CLASS_NAMES (case-insensitive)
        matched = False
        for i, cn in enumerate(CLASS_NAMES):
            if dc.lower().replace(" ", "").replace("-", "") == \
               cn.lower().replace(" ", "").replace("-", ""):
                name_map[dc] = i
                matched = True
                break
        if not matched:
            print(f"  ⚠️  Unknown class: '{dc}' — assigning new ID")
            name_map[dc] = len(name_map)

    print(f"\n📋 Class mapping ({len(name_map)} classes):")
    for name, idx in sorted(name_map.items(), key=lambda x: x[1]):
        print(f"   {idx:2d}: {name}")

    return name_map


# ─────────────────────────────────────────
# 4. FIND IMAGE FILES
# ─────────────────────────────────────────
def find_image_files(raw_dir):
    """Build a mapping of filename -> full path for all images."""
    image_map = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for fpath in glob.glob(os.path.join(raw_dir, "**", ext), recursive=True):
            fname = os.path.basename(fpath)
            image_map[fname] = fpath
    print(f"\n🖼️  Found {len(image_map)} image files on disk")
    return image_map


# ─────────────────────────────────────────
# 5. CONVERT TO YOLO FORMAT
# ─────────────────────────────────────────
def convert_to_yolo(annotations, class_map, image_map):
    """
    Convert absolute bbox (x1,y1,x2,y2) to YOLO normalized format.
    Returns: {filename: [(class_id, xc, yc, w, h), ...]}
    """
    yolo_labels = {}
    skipped_no_image = 0
    skipped_bad_size = 0

    for filename, bboxes in annotations.items():
        if filename not in image_map:
            skipped_no_image += 1
            continue

        # Read image to get dimensions
        img = cv2.imread(image_map[filename])
        if img is None:
            skipped_bad_size += 1
            continue

        img_h, img_w = img.shape[:2]
        if img_h == 0 or img_w == 0:
            skipped_bad_size += 1
            continue

        yolo_bboxes = []
        for (class_name, x1, y1, x2, y2) in bboxes:
            if class_name not in class_map:
                continue

            class_id = class_map[class_name]

            # Clamp coordinates
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            # Convert to YOLO: normalized center + width/height
            xc = ((x1 + x2) / 2.0) / img_w
            yc = ((y1 + y2) / 2.0) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h

            # Validate
            if 0 < w <= 1 and 0 < h <= 1:
                yolo_bboxes.append((class_id, xc, yc, w, h))

        if yolo_bboxes:
            yolo_labels[filename] = yolo_bboxes

    print(f"\n🔄 YOLO conversion:")
    print(f"   Converted: {len(yolo_labels)} images")
    print(f"   Total boxes: {sum(len(v) for v in yolo_labels.values())}")
    print(f"   Skipped (no image file): {skipped_no_image}")
    print(f"   Skipped (bad image): {skipped_bad_size}")

    return yolo_labels


# ─────────────────────────────────────────
# 6. SPLIT DATASET
# ─────────────────────────────────────────
def split_dataset(yolo_labels, seed=SEED):
    """Split images into train/val/test with stratification by class."""
    random.seed(seed)

    # Group images by their primary class (first bbox class)
    class_images = defaultdict(list)
    for filename, bboxes in yolo_labels.items():
        primary_class = bboxes[0][0]
        class_images[primary_class].append(filename)

    train, val, test = [], [], []

    for cls_id, filenames in sorted(class_images.items()):
        random.shuffle(filenames)
        n = len(filenames)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val   = max(1, int(n * VAL_RATIO))

        train.extend(filenames[:n_train])
        val.extend(filenames[n_train:n_train + n_val])
        test.extend(filenames[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"\n📂 Dataset split:")
    print(f"   Train: {len(train)} images")
    print(f"   Val:   {len(val)} images")
    print(f"   Test:  {len(test)} images")
    print(f"   Total: {len(train) + len(val) + len(test)} images")

    return {"train": train, "val": val, "test": test}


# ─────────────────────────────────────────
# 7. WRITE YOLO FILES
# ─────────────────────────────────────────
def write_yolo_files(splits, yolo_labels, image_map, class_names):
    """Copy images and write YOLO label .txt files to processed dirs."""

    for split_name, filenames in splits.items():
        img_dir = os.path.join(IMAGES_DIR, split_name)
        lbl_dir = os.path.join(LABELS_DIR, split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for filename in filenames:
            # Copy image
            src_path = image_map[filename]
            dst_img  = os.path.join(img_dir, filename)
            shutil.copy2(src_path, dst_img)

            # Write YOLO label
            base = os.path.splitext(filename)[0]
            dst_lbl = os.path.join(lbl_dir, base + ".txt")
            with open(dst_lbl, "w") as fp:
                for (cls_id, xc, yc, w, h) in yolo_labels[filename]:
                    fp.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"\n✅ YOLO files written to {DATA_PROCESSED}")


# ─────────────────────────────────────────
# 8. CREATE dataset.yaml
# ─────────────────────────────────────────
def create_dataset_yaml(class_names):
    """Create the YAML config file required by YOLO training."""
    yaml_content = f"""# Brand Logo Detection — YOLOv8 dataset config
# Auto-generated by prepare_data.py

path: {DATA_PROCESSED}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    with open(DATASET_YAML, "w") as fp:
        fp.write(yaml_content)

    print(f"\n📄 dataset.yaml created at {DATASET_YAML}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BRAND LOGO DETECTION — DATA PREPARATION")
    print("=" * 60)

    # Step 1: Find raw files
    print("\n[1/7] Finding raw data files...")
    find_raw_files(DATA_RAW)

    # Step 2: Parse annotations
    print("\n[2/7] Parsing annotations...")
    annotations, detected_classes = parse_annotations(DATA_RAW)

    if not annotations:
        print("\n❌ No annotations found! Check your dataset structure.")
        print("   Expected: tab-separated file with filename, class, subset, x1, y1, x2, y2")
        return

    # Step 3: Build class mapping
    print("\n[3/7] Building class mapping...")
    class_map = build_class_mapping(detected_classes)
    # Build ordered list from class_map
    final_class_names = [""] * len(class_map)
    for name, idx in class_map.items():
        if idx < len(final_class_names):
            final_class_names[idx] = name

    # Step 4: Find image files
    print("\n[4/7] Finding image files...")
    image_map = find_image_files(DATA_RAW)

    # Step 5: Convert to YOLO format
    print("\n[5/7] Converting to YOLO format...")
    yolo_labels = convert_to_yolo(annotations, class_map, image_map)

    if not yolo_labels:
        print("\n❌ No valid YOLO labels generated! Check images and annotations.")
        return

    # Step 6: Split dataset
    print("\n[6/7] Splitting dataset...")
    splits = split_dataset(yolo_labels)

    # Step 7: Write files
    print("\n[7/7] Writing YOLO files & dataset.yaml...")
    write_yolo_files(splits, yolo_labels, image_map, final_class_names)
    create_dataset_yaml(final_class_names)

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ DATA PREPARATION COMPLETE!")
    print("=" * 60)
    total = sum(len(v) for v in splits.values())
    print(f"  Total images  : {total}")
    print(f"  Total classes : {len(class_map)}")
    print(f"  Train         : {len(splits['train'])}")
    print(f"  Val           : {len(splits['val'])}")
    print(f"  Test          : {len(splits['test'])}")
    print(f"  YAML          : {DATASET_YAML}")
    print("=" * 60)


if __name__ == "__main__":
    main()
