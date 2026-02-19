"""
build_dataset.py

Creates a train / val / test dataset directory structure from preprocessed
scaleogram images located in each modulation type's data_gen/preproc_cp folder.

Directory layout produced:
    dataset/
        train/  (1000 images per class)
        val/    (100 images per class)
        test/   (100 images per class)
    Each split contains subdirectories:
        8psk, 16qam, 64qam, bfm, bpsk, cpfsk, gfsk, pam4, qpsk

Source directories (relative to this script):
    ./<mod>/data_gen/preproc_cp/

Images are randomly shuffled, then partitioned so that train, val, and test
sets are mutually exclusive.
"""

import os
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = SCRIPT_DIR / "dataset"

SPLITS = {
    "train": 1000,
    "val": 100,
    "test": 100,
}

MOD_LABELS = ["8psk", "16qam", "64qam", "bfm", "bpsk", "cpfsk", "gfsk", "pam4", "qpsk"]

# Map each label to its source directory (relative to SCRIPT_DIR)
SOURCE_DIRS = {label: SCRIPT_DIR / label / "data_gen" / "preproc_cp" for label in MOD_LABELS}

RANDOM_SEED = 42

# Only consider files with these extensions as images
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_image_files(src_dir: Path) -> list[Path]:
    """Return a sorted list of image file paths in *src_dir*."""
    if not src_dir.is_dir():
        return []
    return sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_directory_structure():
    """Create dataset/{train,val,test}/{label} directories."""
    for split in ["train", "val", "test"]:
        for label in MOD_LABELS:
            dir_path = DATASET_ROOT / split / label
            dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory structure created under: {DATASET_ROOT}")


def populate_dataset():
    """Copy images into train, val, and test splits."""
    rng = random.Random(RANDOM_SEED)

    summary = []  # collect per-label stats for final report

    for label in MOD_LABELS:
        src_dir = SOURCE_DIRS[label]
        images = collect_image_files(src_dir)
        total = len(images)

        # Shuffle deterministically
        rng.shuffle(images)

        train_count = SPLITS["train"]
        val_count = SPLITS["val"]
        test_count = SPLITS["test"]
        needed = train_count + val_count + test_count

        if total == 0:
            print(f"  [{label:>5s}]  WARNING — source dir not found or empty: {src_dir}")
            summary.append((label, 0, 0, 0, 0))
            continue

        if total < needed:
            # Scale proportionally if not enough images
            ratio = total / needed
            train_count = int(round(train_count * ratio))
            val_count = int(round(val_count * ratio))
            test_count = total - train_count - val_count
            print(f"  [{label:>5s}]  WARNING — only {total} images available "
                  f"(need {needed}). Allocating train={train_count}, "
                  f"val={val_count}, test={test_count}")

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:train_count + val_count + test_count]

        # Copy files into their respective split directories
        for split_name, split_images in [("train", train_images),
                                          ("val", val_images),
                                          ("test", test_images)]:
            dest_dir = DATASET_ROOT / split_name / label
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)

        actual_train = len(train_images)
        actual_val = len(val_images)
        actual_test = len(test_images)
        print(f"  [{label:>5s}]  total={total:>5d}  "
              f"train={actual_train:>5d}  val={actual_val:>5d}  test={actual_test:>5d}")
        summary.append((label, total, actual_train, actual_val, actual_test))

    return summary


def print_summary(summary):
    """Print a final summary table."""
    print("\n" + "=" * 65)
    print(f"{'Label':>7s} | {'Source':>7s} | {'Train':>7s} | {'Val':>7s} | {'Test':>7s}")
    print("-" * 65)
    totals = [0, 0, 0, 0]
    for label, src, tr, va, te in summary:
        print(f"{label:>7s} | {src:>7d} | {tr:>7d} | {va:>7d} | {te:>7d}")
        totals[0] += src
        totals[1] += tr
        totals[2] += va
        totals[3] += te
    print("-" * 65)
    print(f"{'TOTAL':>7s} | {totals[0]:>7d} | {totals[1]:>7d} | {totals[2]:>7d} | {totals[3]:>7d}")
    print("=" * 65)


def main():
    print("Building dataset...")
    print(f"  Script directory : {SCRIPT_DIR}")
    print(f"  Dataset root     : {DATASET_ROOT}")
    print(f"  Random seed      : {RANDOM_SEED}")
    print()

    create_directory_structure()
    print("\nPopulating splits:")
    summary = populate_dataset()
    print_summary(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
