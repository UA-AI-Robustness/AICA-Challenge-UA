"""
setup_dataset.py — organize captured frames into a YOLO dataset.

Two jobs, depending on where you are in the workflow:

  1. COLLECT (before labeling): gather the RGB frames from captures/rgb into a
     flat folder you can upload to Roboflow / open in CVAT.
         python setup_dataset.py collect

  2. SPLIT (after local labeling): if you labeled locally (CVAT/LabelImg) and
     have images + YOLO-format .txt labels, split them into train/val under the
     standard YOLO layout next to data.yaml.
         python setup_dataset.py split --images labeled/images --labels labeled/labels --val 0.2

If you label in ROBOFLOW instead, you don't need 'split' — Roboflow exports a
ready dataset (images/, labels/, data.yaml). Just point training at that export.

Resulting layout (for the local route):
    dataset/
      data.yaml
      images/train/*.png
      images/val/*.png
      labels/train/*.txt
      labels/val/*.txt
"""

import argparse
import random
import shutil
from pathlib import Path


CAPTURES_RGB = Path("captures/rgb")
COLLECT_OUT = Path("dataset_inbox")          # flat folder to upload/label
DATASET_ROOT = Path("dataset")               # final YOLO dataset


def collect():
    """Copy captured RGB frames into a flat inbox for labeling."""
    if not CAPTURES_RGB.exists():
        print(f"[collect] no captures found at {CAPTURES_RGB.resolve()}")
        print("[collect] run a survey capture first (or point this at your frames).")
        return
    COLLECT_OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(CAPTURES_RGB.glob("*.png"))
    for p in imgs:
        shutil.copy2(p, COLLECT_OUT / p.name)
    print(f"[collect] copied {len(imgs)} image(s) to {COLLECT_OUT.resolve()}")
    print("[collect] next: upload this folder to Roboflow, or open it in CVAT/LabelImg to label.")


def split(images_dir: str, labels_dir: str, val_frac: float, seed: int):
    """Split locally-labeled images+labels into train/val under dataset/."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.exists() or not labels_dir.exists():
        print(f"[split] images_dir or labels_dir not found:\n  {images_dir}\n  {labels_dir}")
        return

    # pair images with their label files (same stem)
    pairs = []
    for img in sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))):
        lbl = labels_dir / (img.stem + ".txt")
        # images with no label file are treated as background (empty label ok)
        pairs.append((img, lbl if lbl.exists() else None))

    if not pairs:
        print(f"[split] no images found in {images_dir}")
        return

    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DATASET_ROOT / sub).mkdir(parents=True, exist_ok=True)

    def place(pairs_list, split_name):
        for img, lbl in pairs_list:
            shutil.copy2(img, DATASET_ROOT / "images" / split_name / img.name)
            dst_lbl = DATASET_ROOT / "labels" / split_name / (img.stem + ".txt")
            if lbl is not None:
                shutil.copy2(lbl, dst_lbl)
            else:
                dst_lbl.write_text("")  # background image: empty label

    place(train_pairs, "train")
    place(val_pairs, "val")
    print(f"[split] {len(train_pairs)} train, {len(val_pairs)} val -> {DATASET_ROOT.resolve()}")
    print("[split] copy your data.yaml into the dataset/ root before training.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AICA dataset setup")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("collect", help="gather captures/rgb into a flat folder for labeling")

    sp = sub.add_parser("split", help="split locally-labeled data into train/val")
    sp.add_argument("--images", required=True, help="folder of labeled images")
    sp.add_argument("--labels", required=True, help="folder of YOLO .txt labels")
    sp.add_argument("--val", type=float, default=0.2, help="val fraction (default 0.2)")
    sp.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.cmd == "collect":
        collect()
    elif args.cmd == "split":
        split(args.images, args.labels, args.val, args.seed)
