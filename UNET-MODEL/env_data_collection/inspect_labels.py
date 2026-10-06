"""
Step 3: sanity-check collected (image, mask) pairs before training.

Randomly samples N pairs from data/ (or --data_dir), tiles RGB|mask|overlay
side by side, and saves the result as a contact sheet PNG. Also reports
simple stats (fully-black / fully-white masks) so you can spot bad
calibration or misaligned frames early.

Usage:
    env\\Scripts\\python.exe env_data_collection\\inspect_labels.py
    env\\Scripts\\python.exe env_data_collection\\inspect_labels.py --data_dir ..\\data_val --n 12
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np

import config_collection as cfg


def build_contact_sheet(pairs, tile_size=220):
    rows = []
    for img_path, mask_path in pairs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        img = cv2.resize(img, (tile_size, tile_size))
        mask_bgr = cv2.cvtColor(cv2.resize(mask, (tile_size, tile_size)), cv2.COLOR_GRAY2BGR)

        overlay = img.copy()
        red = np.zeros_like(img)
        red[..., 2] = 255
        obstacle = cv2.resize(mask, (tile_size, tile_size)) > 127
        overlay[obstacle] = cv2.addWeighted(img, 0.4, red, 0.6, 0)[obstacle]

        label = os.path.basename(img_path)
        strip = np.hstack([img, mask_bgr, overlay])
        strip = cv2.copyMakeBorder(strip, 0, 22, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(strip, label, (4, tile_size + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(strip)

    if not rows:
        return None
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=cfg.OUTPUT_ROOT)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--out", type=str, default="env_data_collection/label_check.png")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    masks_dir = os.path.join(args.data_dir, "masks")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    sample = random.sample(image_paths, min(args.n, len(image_paths)))
    pairs = [(p, os.path.join(masks_dir, os.path.basename(p))) for p in sample]
    pairs = [(i, m) for i, m in pairs if os.path.exists(m)]

    all_black, all_white, missing_mask = 0, 0, len(sample) - len(pairs)
    for _, mask_path in pairs:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if np.all(mask == 0):
            all_black += 1
        elif np.all(mask == 255):
            all_white += 1

    print(f"Checked {len(pairs)} pairs from {args.data_dir}:")
    print(f"  missing mask file : {missing_mask}")
    print(f"  fully-black masks : {all_black} (should be rare -- open-sky/open-field shots only)")
    print(f"  fully-white masks : {all_white} (should be rare -- fully-obstructed shots only)")

    sheet = build_contact_sheet(pairs)
    if sheet is not None:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, sheet)
        print(f"Saved contact sheet ({len(pairs)} rows, image|mask|overlay) to {args.out}")


if __name__ == "__main__":
    main()
