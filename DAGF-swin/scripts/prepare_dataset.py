#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate degraded / GT image pairs for DAGF-Swin training or evaluation.

Example:
  python scripts/prepare_dataset.py \\
    --gt_dir ./data/gt \\
    --out_dir ./data/train \\
    --mode random --seed 42

Output layout:
  out_dir/
    gt/   *.png
    low/  *.png   (same filenames as gt/)
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from degradation import apply_detail_smearing_degradation  # noqa: E402


def collect_images(path: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if os.path.isfile(path):
        return [path]
    files = []
    for name in sorted(os.listdir(path)):
        fp = os.path.join(path, name)
        if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in exts:
            files.append(fp)
    if not files:
        raise FileNotFoundError(f"No images found in {path}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Prepare degraded/GT pairs (detail-smearing synthesis)")
    ap.add_argument("--gt_dir", required=True, help="Folder or file of clean GT images")
    ap.add_argument("--out_dir", required=True, help="Output root (creates gt/ and low/)")
    ap.add_argument(
        "--mode", default="random",
        choices=["random", "bilateral", "jpeg", "downsampling", "mixed"],
        help="Degradation type; random = 3:3:3:1 mix",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy_gt", action="store_true", default=True,
                    help="Copy GT into out_dir/gt/ (default: True)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    gt_paths = collect_images(args.gt_dir)
    gt_out = os.path.join(args.out_dir, "gt")
    low_out = os.path.join(args.out_dir, "low")
    os.makedirs(gt_out, exist_ok=True)
    os.makedirs(low_out, exist_ok=True)

    for src in tqdm(gt_paths, desc="prepare"):
        name = os.path.basename(src)
        pil = Image.open(src).convert("RGB")
        gt_t = TF.to_tensor(pil)

        if args.mode == "random":
            low_t = apply_detail_smearing_degradation(gt_t.clone(), "random")
        else:
            low_t = apply_detail_smearing_degradation(gt_t.clone(), args.mode)

        if args.copy_gt:
            TF.to_pil_image(gt_t).save(os.path.join(gt_out, name))
        TF.to_pil_image(low_t).save(os.path.join(low_out, name))

    print(f"Done. {len(gt_paths)} pairs -> {args.out_dir}")
    print(f"  gt/  {gt_out}")
    print(f"  low/ {low_out}")


if __name__ == "__main__":
    main()
