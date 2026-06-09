#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download pretrained DAGF-Swin weights.

Before publishing, replace PRETRAINED_URL with your GitHub Release / Zenodo file URL.

Usage:
  python scripts/download_pretrained.py
  python scripts/download_pretrained.py --output weights/dagf_swin_best.pth
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

# TODO: replace with your Release asset URL after uploading the checkpoint
PRETRAINED_URL = ""
PRETRAINED_SHA256 = ""  # optional, e.g. "abc123..."

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights",
    "dagf_swin_best.pth",
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--url", default=PRETRAINED_URL, help="Override download URL")
    args = ap.parse_args()

    url = (args.url or "").strip()
    if not url:
        print(
            "Pretrained URL is not configured yet.\n"
            "1. Upload dagf_swin_best.pth to GitHub Releases or Zenodo.\n"
            "2. Set PRETRAINED_URL in scripts/download_pretrained.py\n"
            "   or pass --url https://...\n"
            "3. Optionally set PRETRAINED_SHA256 for integrity check.\n"
            f"\nExpected local path after download: {args.output}"
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"Downloading -> {args.output}")
    urllib.request.urlretrieve(url, args.output)

    if PRETRAINED_SHA256:
        digest = sha256_file(args.output)
        if digest.lower() != PRETRAINED_SHA256.lower():
            os.remove(args.output)
            raise SystemExit(f"SHA256 mismatch: expected {PRETRAINED_SHA256}, got {digest}")

    print("Download complete.")


if __name__ == "__main__":
    main()
