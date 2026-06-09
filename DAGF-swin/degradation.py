# -*- coding: utf-8 -*-
"""Detail-smearing degradation synthesis (3:3:3:1 mix). Used for training-data preparation."""

from __future__ import annotations

import random
from io import BytesIO

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def apply_bilateral_filter(image_tensor, d=None, sigma_color=None, sigma_space=None):
    if d is None:
        d = random.choice([5, 7, 9])
    if sigma_color is None:
        sigma_color = random.uniform(30, 80)
    if sigma_space is None:
        sigma_space = random.uniform(30, 80)

    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    filtered = cv2.bilateralFilter(image_np, d, sigma_color, sigma_space)
    return torch.from_numpy(filtered).permute(2, 0, 1).float() / 255.0


def apply_downsampling_degradation(image_tensor, scale=None):
    if scale is None:
        scale = random.choice([0.5, 0.6, 0.7, 0.8])

    _, h, w = image_tensor.shape
    h_down, w_down = int(h * scale), int(w * scale)
    down = F.interpolate(
        image_tensor.unsqueeze(0), size=(h_down, w_down),
        mode="bilinear", align_corners=False,
    )
    up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
    return up.squeeze(0)


def apply_jpeg_compression(image_tensor, quality=None):
    if quality is None:
        quality = random.randint(50, 85)

    pil = TF.to_pil_image(image_tensor)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=False, subsampling="4:2:0")
    buf.seek(0)
    compressed = Image.open(buf)
    compressed.load()
    return TF.to_tensor(compressed)


def apply_mixed_degradation(image_tensor):
    degraded = apply_downsampling_degradation(image_tensor)
    degraded = apply_bilateral_filter(degraded)
    degraded = apply_jpeg_compression(degraded, quality=random.randint(60, 85))
    return degraded


def apply_detail_smearing_degradation(image_tensor, degradation_type="random"):
    """
    Args:
        image_tensor: (C, H, W) in [0, 1]
        degradation_type: bilateral | jpeg | downsampling | mixed | random
    """
    if degradation_type == "random":
        degradation_type = random.choices(
            ["bilateral", "jpeg", "downsampling", "mixed"],
            weights=[3, 3, 3, 1],
            k=1,
        )[0]

    if degradation_type == "bilateral":
        return apply_bilateral_filter(image_tensor)
    if degradation_type == "jpeg":
        return apply_jpeg_compression(image_tensor)
    if degradation_type == "downsampling":
        return apply_downsampling_degradation(image_tensor)
    if degradation_type == "mixed":
        return apply_mixed_degradation(image_tensor)
    return image_tensor.clone()
