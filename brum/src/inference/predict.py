"""Inference: tile-stitch prediction with Gaussian-weighted blending."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

from src.data.tiling import compute_chip_grid, chip_image

logger = logging.getLogger(__name__)


def load_model_for_inference(
    checkpoint_path: str,
    cfg: dict,
    device: torch.device,
) -> nn.Module:
    """Load a model from checkpoint for inference.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        cfg: Config dict.
        device: Target torch device.

    Returns:
        Model in eval mode on device.
    """
    from src.models.factory import build_model

    model = build_model(cfg, checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()
    return model


def predict_on_chip(
    chip_array: np.ndarray,
    model: nn.Module,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Run inference on a single chip.

    Args:
        chip_array: (H, W, 3) uint8 RGB chip.
        model: Segmentation model in eval mode.
        transform: Val transforms (Normalize + ToTensorV2).
        device: Torch device.

    Returns:
        (H, W) float32 probability map.
    """
    # Pad to chip_size if needed (edge chips may be smaller)
    h, w = chip_array.shape[:2]
    aug = transform(image=chip_array)
    img_t = aug["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    model.eval()
    with torch.no_grad():
        logit = model(img_t)
        prob = torch.sigmoid(logit).squeeze().cpu().numpy()  # (H, W)

    return prob.astype(np.float32)


def predict_full_image(
    img_array: np.ndarray,
    model: nn.Module,
    transform,
    device: torch.device,
    cfg: dict,
) -> np.ndarray:
    """Tile an image, predict each chip, and stitch with Gaussian-weighted blending.

    Algorithm:
        1. Generate chip grid.
        2. For each chip: extract → predict → accumulate prob * weight.
        3. Normalize: prob_sum / weight_sum.
        Edge chips are sliced to actual bounds so weight kernels match.

    Args:
        img_array: (H, W, C) uint8 image (RGB or RGBA).
        model: Segmentation model in eval mode.
        transform: Val transforms.
        device: Torch device.
        cfg: Config dict with chip_size, stride, inference.gaussian_sigma_factor.

    Returns:
        (H, W) float32 stitched probability map.
    """
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    h, w = img_array.shape[:2]
    chip_size = cfg.get("chip_size", 512)
    stride = cfg.get("stride", 384)
    sigma_factor = cfg.get("inference", {}).get("gaussian_sigma_factor", 6)
    sigma = chip_size // sigma_factor

    # Pre-compute Gaussian weight kernel for full-size chips
    ones = np.ones((chip_size, chip_size), dtype=np.float32)
    full_weight_kernel = gaussian_filter(ones, sigma=sigma)
    full_weight_kernel = np.clip(full_weight_kernel, 1e-6, None)

    prob_sum = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    grid = compute_chip_grid(w, h, chip_size, stride)

    for x0, y0, x1, y1 in grid:
        chip = chip_image(img_array, (x0, y0, x1, y1), drop_alpha=False)
        actual_h = y1 - y0
        actual_w = x1 - x0

        # Pad chip to chip_size if it's smaller (edge chip)
        if actual_h < chip_size or actual_w < chip_size:
            padded = np.zeros((chip_size, chip_size, 3), dtype=np.uint8)
            padded[:actual_h, :actual_w] = chip[:actual_h, :actual_w]
            prob_full = predict_on_chip(padded, model, transform, device)
            # Slice prob and weight to actual chip size
            prob_chip = prob_full[:actual_h, :actual_w]
            weight_chip = full_weight_kernel[:actual_h, :actual_w]
        else:
            prob_chip = predict_on_chip(chip, model, transform, device)
            weight_chip = full_weight_kernel

        prob_sum[y0:y1, x0:x1] += prob_chip * weight_chip
        weight_sum[y0:y1, x0:x1] += weight_chip

    # Normalize
    weight_sum = np.where(weight_sum < 1e-9, 1.0, weight_sum)
    stitched = (prob_sum / weight_sum).astype(np.float32)
    return stitched
