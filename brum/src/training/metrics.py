"""Evaluation metrics for binary segmentation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection over Union (IoU / Jaccard index).

    Args:
        pred_mask: (H, W) float probability map.
        true_mask: (H, W) binary ground truth.
        threshold: Binarization threshold for pred_mask.

    Returns:
        IoU in [0, 1]. Returns 0.0 for empty masks (not NaN).
    """
    pred_binary = (pred_mask >= threshold).astype(bool)
    true_binary = true_mask.astype(bool)

    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_precision_recall_f1(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision, recall, and F1 score.

    Args:
        pred_mask: (H, W) float probability map.
        true_mask: (H, W) binary ground truth.
        threshold: Binarization threshold for pred_mask.

    Returns:
        Dict with keys: precision, recall, f1.
        All values are 0.0 when the denominator is 0.
    """
    pred_binary = (pred_mask >= threshold).astype(bool)
    true_binary = true_mask.astype(bool)

    tp = np.logical_and(pred_binary, true_binary).sum()
    fp = np.logical_and(pred_binary, ~true_binary).sum()
    fn = np.logical_and(~pred_binary, true_binary).sum()

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean of each metric across a list of per-sample metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def compute_epoch_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mean IoU, precision, recall, F1 over an entire dataloader.

    Args:
        model: Segmentation model in eval mode.
        dataloader: DataLoader yielding (image_tensor, mask_tensor) pairs.
        device: Torch device.
        threshold: Probability threshold for binarization.

    Returns:
        Dict with keys: iou, precision, recall, f1.
    """
    model.eval()
    per_sample_metrics: list[dict[str, float]] = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
            masks_np = masks.cpu().numpy()  # (B, 1, H, W)

            for i in range(probs.shape[0]):
                p = probs[i, 0]  # (H, W)
                t = masks_np[i, 0]  # (H, W)

                iou = compute_iou(p, t, threshold)
                prf = compute_precision_recall_f1(p, t, threshold)
                per_sample_metrics.append({"iou": iou, **prf})

    return aggregate_metrics(per_sample_metrics)
