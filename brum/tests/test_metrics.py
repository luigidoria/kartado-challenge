"""Unit tests for src/training/metrics.py and src/training/losses.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mask(h: int, w: int, fill_value: float = 0.0) -> np.ndarray:
    return np.full((h, w), fill_value, dtype=np.float32)


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


def test_compute_iou_perfect():
    """Perfect overlap → IoU = 1.0."""
    from src.training.metrics import compute_iou

    pred = _make_mask(10, 10, 1.0)
    true = _make_mask(10, 10, 1.0)
    assert compute_iou(pred, true) == pytest.approx(1.0)


def test_compute_iou_no_overlap():
    """No overlap → IoU = 0.0."""
    from src.training.metrics import compute_iou

    pred = np.zeros((10, 10), dtype=np.float32)
    pred[:5] = 1.0
    true = np.zeros((10, 10), dtype=np.float32)
    true[5:] = 1.0
    assert compute_iou(pred, true) == pytest.approx(0.0)


def test_compute_iou_empty_masks():
    """Both masks empty → 0.0 (not NaN)."""
    from src.training.metrics import compute_iou

    pred = _make_mask(10, 10, 0.0)
    true = _make_mask(10, 10, 0.0)
    result = compute_iou(pred, true)
    assert result == pytest.approx(0.0)
    assert not np.isnan(result)


def test_compute_iou_partial_overlap():
    """50% overlap → IoU = 1/3 ≈ 0.333."""
    from src.training.metrics import compute_iou

    # Pred: left half; True: right half (with 50% center overlap)
    # pred: columns 0-7 (8 pixels per row in 10-wide), true: columns 2-9 (8 px per row)
    # overlap: columns 2-7 (6 px per row), union: columns 0-9 (10 px per row)
    pred = np.zeros((1, 10), dtype=np.float32)
    pred[0, :8] = 1.0  # 8 pixels
    true = np.zeros((1, 10), dtype=np.float32)
    true[0, 2:] = 1.0  # 8 pixels
    # intersection = 6, union = 10
    iou = compute_iou(pred, true)
    assert iou == pytest.approx(6 / 10, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_precision_recall_f1
# ---------------------------------------------------------------------------


def test_precision_recall_f1_perfect():
    """Perfect prediction → precision=recall=f1=1.0."""
    from src.training.metrics import compute_precision_recall_f1

    mask = np.ones((10, 10), dtype=np.float32)
    result = compute_precision_recall_f1(mask, mask)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_precision_recall_f1_all_fp():
    """All predicted positive, none truly positive → precision=recall=f1=0.0."""
    from src.training.metrics import compute_precision_recall_f1

    pred = np.ones((10, 10), dtype=np.float32)
    true = np.zeros((10, 10), dtype=np.float32)
    result = compute_precision_recall_f1(pred, true)
    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_precision_recall_f1_known_values():
    """4 TP, 2 FP, 1 FN → precision=4/6, recall=4/5, f1=2*(4/6)*(4/5)/(4/6+4/5)."""
    from src.training.metrics import compute_precision_recall_f1

    pred = np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.float32)
    true = np.array([1, 1, 1, 1, 0, 0, 1], dtype=np.float32)

    result = compute_precision_recall_f1(pred.reshape(1, -1), true.reshape(1, -1))

    expected_precision = 4 / 6
    expected_recall = 4 / 5
    expected_f1 = 2 * expected_precision * expected_recall / (
        expected_precision + expected_recall
    )

    assert result["precision"] == pytest.approx(expected_precision, abs=1e-4)
    assert result["recall"] == pytest.approx(expected_recall, abs=1e-4)
    assert result["f1"] == pytest.approx(expected_f1, abs=1e-4)


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_metrics_mean():
    """Mean of two dicts should equal element-wise average."""
    from src.training.metrics import aggregate_metrics

    m1 = {"iou": 0.4, "f1": 0.6}
    m2 = {"iou": 0.8, "f1": 0.2}
    result = aggregate_metrics([m1, m2])
    assert result["iou"] == pytest.approx(0.6)
    assert result["f1"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# DiceLoss
# ---------------------------------------------------------------------------


def test_dice_loss_perfect():
    """Perfect prediction → Dice loss near 0."""
    from src.training.losses import DiceLoss

    criterion = DiceLoss()
    logits = torch.full((2, 1, 8, 8), 10.0)  # sigmoid → ~1.0
    targets = torch.ones(2, 1, 8, 8)
    loss = criterion(logits, targets)
    assert float(loss) < 0.01


# ---------------------------------------------------------------------------
# CombinedLoss gradient
# ---------------------------------------------------------------------------


def test_combined_loss_gradient_flows():
    """Gradients must flow through CombinedLoss to logits (no NaN/zero grad)."""
    from src.training.losses import CombinedLoss

    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

    loss = criterion(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not (logits.grad == 0).all()
