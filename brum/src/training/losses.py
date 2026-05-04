"""Loss functions: Dice loss and combined BCE + Dice loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Applies sigmoid to logits internally before computing the Dice coefficient.
    Formula: 1 - (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: (B, 1, H, W) raw model output.
            targets: (B, 1, H, W) binary ground truth in [0, 1].

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice_coeff


class CombinedLoss(nn.Module):
    """Combined BCE-with-logits + Dice loss.

    loss = bce_weight * BCEWithLogits(logits, targets)
         + dice_weight * Dice(logits, targets)
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: (B, 1, H, W) raw model output.
            targets: (B, 1, H, W) binary ground truth float in [0, 1].

        Returns:
            Scalar combined loss.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
