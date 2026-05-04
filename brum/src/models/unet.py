"""U-Net + EfficientNet-B0 building segmentation model.

Uses segmentation_models_pytorch==0.5.0.
activation=None → model outputs raw logits (required for BCEWithLogitsLoss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BrumadinhoBuildingSegmenter(nn.Module):
    """U-Net with EfficientNet-B0 encoder, outputting raw logits.

    Output shape: (B, 1, H, W) — single-channel binary segmentation logits.
    Apply sigmoid externally for probabilities.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # raw logits — do NOT set to "sigmoid"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) input tensor, ImageNet-normalized.

        Returns:
            (B, 1, H, W) raw logit tensor.
        """
        return self.model(x)
