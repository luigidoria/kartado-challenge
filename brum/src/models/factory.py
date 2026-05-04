"""Model factory: build and optionally load from checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.unet import BrumadinhoBuildingSegmenter


def build_model(
    cfg: dict,
    checkpoint_path: str | Path | None = None,
) -> BrumadinhoBuildingSegmenter:
    """Build a BrumadinhoBuildingSegmenter from config.

    Args:
        cfg: Config dict (uses cfg["model"] sub-dict).
        checkpoint_path: Optional path to a .pth checkpoint file.
            Expected format: {"model_state_dict": <state_dict>}

    Returns:
        Model in eval mode if checkpoint provided, train mode otherwise.
    """
    model_cfg = cfg.get("model", {})
    model = BrumadinhoBuildingSegmenter(
        encoder_name=model_cfg.get("encoder_name", "efficientnet-b0"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1),
    )

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        model.train()

    return model
