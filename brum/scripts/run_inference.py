#!/usr/bin/env python3
"""CLI inference entrypoint for Brumadinho Building Detection.

Usage:
    python scripts/run_inference.py \\
        --image data/raw/img0.png \\
        --checkpoint data/gold/best_model.pth \\
        --config conf/config.yaml \\
        --output data/gold/annotated_img0.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running from brum/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brumadinho Building Detection — inference CLI"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input satellite PNG",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        default="conf/config.yaml",
        help="Path to config.yaml (default: conf/config.yaml)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save annotated output PNG",
    )
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import torch
    import yaml
    from PIL import Image

    from src.data.transforms import get_val_transforms
    from src.inference.postprocess import run_full_postprocess
    from src.inference.predict import load_model_for_inference, predict_full_image
    from src.viz.overlay import add_legend, draw_building_overlays, save_annotated_image

    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_path = Path(args.output)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Brumadinho Building Detection ===")
    print(f"Image: {image_path}")

    # Load image
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil)
    if img_array.shape[2] == 4:
        img_rgb = img_array[:, :, :3]
    else:
        img_rgb = img_array

    # Load model
    model = load_model_for_inference(str(checkpoint_path), cfg, device)

    # Predict
    transform = get_val_transforms(cfg.get("chip_size", 512))
    prob_map = predict_full_image(img_rgb, model, transform, device, cfg)

    # Postprocess
    post = run_full_postprocess(prob_map, img_array, cfg)

    total = post["all_buildings"]
    n_impact = post["buildings_in_impact_zone"]
    n_safe = post["buildings_safe"]

    print(f"Total buildings detected: {total}")
    print(f"  Inside impact zone: {n_impact}  (red)")
    print(f"  Outside impact zone: {n_safe}  (green)")

    # Visualize
    annotated = draw_building_overlays(
        img_rgb,
        post["safe_polygons"],
        post["in_impact_polygons"],
        impact_zone_polygon=post.get("impact_polygon"),
    )
    annotated = add_legend(annotated, n_safe, n_impact)
    save_annotated_image(annotated, output_path)

    print(f"Annotated image saved: {output_path}")


if __name__ == "__main__":
    main()
