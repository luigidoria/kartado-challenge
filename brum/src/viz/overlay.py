"""Visualization: annotated building overlay on satellite imagery."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon


def draw_probability_overlay(
    img_array: np.ndarray,
    prob_map: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """Blend a probability heatmap (jet colormap) semi-transparently onto the image.

    Args:
        img_array: (H, W, 3) uint8 RGB image.
        prob_map: (H, W) float32 probability map in [0, 1].
        alpha: Heatmap opacity (0 = invisible, 1 = fully opaque). Default 0.35.

    Returns:
        (H, W, 3) uint8 RGB image with heatmap blended underneath.
    """
    h, w = img_array.shape[:2]

    # Resize prob_map to match image dimensions if needed
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalise to [0, 255] and apply jet colormap (OpenCV output is BGR)
    prob_u8 = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    # Convert BGR → RGB to match img_array
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend: result = (1-alpha)*img + alpha*heatmap
    blended = cv2.addWeighted(img_array.astype(np.uint8), 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended


def draw_building_overlays(
    img_array: np.ndarray,
    safe_polygons: list[Polygon],
    impact_polygons: list[Polygon],
    impact_zone_polygon: Polygon | None = None,
    alpha: float = 0.45,
    safe_color: tuple[int, int, int] = (0, 200, 0),
    impact_color: tuple[int, int, int] = (220, 0, 0),
    prob_map: np.ndarray | None = None,
    heatmap_alpha: float = 0.35,
) -> np.ndarray:
    """Overlay building masks and optional impact zone boundary on image.

    Safe buildings → filled green with alpha blending.
    Impact buildings → filled red with alpha blending.
    Impact zone boundary → orange polyline (no fill).
    Probability heatmap → jet colormap blended underneath (if prob_map provided).

    Args:
        img_array: (H, W, 3) or (H, W, 4) uint8 RGB/RGBA image.
        safe_polygons: Building polygons outside impact zone.
        impact_polygons: Building polygons inside impact zone.
        impact_zone_polygon: Overall impact zone boundary polygon.
        alpha: Building polygon overlay transparency (0 = transparent, 1 = opaque).
        safe_color: RGB fill color for safe buildings.
        impact_color: RGB fill color for impact buildings.
        prob_map: Optional (H, W) float32 probability map. When provided, a jet
            heatmap is blended into the base image before drawing polygons.
        heatmap_alpha: Opacity of the probability heatmap (default 0.35).

    Returns:
        (H, W, 3) uint8 annotated image.
    """
    # Ensure RGB
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        base = img_array[:, :, :3].copy()
    else:
        base = img_array.copy()

    # Apply probability heatmap as a base layer if provided
    if prob_map is not None:
        base = draw_probability_overlay(base, prob_map, alpha=heatmap_alpha)

    overlay = base.copy()

    def poly_to_cv2_pts(poly: Polygon) -> np.ndarray:
        coords = list(poly.exterior.coords)
        return np.array([[int(x), int(y)] for x, y in coords], dtype=np.int32)

    # Fill safe buildings (green)
    for poly in safe_polygons:
        if poly.is_empty:
            continue
        pts = poly_to_cv2_pts(poly)
        cv2.fillPoly(overlay, [pts], safe_color[::-1])  # OpenCV uses BGR

    # Fill impact buildings (red)
    for poly in impact_polygons:
        if poly.is_empty:
            continue
        pts = poly_to_cv2_pts(poly)
        cv2.fillPoly(overlay, [pts], impact_color[::-1])

    # Blend overlay with base
    annotated = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

    # Draw impact zone boundary (orange polyline)
    if impact_zone_polygon is not None and not impact_zone_polygon.is_empty:
        boundary_pts = np.array(
            [[int(x), int(y)] for x, y in impact_zone_polygon.exterior.coords],
            dtype=np.int32,
        )
        cv2.polylines(
            annotated,
            [boundary_pts],
            isClosed=True,
            color=(0, 165, 255),  # BGR orange
            thickness=2,
        )

    return annotated


def add_legend(
    img_array: np.ndarray,
    count_safe: int,
    count_impact: int,
) -> np.ndarray:
    """Add a semi-transparent legend box in the top-right corner.

    Shows:
        Safe buildings: {count_safe}  (green)
        Impact zone:    {count_impact}  (red)
        Total:          {count_safe + count_impact}

    Args:
        img_array: (H, W, 3) uint8 annotated image.
        count_safe: Number of safe buildings.
        count_impact: Number of buildings in impact zone.

    Returns:
        Image with legend drawn in-place (copy returned).
    """
    img = img_array.copy()
    h, w = img.shape[:2]

    total = count_safe + count_impact
    lines = [
        f"Total buildings: {total}",
        f"  Impact zone:  {count_impact}",
        f"  Safe:         {count_safe}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 22
    padding = 10

    # Compute box size
    max_w = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    box_w = max_w + 2 * padding
    box_h = len(lines) * line_height + 2 * padding

    x0 = w - box_w - 10
    y0 = 10
    x1 = w - 10
    y1 = y0 + box_h

    # Semi-transparent background
    sub = img[y0:y1, x0:x1].copy()
    bg = np.zeros_like(sub)
    bg[:] = (30, 30, 30)  # dark gray BGR
    img[y0:y1, x0:x1] = cv2.addWeighted(sub, 0.35, bg, 0.65, 0)

    # Draw text
    colors_bgr = [(255, 255, 255), (0, 0, 220), (0, 200, 0)]
    for i, (line, color_bgr) in enumerate(zip(lines, colors_bgr)):
        tx = x0 + padding
        ty = y0 + padding + (i + 1) * line_height - 4
        cv2.putText(img, line, (tx, ty), font, font_scale, color_bgr, thickness, cv2.LINE_AA)

    return img


def save_annotated_image(annotated_array: np.ndarray, output_path: str | Path) -> None:
    """Save an annotated numpy array as PNG.

    Args:
        annotated_array: (H, W, 3) uint8 RGB image.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PIL uses RGB, OpenCV uses BGR — annotated_array is RGB from our pipeline
    Image.fromarray(annotated_array).save(str(output_path))
