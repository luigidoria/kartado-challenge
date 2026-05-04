"""Inference postprocessing: instance extraction, classification, counting."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask → instances
# ---------------------------------------------------------------------------


def mask_to_instances(
    prob_map: np.ndarray,
    min_area_px: int = 30,
    threshold: float = 0.5,
) -> list[Polygon]:
    """Convert a probability map to a list of Shapely building polygons.

    Steps:
        1. Threshold → binary mask
        2. Morphological opening (3×3 kernel) to remove noise
        3. cv2.connectedComponentsWithStats → filter by area
        4. cv2.findContours per component → Shapely Polygon

    Args:
        prob_map: (H, W) float32 probability map.
        min_area_px: Minimum component area in pixels to keep.
        threshold: Binarization threshold.

    Returns:
        List of Shapely Polygons (one per detected building instance).
    """
    binary = (prob_map >= threshold).astype(np.uint8) * 255

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    polygons = []
    for label_id in range(1, n_labels):  # 0 is background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue

        # Extract component mask and find contour
        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # Take the largest contour
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue

        pts = contour.reshape(-1, 2)
        poly = Polygon([(int(p[0]), int(p[1])) for p in pts])
        if poly.is_valid and not poly.is_empty and poly.area >= min_area_px:
            polygons.append(poly)

    return polygons


# ---------------------------------------------------------------------------
# Building classification
# ---------------------------------------------------------------------------


def classify_buildings(
    building_polygons: list[Polygon],
    impact_polygon: Polygon | None,
    overlap_threshold: float = 0.5,
) -> tuple[list[Polygon], list[Polygon]]:
    """Classify buildings as inside or outside the impact zone.

    A building is "in impact" if:
        - Its centroid is inside the impact polygon, OR
        - Its area overlap with the impact polygon exceeds overlap_threshold.

    If impact_polygon is None, all buildings are classified as safe
    (conservative assumption — documented behavior).

    Args:
        building_polygons: List of building Polygons in pixel space.
        impact_polygon: Pixel-space impact zone polygon (may be None).
        overlap_threshold: Fraction of building area that must overlap.

    Returns:
        (in_impact, safe) — two lists of Polygons.
    """
    if impact_polygon is None:
        logger.warning(
            "[postprocess] No impact polygon available. "
            "All buildings classified as safe (conservative assumption)."
        )
        return [], list(building_polygons)

    in_impact: list[Polygon] = []
    safe: list[Polygon] = []

    for poly in building_polygons:
        centroid_inside = impact_polygon.contains(poly.centroid)

        if centroid_inside:
            in_impact.append(poly)
            continue

        # Check area overlap fraction
        try:
            intersection_area = poly.intersection(impact_polygon).area
            if poly.area > 0 and intersection_area / poly.area > overlap_threshold:
                in_impact.append(poly)
            else:
                safe.append(poly)
        except Exception:
            safe.append(poly)

    return in_impact, safe


# ---------------------------------------------------------------------------
# Full postprocessing pipeline
# ---------------------------------------------------------------------------


def run_full_postprocess(
    prob_map: np.ndarray,
    img_array: np.ndarray,
    cfg: dict,
) -> dict:
    """Orchestrate the full postprocessing pipeline.

    Steps:
        1. mask_to_instances() on prob_map
        2. extract_red_pixels() → close_pixel_polygon() → pixel-space impact zone
        3. classify_buildings() using impact zone polygon

    Args:
        prob_map: (H, W) float32 stitched probability map.
        img_array: (H, W, C) original image array (used for red pixel extraction).
        cfg: Config dict.

    Returns:
        Dict with:
            all_buildings (int)
            buildings_in_impact_zone (int)
            buildings_safe (int)
            in_impact_polygons (list[Polygon])
            safe_polygons (list[Polygon])
            impact_polygon (Polygon | None)
            prob_map (np.ndarray)
    """
    from src.data.geo import close_pixel_polygon, extract_red_pixels

    inf_cfg = cfg.get("inference", {})
    red_cfg = cfg.get("red_threshold", {})

    threshold = inf_cfg.get("threshold", 0.5)
    min_area_px = inf_cfg.get("min_area_px", 30)
    overlap_threshold = inf_cfg.get("overlap_threshold", 0.5)

    r_min = red_cfg.get("r_min", 150)
    g_max = red_cfg.get("g_max", 80)
    b_max = red_cfg.get("b_max", 80)

    # Step 1: detect building instances
    building_polygons = mask_to_instances(prob_map, min_area_px, threshold)

    # Step 2: extract red pixel boundary → impact polygon in pixel space
    red_pixels = extract_red_pixels(img_array, r_min=r_min, g_max=g_max, b_max=b_max)
    impact_polygon = close_pixel_polygon(red_pixels)

    # Step 3: classify
    in_impact, safe = classify_buildings(
        building_polygons, impact_polygon, overlap_threshold
    )

    return {
        "all_buildings": len(building_polygons),
        "buildings_in_impact_zone": len(in_impact),
        "buildings_safe": len(safe),
        "in_impact_polygons": in_impact,
        "safe_polygons": safe,
        "impact_polygon": impact_polygon,
        "prob_map": prob_map,
    }
