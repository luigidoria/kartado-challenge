"""Inference postprocessing: building classification against impact polygon."""

from __future__ import annotations

import logging

from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


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
