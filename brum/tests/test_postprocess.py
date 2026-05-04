"""Unit tests for src/inference/postprocess.py."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from src.inference.postprocess import classify_buildings, mask_to_instances


# ---------------------------------------------------------------------------
# mask_to_instances
# ---------------------------------------------------------------------------


def test_mask_to_instances_empty():
    """All-zero probability map → no instances."""
    prob_map = np.zeros((100, 100), dtype=np.float32)
    result = mask_to_instances(prob_map)
    assert result == []


def test_mask_to_instances_single_blob():
    """A filled 50×50 square above threshold → exactly 1 polygon."""
    prob_map = np.zeros((200, 200), dtype=np.float32)
    prob_map[75:125, 75:125] = 1.0

    result = mask_to_instances(prob_map, min_area_px=30, threshold=0.5)
    assert len(result) == 1
    assert isinstance(result[0], Polygon)


def test_mask_to_instances_filters_small():
    """A 3×3 component (9px) with min_area_px=30 → filtered out."""
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[10:13, 10:13] = 1.0  # 9 pixels

    result = mask_to_instances(prob_map, min_area_px=30, threshold=0.5)
    assert result == []


def test_mask_to_instances_two_blobs():
    """Two separated squares → 2 polygons."""
    prob_map = np.zeros((300, 300), dtype=np.float32)
    prob_map[10:50, 10:50] = 1.0   # blob 1: 40×40 = 1600 px
    prob_map[200:250, 200:250] = 1.0  # blob 2: 50×50 = 2500 px

    result = mask_to_instances(prob_map, min_area_px=30, threshold=0.5)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# classify_buildings
# ---------------------------------------------------------------------------


def _square_poly(x0, y0, x1, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_classify_buildings_all_inside():
    """All buildings fully inside impact zone → all in_impact."""
    impact = _square_poly(0, 0, 100, 100)
    buildings = [_square_poly(10, 10, 20, 20), _square_poly(30, 30, 40, 40)]

    in_impact, safe = classify_buildings(buildings, impact)
    assert len(in_impact) == 2
    assert len(safe) == 0


def test_classify_buildings_all_outside():
    """All buildings outside impact zone → all safe."""
    impact = _square_poly(0, 0, 10, 10)
    buildings = [_square_poly(50, 50, 60, 60), _square_poly(70, 70, 80, 80)]

    in_impact, safe = classify_buildings(buildings, impact)
    assert len(in_impact) == 0
    assert len(safe) == 2


def test_classify_buildings_partial_overlap():
    """Building with 60% overlap → in_impact (threshold=0.5)."""
    impact = _square_poly(0, 0, 8, 10)  # 80×100 = 80 px area
    # Building: [2,0] → [12,10] = 100 px; overlap with impact = [2,0]→[8,10] = 60 px
    building = _square_poly(2, 0, 12, 10)

    in_impact, safe = classify_buildings([building], impact, overlap_threshold=0.5)
    assert len(in_impact) == 1
    assert len(safe) == 0


def test_classify_buildings_none_polygon():
    """No impact polygon → all buildings classified as safe."""
    buildings = [_square_poly(10, 10, 20, 20), _square_poly(30, 30, 40, 40)]

    in_impact, safe = classify_buildings(buildings, impact_polygon=None)
    assert len(in_impact) == 0
    assert len(safe) == 2
