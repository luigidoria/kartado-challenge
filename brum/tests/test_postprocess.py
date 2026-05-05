"""Unit tests for src/inference/postprocess.py."""

from __future__ import annotations

from shapely.geometry import Polygon

from src.inference.postprocess import classify_buildings


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
