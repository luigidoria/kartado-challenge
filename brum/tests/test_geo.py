"""Unit tests for src/data/geo.py."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from src.data.geo import (
    close_pixel_polygon,
    estimate_pixel_to_geo_affine,
    extract_red_pixels,
    geo_to_pixel,
    pixel_to_geo,
    rasterize_buildings_to_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_affine():
    """A 100×100 pixel image mapping to a 1×1 degree bbox."""
    return {
        "px_x_min": 0,
        "px_x_max": 100,
        "px_y_min": 0,
        "px_y_max": 100,
        "lon_min": 10.0,
        "lon_max": 11.0,
        "lat_min": -21.0,
        "lat_max": -20.0,
        "scale_x": 0.01,
        "scale_y": -0.01,
    }


# ---------------------------------------------------------------------------
# extract_red_pixels
# ---------------------------------------------------------------------------


def test_extract_red_pixels_finds_known_red():
    """Synthetic image with 3 known red pixels — all should be found."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[10, 5] = [200, 30, 30]
    img[20, 15] = [255, 0, 0]
    img[30, 40] = [180, 70, 50]

    result = extract_red_pixels(img, r_min=150, g_max=80, b_max=80)
    assert result.shape[1] == 2
    found = set(map(tuple, result.tolist()))
    assert (5, 10) in found
    assert (15, 20) in found
    assert (40, 30) in found


def test_extract_red_pixels_empty_image():
    """All-black image → empty result with shape (0, 2)."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    result = extract_red_pixels(img)
    assert result.shape == (0, 2)


def test_extract_red_pixels_ignores_near_red():
    """R=200, G=100, B=100 is NOT red (g_max=80 exceeded)."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5, 5] = [200, 100, 100]  # G=100 > g_max=80 → should be excluded
    result = extract_red_pixels(img, r_min=150, g_max=80, b_max=80)
    assert result.shape == (0, 2)


# ---------------------------------------------------------------------------
# close_pixel_polygon
# ---------------------------------------------------------------------------


def test_close_pixel_polygon_convex_hull():
    """4 corners of a square → convex hull polygon contains all corners."""
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    poly = close_pixel_polygon(pts, method="convex_hull")
    assert poly is not None
    assert isinstance(poly, Polygon)
    for x, y in pts:
        # Convex hull boundary passes through all corners
        from shapely.geometry import Point
        assert poly.buffer(1e-6).contains(Point(x, y))


def test_close_pixel_polygon_returns_none_for_2_pixels():
    """Fewer than 3 points → None."""
    pts = np.array([[0, 0], [10, 0]])
    assert close_pixel_polygon(pts) is None


# ---------------------------------------------------------------------------
# geo ↔ pixel roundtrip
# ---------------------------------------------------------------------------


def test_geo_to_pixel_roundtrip(simple_affine):
    """pixel_to_geo → geo_to_pixel error should be < 1 pixel."""
    for px, py in [(0, 0), (50, 50), (100, 100), (25, 75)]:
        lon, lat = pixel_to_geo(px, py, simple_affine)
        rx, ry = geo_to_pixel(lon, lat, simple_affine)
        assert abs(rx - px) < 1.0, f"x roundtrip error: {abs(rx - px):.4f}"
        assert abs(ry - py) < 1.0, f"y roundtrip error: {abs(ry - py):.4f}"


# ---------------------------------------------------------------------------
# rasterize_buildings_to_mask
# ---------------------------------------------------------------------------


def test_rasterize_buildings_fills_correct_pixels(simple_affine):
    """A single square building polygon fills the expected pixel region."""
    import geopandas as gpd

    # Build a building polygon in geo space that should map to
    # roughly pixel [20,20] → [40,40] in a 100×100 image
    lon0, lat0 = pixel_to_geo(20, 20, simple_affine)
    lon1, lat1 = pixel_to_geo(40, 40, simple_affine)

    poly = Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = rasterize_buildings_to_mask(gdf, img, simple_affine)

    assert mask.dtype == np.uint8
    assert mask[30, 30] == 255, "Center of building should be filled"
    assert mask[5, 5] == 0, "Outside building should be empty"


# ---------------------------------------------------------------------------
# KML parsing (integration tests — require real KML file)
# ---------------------------------------------------------------------------


def _get_kml_path():
    from pathlib import Path

    candidates = [
        Path("data/raw/impact_zone.kml"),
        Path("brum/data/raw/impact_zone.kml"),
        Path("../data/raw/impact_zone.kml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@pytest.mark.skipif(_get_kml_path() is None, reason="KML file not present")
def test_parse_kml_returns_closed_polygon():
    """The merged polygon's exterior must start and end at the same point."""
    from src.data.ingest import parse_kml

    kml_path = _get_kml_path()
    gdf = parse_kml(kml_path)

    merged = gdf[gdf["placemark_id"] == "merged_polygon"]
    assert len(merged) == 1
    poly = merged.geometry.iloc[0]
    assert poly.geom_type == "Polygon"

    ext = list(poly.exterior.coords)
    assert ext[0] == ext[-1], "Polygon is not closed"


@pytest.mark.skipif(_get_kml_path() is None, reason="KML file not present")
def test_parse_kml_contains_brumadinho_coords():
    """Merged polygon should be near Brumadinho: lon~-44.15, lat~-20.15."""
    from src.data.ingest import parse_kml

    kml_path = _get_kml_path()
    gdf = parse_kml(kml_path)

    merged = gdf[gdf["placemark_id"] == "merged_polygon"]
    poly = merged.geometry.iloc[0]
    min_lon, min_lat, max_lon, max_lat = poly.bounds

    assert -44.25 < min_lon < -44.05, f"Unexpected lon range: {min_lon}"
    assert -20.25 < min_lat < -20.05, f"Unexpected lat range: {min_lat}"
