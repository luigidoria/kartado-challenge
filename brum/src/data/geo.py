"""Silver layer: spatial utilities for red pixel extraction, affine mapping,
MS Building Footprints download and rasterization."""

from __future__ import annotations

import gzip
import io
import json
import logging
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import MultiPoint, Polygon, shape

logger = logging.getLogger(__name__)

MSFT_DATASET_LINKS_URL = (
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
)


# ---------------------------------------------------------------------------
# Red pixel extraction
# ---------------------------------------------------------------------------


def extract_red_pixels(
    img_array: np.ndarray,
    r_min: int = 150,
    g_max: int = 80,
    b_max: int = 80,
) -> np.ndarray:
    """Return (N, 2) array of [x, y] pixel coordinates where pixels are "red".

    Red criterion: R >= r_min AND G <= g_max AND B <= b_max.
    Works with both RGB and RGBA arrays (alpha channel ignored).
    """
    r = img_array[:, :, 0].astype(np.int32)
    g = img_array[:, :, 1].astype(np.int32)
    b = img_array[:, :, 2].astype(np.int32)

    mask = (r >= r_min) & (g <= g_max) & (b <= b_max)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([xs, ys]).astype(np.int32)


# ---------------------------------------------------------------------------
# Pixel polygon
# ---------------------------------------------------------------------------


def close_pixel_polygon(
    pixel_coords: np.ndarray,
    method: str = "convex_hull",
) -> Polygon | None:
    """Wrap red pixel coords into a polygon.

    Args:
        pixel_coords: (N, 2) array of [x, y].
        method: "convex_hull" (default) — fast, no extra deps.

    Returns:
        Shapely Polygon or None if fewer than 3 unique points.
    """
    if pixel_coords is None or len(pixel_coords) < 3:
        return None

    pts = MultiPoint([(int(x), int(y)) for x, y in pixel_coords])
    if method == "convex_hull":
        hull = pts.convex_hull
        if hull.geom_type == "Polygon":
            return hull
        return None
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Pixel ↔ geo affine mapping
# ---------------------------------------------------------------------------


def estimate_pixel_to_geo_affine(
    pixel_coords: np.ndarray,
    kml_bbox: dict,
) -> dict:
    """Compute a linear affine mapping from red-pixel bounding box to KML bbox.

    Args:
        pixel_coords: (N, 2) [x, y] array of red pixels for this image.
        kml_bbox: dict with lon_min, lon_max, lat_min, lat_max.

    Returns:
        dict with: px_x_min, px_x_max, px_y_min, px_y_max,
                   lon_min, lon_max, lat_min, lat_max,
                   scale_x, scale_y
        NOTE: scale_y is negative — pixel y increases downward, lat decreases.
    """
    xs = pixel_coords[:, 0]
    ys = pixel_coords[:, 1]
    px_x_min, px_x_max = int(xs.min()), int(xs.max())
    px_y_min, px_y_max = int(ys.min()), int(ys.max())

    lon_min = kml_bbox["lon_min"]
    lon_max = kml_bbox["lon_max"]
    lat_min = kml_bbox["lat_min"]
    lat_max = kml_bbox["lat_max"]

    px_x_range = px_x_max - px_x_min
    px_y_range = px_y_max - px_y_min

    scale_x = (lon_max - lon_min) / px_x_range if px_x_range > 0 else 0.0
    # Negative: top of image (y=px_y_min) → lat_max; bottom (y=px_y_max) → lat_min
    scale_y = (lat_min - lat_max) / px_y_range if px_y_range > 0 else 0.0

    return {
        "px_x_min": px_x_min,
        "px_x_max": px_x_max,
        "px_y_min": px_y_min,
        "px_y_max": px_y_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def pixel_to_geo(x: float, y: float, affine: dict) -> tuple[float, float]:
    """Map pixel (x, y) → (lon, lat) using the per-image affine."""
    lon = affine["lon_min"] + (x - affine["px_x_min"]) * affine["scale_x"]
    lat = affine["lat_max"] + (y - affine["px_y_min"]) * affine["scale_y"]
    return lon, lat


def geo_to_pixel(lon: float, lat: float, affine: dict) -> tuple[float, float]:
    """Map (lon, lat) → pixel (x, y) using the per-image affine."""
    if affine["scale_x"] == 0:
        x = affine["px_x_min"]
    else:
        x = affine["px_x_min"] + (lon - affine["lon_min"]) / affine["scale_x"]
    if affine["scale_y"] == 0:
        y = affine["px_y_min"]
    else:
        y = affine["px_y_min"] + (lat - affine["lat_max"]) / affine["scale_y"]
    return x, y


# ---------------------------------------------------------------------------
# MS Building Footprints
# ---------------------------------------------------------------------------


def _get_quadkeys_for_bbox(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    zoom: int = 9,
) -> list[str]:
    """Return all quadkeys at *zoom* that intersect the given bounding box."""
    import mercantile

    tiles = list(mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zooms=zoom))
    return [mercantile.quadkey(t) for t in tiles]


def download_msft_building_footprints(
    quadkey: str,
    output_dir: str | Path,
    zoom: int = 9,
) -> Path:
    """Download MS Building Footprints for a quadkey.

    Downloads dataset-links.csv, finds the URL matching *quadkey*, decompresses
    the .csv.gz (GeoJSON-L format), and saves a .geojsonl file.

    FALLBACK: if the download fails for any reason, an empty GeoJSON file is
    written and its path is returned so the pipeline can continue.

    Args:
        quadkey: Tile quadkey string (e.g. "03121").
        output_dir: Directory to write output file.
        zoom: Tile zoom level (default 9).

    Returns:
        Path to the .geojsonl file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{quadkey}.geojsonl"

    # Return cached copy if present
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info(f"[geo] Cached footprints found: {out_path}")
        return out_path

    try:
        logger.info("[geo] Downloading MS Building Footprints dataset-links.csv …")
        resp = requests.get(MSFT_DATASET_LINKS_URL, timeout=30)
        resp.raise_for_status()

        # CSV has columns: Location, QuadKey, Url (or similar)
        # Find the row matching our quadkey
        url_for_qk: str | None = None
        for line in resp.text.splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and parts[1].strip().strip('"') == quadkey:
                url_for_qk = parts[2].strip().strip('"')
                break

        if url_for_qk is None:
            logger.warning(
                f"[geo] Quadkey {quadkey!r} not found in dataset-links.csv. "
                "Writing empty fallback."
            )
            _write_empty_geojsonl(out_path)
            return out_path

        logger.info(f"[geo] Downloading footprints from {url_for_qk} …")
        gz_resp = requests.get(url_for_qk, timeout=120, stream=True)
        gz_resp.raise_for_status()

        gz_bytes = gz_resp.content
        with gzip.open(io.BytesIO(gz_bytes), "rt", encoding="utf-8") as gz_file:
            content = gz_file.read()

        out_path.write_text(content, encoding="utf-8")
        logger.info(f"[geo] Footprints saved: {out_path}")

    except Exception as exc:
        logger.warning(
            f"[geo] Failed to download MS Building Footprints for quadkey "
            f"{quadkey!r}: {exc}. Writing empty fallback so pipeline continues."
        )
        _write_empty_geojsonl(out_path)

    return out_path


def _write_empty_geojsonl(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def load_footprints_geojsonl(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON-L file into a GeoDataFrame. Returns empty GDF if file is empty."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    features = []
    for line in text.splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") == "Feature":
                features.append(obj)
            elif obj.get("type") in ("Polygon", "MultiPolygon"):
                features.append({"type": "Feature", "geometry": obj, "properties": {}})
        except json.JSONDecodeError:
            continue

    if not features:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    geometries = [shape(f["geometry"]) for f in features]
    confidences = [f["properties"].get("confidence", 1.0) for f in features]
    return gpd.GeoDataFrame({"geometry": geometries, "confidence": confidences}, crs="EPSG:4326")


def get_footprints_for_bbox(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    output_dir: str | Path,
    zoom: int = 9,
) -> gpd.GeoDataFrame:
    """Download and return all MS Building Footprints intersecting the bbox."""
    quadkeys = _get_quadkeys_for_bbox(lon_min, lat_min, lon_max, lat_max, zoom)
    gdfs = []
    for qk in quadkeys:
        p = download_msft_building_footprints(qk, output_dir, zoom)
        gdf = load_footprints_geojsonl(p)
        if not gdf.empty:
            gdfs.append(gdf)

    if not gdfs:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    combined = gpd.pd.concat(gdfs, ignore_index=True)
    bbox_poly = Polygon(
        [
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
        ]
    )
    within = combined[combined.geometry.intersects(bbox_poly)]
    return within.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_buildings_to_mask(
    buildings_gdf: gpd.GeoDataFrame,
    image_array: np.ndarray,
    affine: dict,
) -> np.ndarray:
    """Rasterize building footprint polygons onto a binary pixel mask.

    Uses geo_to_pixel() to project each footprint polygon into pixel space,
    then cv2.fillPoly() to render filled regions.

    Args:
        buildings_gdf: GeoDataFrame with building polygon geometries (EPSG:4326).
        image_array: (H, W, C) image array (used only for shape).
        affine: Per-image affine dict from estimate_pixel_to_geo_affine().

    Returns:
        (H, W) uint8 binary mask (0 = background, 255 = building).
    """
    h, w = image_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if buildings_gdf.empty:
        return mask

    for geom in buildings_gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        polys = []
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)

        for poly in polys:
            if poly.is_empty:
                continue
            ring = list(poly.exterior.coords)
            pixel_ring = []
            for lon, lat in ring:
                px, py = geo_to_pixel(lon, lat, affine)
                px = int(np.clip(round(px), 0, w - 1))
                py = int(np.clip(round(py), 0, h - 1))
                pixel_ring.append([px, py])
            if len(pixel_ring) >= 3:
                pts = np.array([pixel_ring], dtype=np.int32)
                cv2.fillPoly(mask, pts, 255)

    return mask
