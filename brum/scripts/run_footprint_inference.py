#!/usr/bin/env python3
"""Deterministic catalog-projection inference.

Projects cached MS Building Footprints into pixel space via the per-image
red-pixel affine, classifies each by overlap with the impact zone, and draws
annotated output PNGs.

Usage (from brum/):
    python scripts/run_footprint_inference.py
    python scripts/run_footprint_inference.py --images data/bronze/img0.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running from brum/
_SCRIPT_DIR = Path(__file__).resolve().parent
_BRUM_ROOT = _SCRIPT_DIR.parent
if str(_BRUM_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRUM_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brumadinho footprint-projection building detection"
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Paths to input PNGs (default: all PNGs in data/bronze/)",
    )
    parser.add_argument(
        "--footprints",
        default=None,
        help="Path to .geojsonl footprint file (default: data/silver/footprints/211022203.geojsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for annotated PNGs (default: data/gold/annotated/)",
    )
    parser.add_argument(
        "--config",
        default="conf/config.yaml",
        help="Path to config.yaml (default: conf/config.yaml)",
    )
    return parser.parse_args()


def build_mud_mask(img_rgba: "np.ndarray") -> "np.ndarray":
    """Return boolean mask (H, W) True where pixel is mud-covered or transparent."""
    import numpy as np
    alpha = img_rgba[:, :, 3]
    r, g, b = img_rgba[:, :, 0], img_rgba[:, :, 1], img_rgba[:, :, 2]
    transparent = alpha < 128
    mud = (r > 110) & (g > 60) & (g < 140) & (b < 90) & (r > g) & (g > b)
    return transparent | mud


def is_building_mud_covered(centroid_px, mud_mask: "np.ndarray", radius: int = 5) -> bool:
    """Sample a small neighborhood around the building centroid in the mud mask."""
    cx, cy = int(centroid_px.x), int(centroid_px.y)
    h, w = mud_mask.shape
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return True  # out-of-bounds → exclude
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    patch = mud_mask[y0:y1, x0:x1]
    return patch.mean() > 0.60  # >60% of neighborhood is mud → buried


def run_footprint_inference(
    image_paths: list[Path],
    footprint_path: Path,
    output_dir: Path,
    cfg: dict,
) -> list[dict]:
    """Run deterministic footprint-based inference on a list of images.

    Args:
        image_paths: List of input PNG paths.
        footprint_path: Path to .geojsonl MS Building Footprints file.
        output_dir: Directory to write annotated PNGs.
        cfg: Config dict (needs kml_bbox and red_threshold keys).

    Returns:
        List of result dicts with keys: image, all_buildings,
        buildings_in_impact_zone, buildings_safe, annotated_path.
    """
    import numpy as np
    from PIL import Image
    from shapely.geometry import Polygon

    from src.data.geo import (
        extract_red_pixels,
        close_pixel_polygon,
        estimate_pixel_to_geo_affine,
        geo_to_pixel,
        load_footprints_geojsonl,
    )
    from src.inference.postprocess import classify_buildings
    from src.viz.overlay import draw_building_overlays, add_legend, save_annotated_image

    kml_bbox = cfg["kml_bbox"]
    red_cfg = cfg.get("red_threshold", {})
    r_min = red_cfg.get("r_min", 150)
    g_max = red_cfg.get("g_max", 80)
    b_max = red_cfg.get("b_max", 80)

    # Load and filter footprints to KML bbox
    print(f"[footprint] Loading footprints from {footprint_path} …")
    gdf = load_footprints_geojsonl(footprint_path)
    print(f"[footprint] Raw footprints: {len(gdf)}")

    bbox_poly = Polygon([
        (kml_bbox["lon_min"], kml_bbox["lat_min"]),
        (kml_bbox["lon_max"], kml_bbox["lat_min"]),
        (kml_bbox["lon_max"], kml_bbox["lat_max"]),
        (kml_bbox["lon_min"], kml_bbox["lat_max"]),
    ])
    gdf = gdf[gdf.geometry.intersects(bbox_poly)].reset_index(drop=True)
    print(f"[footprint] Footprints inside KML bbox: {len(gdf)}")

    min_conf = cfg.get("footprint_min_confidence", 0.90)
    if "confidence" in gdf.columns:
        gdf = gdf[gdf["confidence"] >= min_conf].reset_index(drop=True)
    print(f"[footprint] After confidence filter (≥{min_conf}): {len(gdf)}")

    if gdf.empty:
        print("[footprint] WARNING: No footprints found inside the KML bbox.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for img_path in image_paths:
        print(f"\n[footprint] Processing {img_path.name} …")

        img_rgba = np.array(Image.open(img_path).convert("RGBA"))
        mud_mask = build_mud_mask(img_rgba)
        img = img_rgba[:, :, :3]
        h, w = img.shape[:2]
        image_bounds = Polygon([(0, 0), (w, 0), (w, h), (0, h)])

        # Extract red pixels → impact zone polygon in pixel space
        red_pixels = extract_red_pixels(img, r_min=r_min, g_max=g_max, b_max=b_max)
        print(f"         Red pixels: {len(red_pixels)}")
        impact_polygon = close_pixel_polygon(red_pixels)

        # Per-image affine: red pixel bbox → KML geo bbox
        if len(red_pixels) >= 2:
            affine = estimate_pixel_to_geo_affine(red_pixels, kml_bbox)
        else:
            print("         WARNING: <2 red pixels; using full-image fallback affine")
            affine = {
                "px_x_min": 0, "px_x_max": w - 1,
                "px_y_min": 0, "px_y_max": h - 1,
                "lon_min": kml_bbox["lon_min"], "lon_max": kml_bbox["lon_max"],
                "lat_min": kml_bbox["lat_min"], "lat_max": kml_bbox["lat_max"],
                "scale_x": (kml_bbox["lon_max"] - kml_bbox["lon_min"]) / max(w - 1, 1),
                "scale_y": (kml_bbox["lat_min"] - kml_bbox["lat_max"]) / max(h - 1, 1),
            }

        # Project footprint polygons into pixel space
        pixel_polygons: list[Polygon] = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue

            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for poly in polys:
                if poly.is_empty:
                    continue
                pixel_ring = [
                    geo_to_pixel(lon, lat, affine)
                    for lon, lat in poly.exterior.coords
                ]
                pixel_poly = Polygon(pixel_ring)
                if not pixel_poly.is_valid:
                    pixel_poly = pixel_poly.buffer(0)
                if pixel_poly.is_empty:
                    continue
                if pixel_poly.intersects(image_bounds):
                    pixel_polygons.append(pixel_poly)

        print(f"         Footprints projected to pixel space: {len(pixel_polygons)}")

        # Filter out mud/flood-covered buildings
        pixel_polygons = [
            p for p in pixel_polygons
            if not is_building_mud_covered(p.centroid, mud_mask)
        ]
        print(f"         After mud filter: {len(pixel_polygons)}")

        # Classify buildings: inside vs outside impact zone
        in_impact, safe = classify_buildings(pixel_polygons, impact_polygon)
        total = len(pixel_polygons)
        print(f"         Total={total}  Impact={len(in_impact)}  Safe={len(safe)}")

        # Draw overlay and save
        annotated = draw_building_overlays(
            img,
            safe_polygons=safe,
            impact_polygons=in_impact,
            impact_zone_polygon=impact_polygon,
        )
        annotated = add_legend(annotated, len(safe), len(in_impact))

        out_path = output_dir / f"{img_path.stem}_annotated.png"
        save_annotated_image(annotated, out_path)
        print(f"         → {out_path}")

        results.append({
            "image": img_path.name,
            "all_buildings": total,
            "buildings_in_impact_zone": len(in_impact),
            "buildings_safe": len(safe),
            "annotated_path": str(out_path),
        })

    return results


def _print_summary(results: list[dict]) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print("  FOOTPRINT INFERENCE SUMMARY")
    print(bar)

    col_img = max(len("Image"), max(len(r["image"]) for r in results))
    hdr = f"{'Image':<{col_img}}  {'Total':>5}  {'Impact':>6}  {'Safe':>6}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    grand_total = grand_impact = grand_safe = 0
    for r in results:
        print(
            f"{r['image']:<{col_img}}  "
            f"{r['all_buildings']:>5}  "
            f"{r['buildings_in_impact_zone']:>6}  "
            f"{r['buildings_safe']:>6}"
        )
        grand_total += r["all_buildings"]
        grand_impact += r["buildings_in_impact_zone"]
        grand_safe += r["buildings_safe"]

    print(sep)
    print(f"{'TOTAL':<{col_img}}  {grand_total:>5}  {grand_impact:>6}  {grand_safe:>6}")
    print()
    print("Annotated images:")
    for r in results:
        print(f"  {r['annotated_path']}")
    print()


def main() -> None:
    import yaml

    args = parse_args()

    brum_root = _BRUM_ROOT
    cfg_path = brum_root / args.config
    if not cfg_path.exists():
        # Try resolving relative to cwd
        cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve image paths
    if args.images:
        image_paths = [Path(p).resolve() for p in args.images]
    else:
        bronze_dir = (brum_root / cfg["paths"]["bronze_dir"]).resolve()
        image_paths = sorted(bronze_dir.glob("*.png"))

    if not image_paths:
        print("ERROR: No PNG images found.", file=sys.stderr)
        sys.exit(1)

    # Resolve footprint path
    if args.footprints:
        footprint_path = Path(args.footprints).resolve()
    else:
        footprint_path = (
            brum_root / cfg["paths"]["silver_dir"] / "footprints" / "211022203.geojsonl"
        ).resolve()

    if not footprint_path.exists():
        print(f"ERROR: Footprint file not found: {footprint_path}", file=sys.stderr)
        print("  Run the full pipeline first: python scripts/run_pipeline.py", file=sys.stderr)
        sys.exit(1)

    # Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (brum_root / cfg["paths"]["gold_dir"] / "annotated").resolve()

    print("Brumadinho Footprint-Based Building Detection")
    print(f"  brum root     : {brum_root}")
    print(f"  images        : {len(image_paths)}")
    print(f"  footprints    : {footprint_path}")
    print(f"  output dir    : {output_dir}")

    results = run_footprint_inference(image_paths, footprint_path, output_dir, cfg)
    _print_summary(results)


if __name__ == "__main__":
    main()
