"""
run_pipeline.py — Brumadinho footprint-projection pipeline.

Runs the full deterministic flow:
    Stage 1 — Ingest (Bronze): copy raw PNGs + KML, compute MD5s.
    Stage 2 — Fetch footprints (only if cache file is missing).
    Stage 3 — Inference: project MS Building Footprints into pixel space,
              classify against the red impact polygon, and write
              annotated PNGs to data/gold/annotated/.

Usage (from brum/):
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path: ensure brum/ root is importable as "src.*"
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent        # brum/scripts/
_BRUM_ROOT  = _SCRIPT_DIR.parent                    # brum/
if str(_BRUM_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRUM_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _load_config(cfg_path: Path) -> dict:
    import yaml
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Stage 1 — Ingest (Bronze)
# ---------------------------------------------------------------------------

def stage_ingest(cfg: dict, brum_root: Path) -> tuple[list[dict], object]:
    """Copy raw PNGs to bronze, parse KML, compute MD5s."""
    _banner("STAGE 1 — INGEST (Bronze)")

    from src.data.ingest import copy_raw_to_bronze, parse_kml

    raw_dir    = (brum_root / cfg["paths"]["raw_dir"]).resolve()
    bronze_dir = (brum_root / cfg["paths"]["bronze_dir"]).resolve()
    kml_path   = (brum_root / cfg["paths"]["kml_path"]).resolve()

    print(f"[ingest] Raw dir    : {raw_dir}")
    print(f"[ingest] Bronze dir : {bronze_dir}")
    print(f"[ingest] KML path   : {kml_path}")

    t0 = time.time()
    bronze_records = copy_raw_to_bronze(raw_dir, bronze_dir, overwrite=True)
    print(f"[ingest] Copied {len(bronze_records)} files to bronze  ({time.time() - t0:.1f}s)")

    for rec in bronze_records:
        print(f"         {rec['tile_id']:30s}  md5={rec['checksum'][:12]}…  "
              f"{rec['file_type']}  {rec['image_width']}x{rec['image_height']}")

    t0 = time.time()
    kml_gdf = parse_kml(kml_path)
    print(f"[ingest] KML parsed: {len(kml_gdf)} geometries  ({time.time() - t0:.1f}s)")
    for _, row in kml_gdf.iterrows():
        print(f"         {row['placemark_id']:20s}  type={row.geometry.geom_type}  "
              f"n_coords={row['n_coords']}")

    return bronze_records, kml_gdf


# ---------------------------------------------------------------------------
# Stage 2 — Fetch MS Building Footprints (cache hit if file present)
# ---------------------------------------------------------------------------

def stage_fetch_footprints(cfg: dict, brum_root: Path) -> Path:
    """Ensure the MS Building Footprints geojsonl cache exists for the AOI bbox.

    Returns the path to the cached .geojsonl file.
    """
    _banner("STAGE 2 — FETCH FOOTPRINTS")

    from src.data.geo import get_footprints_for_bbox

    silver_dir = (brum_root / cfg["paths"]["silver_dir"]).resolve()
    footprints_dir = silver_dir / "footprints"
    footprint_path = footprints_dir / "211022203.geojsonl"

    if footprint_path.exists():
        print(f"[footprints] Cache hit: {footprint_path}")
        return footprint_path

    print(f"[footprints] Cache miss — downloading to {footprints_dir}")
    kml_bbox = cfg["kml_bbox"]
    t0 = time.time()
    buildings_gdf = get_footprints_for_bbox(
        lon_min=kml_bbox["lon_min"],
        lat_min=kml_bbox["lat_min"],
        lon_max=kml_bbox["lon_max"],
        lat_max=kml_bbox["lat_max"],
        output_dir=footprints_dir,
        zoom=9,
    )
    print(f"[footprints] Downloaded {len(buildings_gdf)} polygons  "
          f"({time.time() - t0:.1f}s)  →  {footprint_path}")
    return footprint_path


# ---------------------------------------------------------------------------
# Stage 3 — Inference (footprint projection)
# ---------------------------------------------------------------------------

def stage_inference(
    cfg: dict,
    brum_root: Path,
    footprint_path: Path,
    bronze_records: list[dict],
) -> list[dict]:
    """Annotate all raw PNGs by projecting MS Building Footprints into pixel space."""
    _banner("STAGE 3 — INFERENCE (footprint projection)")

    from scripts.run_footprint_inference import run_footprint_inference

    gold_dir      = (brum_root / cfg["paths"]["gold_dir"]).resolve()
    annotated_dir = gold_dir / "annotated"

    png_records = [r for r in bronze_records if r["file_type"] == "png"]
    image_paths = [Path(rec["bronze_path"]) for rec in png_records]

    results = run_footprint_inference(image_paths, footprint_path, annotated_dir, cfg)
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    _banner("SUMMARY")

    col_img = max(len("Image"), max(len(r["image"]) for r in results))
    hdr = f"{'Image':<{col_img}}  {'Total':>5}  {'Impact':>6}  {'Safe':>6}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    grand_total = grand_impact = grand_safe = 0
    for r in results:
        print(f"{r['image']:<{col_img}}  "
              f"{r['all_buildings']:>5}  "
              f"{r['buildings_in_impact_zone']:>6}  "
              f"{r['buildings_safe']:>6}")
        grand_total  += r["all_buildings"]
        grand_impact += r["buildings_in_impact_zone"]
        grand_safe   += r["buildings_safe"]

    print(sep)
    print(f"{'TOTAL':<{col_img}}  {grand_total:>5}  {grand_impact:>6}  {grand_safe:>6}")
    print()
    print("Annotated images:")
    for r in results:
        print(f"  {r['annotated_path']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    brum_root = _BRUM_ROOT
    cfg_path  = brum_root / "conf" / "config.yaml"

    print("Brumadinho Building Detection Pipeline (footprint projection)")
    print(f"  brum root : {brum_root}")
    print(f"  config    : {cfg_path}")

    cfg = _load_config(cfg_path)

    t_total = time.time()

    # Stage 1
    bronze_records, _ = stage_ingest(cfg, brum_root)

    # Stage 2
    footprint_path = stage_fetch_footprints(cfg, brum_root)

    # Stage 3
    results = stage_inference(cfg, brum_root, footprint_path, bronze_records)

    _print_summary(results)
    print(f"Pipeline complete in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
