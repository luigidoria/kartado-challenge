"""
run_pipeline.py — Brumadinho end-to-end building detection pipeline.

Runs the full flow:
    Stage 1 — Ingest (Bronze): copy raw PNGs + KML, compute MD5s.
    Stage 2 — Load YOLO model (path from config.yaml: yolo_model_path).
    Stage 3 — Inference: YOLO segmentation + flood-fill impact zone mask,
              classify each detection as inside/outside, write annotated
              PNGs to data/gold/annotated/.

Usage (from brum/):
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging
import re
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
# Stage 2 — Load YOLO model
# ---------------------------------------------------------------------------

def stage_load_model(cfg: dict, brum_root: Path):
    """Load the YOLO model from the path defined in config."""
    _banner("STAGE 2 — LOAD YOLO MODEL")

    from ultralytics import YOLO

    model_path = (brum_root / cfg.get("yolo_model_path", "data/raw/best.pt")).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    print(f"[model] Loading {model_path}")
    t0 = time.time()
    model = YOLO(str(model_path))
    print(f"[model] Loaded in {time.time() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Stage 3 — Inference (YOLO + raster zone mask)
# ---------------------------------------------------------------------------

def stage_inference(
    cfg: dict,
    brum_root: Path,
    model,
    bronze_records: list[dict],
) -> list[dict]:
    """Detect buildings in all bronze PNGs using YOLO + flood-fill zone mask."""
    _banner("STAGE 3 — INFERENCE (YOLO)")

    from scripts.pipeline_IA import run_pipeline as yolo_run_pipeline

    gold_dir      = (brum_root / cfg["paths"]["gold_dir"]).resolve()
    annotated_dir = gold_dir / "annotated"
    conf          = cfg.get("yolo_confidence_threshold", 0.10)

    png_records = [r for r in bronze_records if r["file_type"] == "png"]
    image_paths = [Path(rec["bronze_path"]) for rec in png_records]

    results = [
        yolo_run_pipeline(img_path, model, annotated_dir, conf)
        for img_path in image_paths
    ]
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
# Update presentation.html counts
# ---------------------------------------------------------------------------

def _update_presentation(results: list[dict], brum_root: Path) -> None:
    """Patch the count badges in presentation.html with the latest pipeline results.

    Finds each gallery-item block by image stem (img0, img1, …) and replaces
    the three count badges (total, safe, impact) in-place.
    """
    html_path = brum_root / "presentation.html"
    if not html_path.exists():
        logger.warning("[presentation] presentation.html not found — skipping update")
        return

    html = html_path.read_text(encoding="utf-8")

    for r in results:
        stem = Path(r["image"]).stem          # "img0"
        total   = r["all_buildings"]
        impact  = r["buildings_in_impact_zone"]
        safe    = r["buildings_safe"]

        # Match the three badges inside the gallery-item for this image.
        # The comment <!-- imgN --> anchors the replacement to the right block.
        pattern = (
            r"(<!--\s*" + re.escape(stem) + r"\s*-->.*?"
            r'<span class="count-badge count-total">).*?(</span>.*?'
            r'<span class="count-badge count-safe">).*?(</span>.*?'
            r'<span class="count-badge count-impact">).*?(</span>)'
        )
        replacement = (
            rf'\g<1>Total: {total} edificacoes\g<2>'
            rf'Seguras: {safe}\g<3>'
            rf'Zona de impacto: {impact}\g<4>'
        )
        html, n = re.subn(pattern, replacement, html, count=1, flags=re.DOTALL)
        if n:
            print(f"[presentation] {stem}: Total={total}  Impact={impact}  Safe={safe}")
        else:
            logger.warning("[presentation] Could not find gallery block for %s", stem)

    html_path.write_text(html, encoding="utf-8")
    print(f"[presentation] presentation.html atualizado -> {html_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    brum_root = _BRUM_ROOT
    cfg_path  = brum_root / "conf" / "config.yaml"

    print("Brumadinho Building Detection Pipeline (YOLO)")
    print(f"  brum root : {brum_root}")
    print(f"  config    : {cfg_path}")

    cfg = _load_config(cfg_path)

    t_total = time.time()

    # Stage 1
    bronze_records, _ = stage_ingest(cfg, brum_root)

    # Stage 2
    model = stage_load_model(cfg, brum_root)

    # Stage 3
    results = stage_inference(cfg, brum_root, model, bronze_records)

    _print_summary(results)
    _update_presentation(results, brum_root)
    print(f"Pipeline complete in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
