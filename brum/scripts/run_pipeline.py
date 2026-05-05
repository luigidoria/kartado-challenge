"""
run_pipeline.py — Brumadinho building detection pipeline (no Spark/Delta).

Usage:
    python scripts/run_pipeline.py [--skip-train] [--epochs N]

Working directory must be brum/:
    cd brum && python scripts/run_pipeline.py
"""

from __future__ import annotations

import argparse
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
# Stage 2 — Silver
# ---------------------------------------------------------------------------

def stage_silver(
    cfg: dict,
    brum_root: Path,
    bronze_records: list[dict],
    kml_gdf,
) -> list[dict]:
    """Extract red pixels, estimate affines, download MS footprints,
    rasterize masks, tile into chips."""
    _banner("STAGE 2 — SILVER")

    import numpy as np
    from PIL import Image

    from src.data.geo import (
        extract_red_pixels,
        estimate_pixel_to_geo_affine,
        get_footprints_for_bbox,
        rasterize_buildings_to_mask,
    )
    from src.data.tiling import tile_all_images

    silver_dir = (brum_root / cfg["paths"]["silver_dir"]).resolve()
    footprints_dir = silver_dir / "footprints"
    silver_dir.mkdir(parents=True, exist_ok=True)

    kml_bbox = cfg["kml_bbox"]
    red_cfg  = cfg.get("red_threshold", {})

    png_records = [r for r in bronze_records if r["file_type"] == "png"]
    print(f"[silver] Processing {len(png_records)} PNG images")

    print(f"[silver] Downloading MS Building Footprints …")
    t0 = time.time()
    buildings_gdf = get_footprints_for_bbox(
        lon_min=kml_bbox["lon_min"],
        lat_min=kml_bbox["lat_min"],
        lon_max=kml_bbox["lon_max"],
        lat_max=kml_bbox["lat_max"],
        output_dir=footprints_dir,
        zoom=9,
    )
    print(f"[silver] MS footprints: {len(buildings_gdf)} polygons  ({time.time() - t0:.1f}s)")

    image_paths: list[Path] = []
    mask_arrays: list[np.ndarray] = []
    affines:     list[dict] = []

    for rec in png_records:
        img_path = Path(rec["bronze_path"])
        print(f"\n[silver] {img_path.name}")

        with Image.open(img_path) as pil_img:
            img_array = np.array(pil_img)

        red_pixels = extract_red_pixels(
            img_array,
            r_min=red_cfg.get("r_min", 150),
            g_max=red_cfg.get("g_max", 80),
            b_max=red_cfg.get("b_max", 80),
        )
        print(f"         Red pixels: {len(red_pixels)}")

        if len(red_pixels) >= 2:
            affine = estimate_pixel_to_geo_affine(red_pixels, kml_bbox)
        else:
            h, w = img_array.shape[:2]
            affine = {
                "px_x_min": 0, "px_x_max": w - 1,
                "px_y_min": 0, "px_y_max": h - 1,
                "lon_min": kml_bbox["lon_min"], "lon_max": kml_bbox["lon_max"],
                "lat_min": kml_bbox["lat_min"], "lat_max": kml_bbox["lat_max"],
                "scale_x": (kml_bbox["lon_max"] - kml_bbox["lon_min"]) / max(w - 1, 1),
                "scale_y": (kml_bbox["lat_min"] - kml_bbox["lat_max"]) / max(h - 1, 1),
            }
            print(f"         WARNING: <2 red pixels; using full-image fallback affine")

        mask = rasterize_buildings_to_mask(buildings_gdf, img_array, affine)
        print(f"         Building px in mask: {int((mask > 0).sum())}")

        image_paths.append(img_path)
        mask_arrays.append(mask)
        affines.append(affine)

    print(f"\n[silver] Tiling {len(image_paths)} images …")
    t0 = time.time()
    silver_records = tile_all_images(image_paths, mask_arrays, silver_dir, affines, cfg)

    n_train = sum(1 for r in silver_records if r["split"] == "train")
    n_val   = sum(1 for r in silver_records if r["split"] == "val")
    n_test  = sum(1 for r in silver_records if r["split"] == "test")
    print(f"[silver] {len(silver_records)} chips  "
          f"({n_train} train / {n_val} val / {n_test} test)  ({time.time() - t0:.1f}s)")

    return silver_records


# ---------------------------------------------------------------------------
# Stage 3 — Train
# ---------------------------------------------------------------------------

def stage_train(
    cfg: dict,
    brum_root: Path,
    silver_records: list[dict],
    epochs_override: int | None = None,
) -> str:
    """Full training loop with MLflow local backend. Returns checkpoint path."""
    _banner("STAGE 3 — TRAIN")

    import mlflow
    from src.training.train import train

    gold_dir = (brum_root / cfg["paths"]["gold_dir"]).resolve()
    gold_dir.mkdir(parents=True, exist_ok=True)

    if epochs_override is not None:
        cfg = dict(cfg)
        cfg["training"] = dict(cfg.get("training", {}))
        cfg["training"]["epochs"] = epochs_override
        print(f"[train] Epochs overridden to {epochs_override}")

    # Resolve MLflow tracking URI to absolute path
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    if not Path(tracking_uri).is_absolute():
        tracking_uri = str((brum_root / tracking_uri).resolve())
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[train] MLflow URI       : {tracking_uri}")
    print(f"[train] Experiment       : {mlflow_cfg.get('experiment_name')}")
    print(f"[train] Epochs           : {cfg.get('training', {}).get('epochs', 50)}")

    t0 = time.time()
    best_ckpt = train(cfg, silver_records, output_dir=gold_dir)
    print(f"[train] Done in {time.time() - t0:.1f}s  →  {best_ckpt}")
    return best_ckpt


# ---------------------------------------------------------------------------
# Stage 4 — Inference
# ---------------------------------------------------------------------------

def stage_inference(
    cfg: dict,
    brum_root: Path,
    checkpoint_path: str,
    bronze_records: list[dict],
) -> list[dict]:
    """Predict on all raw PNGs, postprocess, save annotated PNGs."""
    _banner("STAGE 4 — INFERENCE")

    import numpy as np
    import torch
    from PIL import Image

    from src.data.transforms import get_val_transforms
    from src.inference.predict import predict_full_image, load_model_for_inference
    from src.inference.postprocess import run_full_postprocess
    from src.viz.overlay import draw_building_overlays, add_legend, save_annotated_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] Device      : {device}")
    print(f"[infer] Checkpoint  : {checkpoint_path}")

    model     = load_model_for_inference(checkpoint_path, cfg, device)
    transform = get_val_transforms(cfg.get("chip_size", 512))

    gold_dir      = (brum_root / cfg["paths"]["gold_dir"]).resolve()
    annotated_dir = gold_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    png_records = [r for r in bronze_records if r["file_type"] == "png"]
    results: list[dict] = []

    for rec in png_records:
        img_path = Path(rec["bronze_path"])
        print(f"\n[infer] {img_path.name} …")
        t0 = time.time()

        with Image.open(img_path) as pil_img:
            img_array = np.array(pil_img.convert("RGB"))

        prob_map = predict_full_image(img_array, model, transform, device, cfg)
        print(f"         prob_map max={prob_map.max():.3f}  ({time.time() - t0:.1f}s)")

        post = run_full_postprocess(prob_map, img_array, cfg)
        print(f"         total={post['all_buildings']}  "
              f"impact={post['buildings_in_impact_zone']}  "
              f"safe={post['buildings_safe']}")

        annotated = draw_building_overlays(
            img_array,
            safe_polygons=post["safe_polygons"],
            impact_polygons=post["in_impact_polygons"],
            impact_zone_polygon=post.get("impact_polygon"),
            prob_map=prob_map,
        )
        annotated = add_legend(annotated, post["buildings_safe"], post["buildings_in_impact_zone"])

        out_path = annotated_dir / f"{img_path.stem}_annotated.png"
        save_annotated_image(annotated, out_path)
        print(f"         → {out_path}")

        results.append({
            "image": img_path.name,
            "all_buildings": post["all_buildings"],
            "buildings_in_impact_zone": post["buildings_in_impact_zone"],
            "buildings_safe": post["buildings_safe"],
            "annotated_path": str(out_path),
        })

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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brumadinho building detection — end-to-end pipeline (no Spark)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; use existing checkpoint from gold_dir/best_model.pth",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help="Override training epochs (default: value in config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    brum_root = _BRUM_ROOT
    cfg_path  = brum_root / "conf" / "config.yaml"

    print("Brumadinho Building Detection Pipeline")
    print(f"  brum root  : {brum_root}")
    print(f"  config     : {cfg_path}")
    print(f"  skip-train : {args.skip_train}")
    if args.epochs:
        print(f"  epochs     : {args.epochs}")

    cfg = _load_config(cfg_path)

    t_total = time.time()

    # Stage 1
    bronze_records, kml_gdf = stage_ingest(cfg, brum_root)

    # Stage 2
    silver_records = stage_silver(cfg, brum_root, bronze_records, kml_gdf)

    # Stage 3
    gold_dir       = (brum_root / cfg["paths"]["gold_dir"]).resolve()
    best_ckpt_path = str(gold_dir / "best_model.pth")
    default_ckpt   = (brum_root / cfg["paths"]["checkpoint_path"]).resolve()

    if args.skip_train:
        _banner("STAGE 3 — TRAIN (SKIPPED)")
        if Path(best_ckpt_path).exists():
            checkpoint_path = best_ckpt_path
        elif default_ckpt.exists():
            checkpoint_path = str(default_ckpt)
        else:
            logger.error(
                f"No checkpoint found at {best_ckpt_path} or {default_ckpt}. "
                "Run without --skip-train first."
            )
            sys.exit(1)
        print(f"[train] Using: {checkpoint_path}")
    else:
        if not silver_records:
            logger.error("No silver chips — cannot train.")
            sys.exit(1)
        checkpoint_path = stage_train(cfg, brum_root, silver_records, args.epochs)

        mlflow_cfg = cfg.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
        if not Path(tracking_uri).is_absolute():
            tracking_uri = str((brum_root / tracking_uri).resolve())
        print(f"\n[mlflow] Experiment logged. To view results:")
        print(f"         mlflow ui --backend-store-uri {tracking_uri}")
        print(f"         → http://127.0.0.1:5000")
        print(f"         (or run: python scripts/show_results.py)")

    # Stage 4
    results = stage_inference(cfg, brum_root, checkpoint_path, bronze_records)

    _print_summary(results)
    print(f"Pipeline complete in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
