"""pipeline_IA.py — YOLO-based building detection for Brumadinho.

Runs YOLO segmentation inference, builds the impact zone mask from red pixels
in the source image, and classifies each detected building as inside or outside
the zone.

Usage (from brum/):
    python scripts/pipeline_IA.py --images data/bronze/img0.png
    python scripts/pipeline_IA.py                          # all bronze PNGs
    python scripts/pipeline_IA.py --model data/raw/best.pt --output-dir results/
    python scripts/pipeline_IA.py --images img0.png --save-json detections/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# sys.path: ensure brum/ root is importable as "src.*"
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent   # brum/scripts/
_BRUM_ROOT = _SCRIPT_DIR.parent                 # brum/


def _load_config(cfg_path: Path) -> dict:
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: YOLO, image_path: Path, conf: float) -> list[dict]:
    results = model.predict(
        source=str(image_path),
        imgsz=1730,
        conf=conf,
        device="cpu",
    )
    predictions = []
    for result in results:
        if result.masks is None:
            continue
        for i, polygon in enumerate(result.masks.xy):
            predictions.append({
                "object_id": i,
                "class_id": int(result.boxes.cls[i]),
                "confidence": float(result.boxes.conf[i]),
                "polygon": polygon.tolist(),
            })
    return predictions


# ---------------------------------------------------------------------------
# Impact zone mask (pixel-space flood fill from red pixels)
# ---------------------------------------------------------------------------

def build_zone_mask(img_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a filled binary mask of the impact zone from red pixels.

    Detects the red impact-zone outline drawn on the image, closes open
    endpoints along the image border, then flood-fills to produce a solid mask.

    Returns:
        inside_mask: (H, W) uint8 — 255 inside the impact zone.
        bw: (H, W) uint8 — 255 on the red boundary pixels.
    """
    H, W = img_array.shape[:2]

    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    bw = np.zeros((H, W), dtype=np.uint8)
    bw[(r > 180) & (g < 80) & (b < 80)] = 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    bw = cv2.dilate(bw, np.ones((5, 5), np.uint8), iterations=1)

    def perimeter_pos(x: int, y: int) -> int:
        if y == 0:   return x
        if x == W-1: return (W-1) + y
        if y == H-1: return (W-1) + (H-1) + (W-1-x)
        return 2*(W-1) + (H-1) + (H-1-y)

    border_mask = np.zeros((H, W), dtype=np.uint8)
    border_mask[0, :] = border_mask[H-1, :] = border_mask[:, 0] = border_mask[:, W-1] = 255
    inter = cv2.bitwise_and(bw, border_mask)
    _, _, _, centroids = cv2.connectedComponentsWithStats(inter)
    raw_endpoints = [(int(c[0]), int(c[1])) for c in centroids[1:]]

    _, line_labels = cv2.connectedComponents(bw)

    def get_line_id(cx: int, cy: int) -> int:
        lid = int(line_labels[cy, cx])
        if lid == 0:
            nb = line_labels[max(0, cy-3):min(H, cy+4), max(0, cx-3):min(W, cx+4)]
            nz = nb[nb > 0]
            lid = int(np.bincount(nz.flatten()).argmax()) if len(nz) > 0 else 0
        return lid

    total_perim = 2 * (W + H - 2)
    groups: dict = defaultdict(list)
    for cx, cy in raw_endpoints:
        lid = get_line_id(cx, cy)
        pos = perimeter_pos(cx, cy)
        groups[lid].append((pos, (cx, cy)))

    endpoints: list[tuple[int, int]] = []
    for lid, pts in groups.items():
        if len(pts) == 1:
            continue
        elif len(pts) == 2:
            endpoints.extend(pt for _, pt in pts)
        else:
            best = max(
                ((i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))),
                key=lambda ij: min(
                    (pts[ij[1]][0] - pts[ij[0]][0]) % total_perim,
                    (pts[ij[0]][0] - pts[ij[1]][0]) % total_perim,
                ),
            )
            endpoints.extend([pts[best[0]][1], pts[best[1]][1]])

    endpoints.sort(key=lambda p: perimeter_pos(p[0], p[1]))
    for i in range(0, len(endpoints) - 1, 2):
        cv2.line(bw, endpoints[i], endpoints[i + 1], 255, thickness=5)

    bw_padded = np.pad(bw, 1, constant_values=0)
    flood = bw_padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    exterior = (flood[1:-1, 1:-1] == 255) & (bw == 0)
    inside_mask = np.zeros((H, W), dtype=np.uint8)
    inside_mask[~exterior & (bw == 0)] = 255
    inside_mask = cv2.bitwise_or(inside_mask, bw)

    return inside_mask, bw


# ---------------------------------------------------------------------------
# Annotation and counting
# ---------------------------------------------------------------------------

def annotate_and_count(
    img_array: np.ndarray,
    predictions: list[dict],
    inside_mask: np.ndarray,
    bw: np.ndarray,
    output_path: Path,
    conf: float,
) -> tuple[int, int]:
    """Draw building polygons on the image and save it.

    Returns:
        (inside_count, outside_count)
    """
    H, W = img_array.shape[:2]
    overlay = img_array.copy()

    tint = np.zeros_like(overlay)
    tint[inside_mask == 255] = (0, 60, 0)
    cv2.addWeighted(tint, 0.5, overlay, 1.0, 0, overlay)
    overlay[bw == 255] = (255, 255, 255)

    inside_count = outside_count = 0
    for obj in predictions:
        if obj.get("confidence", 0) < conf:
            continue
        poly = np.array(obj["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 255)
        if cv2.countNonZero(cv2.bitwise_and(poly_mask, inside_mask)) > 0:
            inside_count += 1
            color = (255, 0, 0)   # vermelho = zona de impacto
        else:
            outside_count += 1
            color = (0, 255, 0)   # verde = segura
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(overlay, [poly], True, (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(str(output_path))
    return inside_count, outside_count


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_path: Path,
    model: YOLO,
    output_dir: Path,
    conf: float,
    json_dir: Path | None = None,
) -> dict:
    """Run the full YOLO pipeline on a single image.

    Returns:
        dict with keys: image, all_buildings, inside, outside, annotated_path.
    """
    print(f"\n[YOLO] Processing {image_path.name} …")

    print("  [1/3] Running inference …")
    predictions = run_inference(model, image_path, conf)
    print(f"        {len(predictions)} raw detections")

    if json_dir is not None:
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"{image_path.stem}_detections.json"
        with open(json_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"        JSON → {json_path}")

    print("  [2/3] Building impact zone mask …")
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img).astype(np.uint8)
    inside_mask, bw = build_zone_mask(img_array)

    print("  [3/3] Annotating and counting …")
    out_path = output_dir / f"{image_path.stem}_annotated.png"
    inside, outside = annotate_and_count(img_array, predictions, inside_mask, bw, out_path, conf)

    total = inside + outside
    print(f"        Total={total}  Impact={inside}  Safe={outside}")
    print(f"        → {out_path}")

    return {
        "image": image_path.name,
        "all_buildings": total,
        "buildings_in_impact_zone": inside,
        "buildings_safe": outside,
        "annotated_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], conf: float) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print("  YOLO INFERENCE SUMMARY")
    print(bar)

    col = max(len("Image"), max(len(r["image"]) for r in results))
    hdr = f"{'Image':<{col}}  {'Total':>5}  {'Impact':>6}  {'Safe':>6}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    grand_total = grand_impact = grand_safe = 0
    for r in results:
        print(
            f"{r['image']:<{col}}  "
            f"{r['all_buildings']:>5}  "
            f"{r['buildings_in_impact_zone']:>6}  "
            f"{r['buildings_safe']:>6}"
        )
        grand_total += r["all_buildings"]
        grand_impact += r["buildings_in_impact_zone"]
        grand_safe += r["buildings_safe"]

    print(sep)
    print(f"{'TOTAL':<{col}}  {grand_total:>5}  {grand_impact:>6}  {grand_safe:>6}")
    print(f"\nConf threshold : {conf}")
    print("Annotated images:")
    for r in results:
        print(f"  {r['annotated_path']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brumadinho YOLO-based building detection"
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        metavar="PATH",
        help="Input satellite images (default: all PNGs in data/bronze/)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="YOLO model weights (.pt). Default: yolo_model_path from config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for annotated output PNGs (default: data/gold/annotated/)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Confidence threshold (default: yolo_confidence_threshold from config.yaml)",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        metavar="DIR",
        help="Save per-image raw detections as JSON files in this directory",
    )
    parser.add_argument(
        "--config",
        default="conf/config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: conf/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    if str(_BRUM_ROOT) not in sys.path:
        sys.path.insert(0, str(_BRUM_ROOT))

    args = parse_args()

    cfg_path = (_BRUM_ROOT / args.config).resolve()
    if not cfg_path.exists():
        cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = _load_config(cfg_path)

    # Resolve model path
    if args.model:
        model_path = Path(args.model).resolve()
    else:
        model_path = (_BRUM_ROOT / cfg.get("yolo_model_path", "data/raw/best.pt")).resolve()
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve confidence threshold
    conf = args.conf if args.conf is not None else cfg.get("yolo_confidence_threshold", 0.10)

    # Resolve image paths
    if args.images:
        image_paths = [Path(p).resolve() for p in args.images]
    else:
        bronze_dir = (_BRUM_ROOT / cfg["paths"]["bronze_dir"]).resolve()
        image_paths = sorted(bronze_dir.glob("*.png"))
    if not image_paths:
        print("ERROR: no images found.", file=sys.stderr)
        sys.exit(1)

    # Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (_BRUM_ROOT / cfg["paths"]["gold_dir"] / "annotated").resolve()

    # Resolve JSON dir
    json_dir = Path(args.save_json).resolve() if args.save_json else None

    print("Brumadinho YOLO Building Detection")
    print(f"  brum root  : {_BRUM_ROOT}")
    print(f"  model      : {model_path}")
    print(f"  images     : {len(image_paths)}")
    print(f"  output dir : {output_dir}")
    print(f"  conf       : {conf}")

    model = YOLO(str(model_path))

    results = [
        run_pipeline(img, model, output_dir, conf, json_dir)
        for img in image_paths
    ]

    _print_summary(results, conf)


if __name__ == "__main__":
    main()
