# Brumadinho Building Detection Pipeline

## Context

On January 25, 2019, the Córrego do Feijão iron ore tailings dam (Dam I) operated by Vale S.A. collapsed near Brumadinho, Minas Gerais, Brazil. The mudflow killed 270 people and contaminated the Paraopeba River. This pipeline detects buildings inside the confirmed mudflow impact zone from six pre-disaster satellite PNGs, enabling retrospective damage assessment and informing evacuation planning protocols.

## Problem

Given six RGBA satellite PNGs (~2075×750 px each) with an embedded red-pixel impact-zone boundary, and a KML file defining the boundary in geographic coordinates:

1. Detect all buildings visible in the imagery.
2. Classify each building as **inside** or **outside** the mudflow impact zone.
3. Produce an annotated image with a building count.

## Approach

**Weak supervision via Microsoft Building Footprints.** With no labeled building masks for Brumadinho, building footprints from the [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) dataset are downloaded for the AOI quadkeys and rasterized onto each image chip as training masks. The red boundary pixels serve as the spatial anchor for a per-image linear affine mapping from pixel space to geographic coordinates.

**Tile-and-stitch inference.** Each image is divided into 512×512 chips (stride 384). At inference, chips are predicted independently and stitched using Gaussian-weighted blending to suppress seam artifacts.

**Impact zone classification.** The red pixels in each image are extracted and closed into a convex-hull polygon in pixel space. Buildings are classified as "in impact" if their centroid falls inside this polygon or if their area overlap exceeds 50%.

## Architecture

```
Raw PNGs + KML
     │
     ▼
┌────────────────────────────────────────────────────────┐
│  BRONZE (ingest.py)                                    │
│  MD5 checksums · Delta metadata table                  │
│  KML → merged closed polygon                          │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  SILVER (geo.py · tiling.py · transforms.py)           │
│  Red pixel extraction → per-image affine               │
│  MS Building Footprints download (quadkey Z9)          │
│  Mask rasterization (cv2.fillPoly)                     │
│  512×512 chips · spatial block split (67/17/16%)       │
│  Delta metadata table                                  │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  GOLD (unet.py · train.py · losses.py)                 │
│  U-Net + EfficientNet-B0 (SMP 0.5.0)                  │
│  Loss: 0.5×BCEWithLogits + 0.5×Dice                   │
│  AdamW · CosineAnnealingLR · 50 epochs                │
│  MLflow autolog · checkpoint · pyfunc registry        │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  INFERENCE (predict.py · postprocess.py · overlay.py)  │
│  Tile → predict → Gaussian stitch                     │
│  Connected components · polygon contours               │
│  Impact zone classification (centroid + area overlap)  │
│  Annotated PNG with green/red overlays + legend        │
└────────────────────────────────────────────────────────┘
```

## Data Pipeline

| Stage  | Location               | Format        | Key artifacts                         |
|--------|------------------------|---------------|---------------------------------------|
| Bronze | `data/bronze/`         | PNG + Delta   | MD5-verified copies, metadata table  |
| Silver | `data/silver/chips/`   | PNG chips + Delta | 512×512 image/mask pairs, split info |
| Gold   | `data/gold/`           | .pth + Delta  | Best checkpoint, eval metrics table  |

## Results

| Metric    | Value (test set) |
|-----------|-----------------|
| IoU       | —               |
| Precision | —               |
| Recall    | —               |
| F1        | —               |

*Results populated after training on local hardware with downloaded MS Building Footprints. Placeholder values reflect the pipeline being designed for end-to-end execution, not pre-run artifact.*

## Limitations

1. **Partial-coverage affine approximation.** The pixel-to-geo affine is a linear fit between the red pixel bounding box and the KML bounding box. For images where red pixels cover only part of the image width (img1: x=[893,2074]; img4: x=[429,1598]), the affine maps only the visible boundary portion — introducing distortion for building footprints projected into those images.

2. **Weak labels may miss informal housing.** Microsoft Building Footprints are derived from aerial/satellite imagery using ML models trained primarily on formal, roofed structures. Informal housing with corrugated tin or earthen roofs common in Minas Gerais rural communities may be systematically underrepresented in training masks, causing the model to undercount such structures.

3. **ImageNet encoder bias.** The EfficientNet-B0 encoder uses ImageNet-pretrained weights. Features learned for European and North American building styles may not transfer optimally to Brazilian rural and agricultural structures, potentially reducing recall in such regions.

4. **Convex hull impact polygon.** `close_pixel_polygon()` uses a convex hull of the red boundary pixels. The actual mudflow boundary is non-convex (the river valley creates concave sections). Buildings near re-entrant sections of the true boundary may be mis-classified.

## Reproduction

```bash
# Clone and navigate
git clone <repo> && cd kartado/challenges/brum

# Install dependencies (~5 min on first run)
pip install -r requirements.txt

# Ingest raw data
python -c "
import yaml, sys
sys.path.insert(0, '.')
cfg = yaml.safe_load(open('conf/config.yaml'))
from src.data.ingest import copy_raw_to_bronze, parse_kml, build_spark_session, log_bronze_metadata
from pathlib import Path
records = copy_raw_to_bronze(cfg['paths']['raw_dir'], cfg['paths']['bronze_dir'])
spark = build_spark_session(cfg)
log_bronze_metadata([r for r in records if r['file_type'] != 'kml'],
                    cfg['paths']['bronze_dir'], spark)
"

# Run inference (requires trained checkpoint)
python scripts/run_inference.py \
    --image data/raw/img0.png \
    --checkpoint data/gold/best_model.pth \
    --config conf/config.yaml \
    --output data/gold/annotated_img0.png

# Run tests
pytest tests/ -v

# Lint
ruff check src/ scripts/ notebooks/
black --check src/ scripts/ notebooks/
```

Expected output:
```
=== Brumadinho Building Detection ===
Image: data/raw/img0.png
Total buildings detected: N
  Inside impact zone: M  (red)
  Outside impact zone: N-M  (green)
Annotated image saved: data/gold/annotated_img0.png
```

## Ethical Note

This pipeline is designed for retrospective damage assessment and humanitarian planning — not surveillance. Building detection results should be interpreted in conjunction with ground-truth field surveys. Automated counts should never be used as the sole basis for casualty estimates or insurance decisions. All data inputs are pre-disaster (January 2019), and no personal or identifying information is processed.

## License

MIT — see `LICENSE`.
