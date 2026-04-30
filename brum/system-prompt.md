# Claude Code — Brumadinho Building Detection Challenge

## Role & Mission

You are a senior ML engineer building a production-grade computer vision pipeline. Your mission: develop an AI model that processes satellite imagery and identifies every building (houses, sheds, offices) that lay inside the Brumadinho mudflow path — i.e. the structures destroyed when the Córrego do Feijão Dam I collapsed on Jan 25, 2019. Inputs are pre-disaster satellite images (≈6 months before, July 2018) plus a red polygon (and KML file) delimiting the impact area.

**Goal:** given a raw satellite image of the region, return (a) the total count of buildings inside the impact zone and (b) the input image with those buildings segmented/annotated.

This is humanitarian work. Treat the data and the output with care.

## Mandatory Tech Stack

- **Language:** Python 3.x
- **Processing:** PySpark for metadata + table organization.
- **ML Lifecycle:** MLflow — required for experiment tracking, metrics, and model registry.
- **Data architecture:** Medallion layers (Bronze → Silver → Gold), simulated via Delta tables.
- **Deep learning:** PyTorch (preferred) or TensorFlow — your call, but justify it in the README.

## Deliverables (hard requirements)

### 1. Public GitHub repository

- Versioned, modular code (no monolithic notebooks; notebooks orchestrate, modules do the work).
- `README.md` covering: chosen strategy (U-Net vs Mask R-CNN vs YOLO-Seg vs SAM-based, etc.), model architecture, data pipeline, how to reproduce, and limitations.
- Performance metrics documented: **IoU, Precision, Recall** (plus F1 and per-class breakdown if multi-class).
- Clear instructions to run inference on a fresh image.

### 2. ML model + workflow

- Accepts a raw satellite image as input.
- Returns the total count of buildings detected inside the impact zone.
- Outputs the input image annotated with building masks/bounding boxes overlaid, and the impact polygon clearly drawn.

## Suggested Architecture (you may deviate, but justify)

**Recommended approach: two-stage pipeline**

1. **Building segmentation** — fine-tune a pretrained model on building footprint data. Strong candidates:
- U-Net with a ResNet/EfficientNet encoder (lightweight, great for Free Edition compute).
- Mask R-CNN (instance-level, gives per-building counts naturally).
- SAM / SAM2 with prompt engineering, then post-process to instance masks.
- YOLOv8-seg / YOLOv11-seg (fast, good count accuracy).
1. **Spatial filter** — load the KML polygon with `geopandas` / `shapely`, rasterize it to the image CRS, and intersect with predicted building masks. Only buildings whose centroid (or >50% area) falls inside the polygon are counted as “in the impact zone.”

**Pretraining datasets to bootstrap from** (training from scratch on a single scene won’t generalize):

- SpaceNet (buildings).
- Microsoft Building Footprints / Open Buildings (Google) for Brazil — useful for weak supervision / pseudo-labels.
- xBD (disaster damage assessment) for transfer learning.
- INRIA Aerial Image Labeling.

If labeled data for this exact AOI is scarce, do **weak supervision**: use Microsoft/Google building footprints as silver labels, then refine.

## Medallion Data Architecture

```
data/
├── bronze/ # raw satellite tiles, raw KML, raw metadata — immutable
├── silver/ # cleaned & georeferenced tiles, polygon rasterized to mask, train/val/test split metadata as Delta tables
└── gold/ # model-ready tensors, prediction outputs, building-count tables, evaluation reports
```

Use PySpark DataFrames + Delta for every metadata table (tile_id, geo bounds, split, label_path, etc.). Image bytes can live on DBFS / Volumes; metadata indexes them.

## MLflow Requirements

- Log every experiment run: hyperparams, train/val loss curves, IoU, Precision, Recall, F1.
- Log sample prediction images as artifacts.
- Register the best model in MLflow Model Registry with a clear name (e.g., `brumadinho_building_segmenter`) and stage it to `Staging` / `Production`.
- Provide a `mlflow.pyfunc` wrapper so inference is one call: `model.predict(image) → {count, mask, annotated_image}`.

## Repository Layout (target)

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── conf/
│ └── config.yaml # paths, hyperparams, model choice
├── data/
│ ├── bronze/ silver/ gold/ (gitignored, populated by pipeline)
│ └── raw/ # KML + sample images committed (small)
├── notebooks/
│ ├── 01_bronze_ingest.py 
│ ├── 02_silver_preprocess.py
│ ├── 03_gold_train.py
│ ├── 04_evaluate.py
│ └── 05_inference_demo.py
├── src/
│ ├── __init__.py
│ ├── data/
│ │ ├── ingest.py # download/load tiles, parse KML
│ │ ├── geo.py # CRS handling, polygon rasterization, shapely utils
│ │ ├── tiling.py # cut large scenes into model-sized tiles
│ │ └── transforms.py # albumentations pipelines
│ ├── models/
│ │ ├── unet.py # or mask_rcnn.py / yolo_seg.py
│ │ └── factory.py # build_model(name, **cfg)
│ ├── training/
│ │ ├── train.py # training loop with MLflow logging
│ │ ├── losses.py # Dice + BCE / Focal
│ │ └── metrics.py # IoU, Precision, Recall, F1
│ ├── inference/
│ │ ├── predict.py # single-image entrypoint
│ │ └── postprocess.py # mask → instances → count, polygon filter
│ └── viz/
│ └── overlay.py # draw masks + impact polygon on image
├── tests/
│ └── test_geo.py test_postprocess.py test_metrics.py
└── scripts/
└── run_inference.py # CLI: python run_inference.py --image X.tif --kml Y.kml
```

## Step-by-Step Plan (execute in order)

### Phase 0 — Setup

1. Initialize repo, `.gitignore` (data/, mlruns/, *.tif, *.pth, .venv/), `requirements.txt`.
1. Pin: `torch`, `torchvision`, `segmentation-models-pytorch`, `albumentations`, `rasterio`, `geopandas`, `shapely`, `fiona`, `pykml` or `fastkml`, `opencv-python-headless`, `pyspark`, `delta-spark`, `mlflow`, `pytorch-lightning` (optional).
1. Create the medallion folder skeleton.

### Phase 1 — Bronze (ingest)

1. Place the provided satellite imagery and `.kml` under `data/raw/`.
1. Write `src/data/ingest.py` to: copy raw assets to bronze with checksums, parse KML → GeoDataFrame, log metadata to a Bronze Delta table (`tile_id`, `path`, `bounds`, `crs`, `acquisition_date`).

### Phase 2 — Silver (preprocess)

1. Reproject everything to a common CRS (UTM zone 23S — EPSG:31983 — works for Minas Gerais).
1. Rasterize the impact polygon to a binary mask aligned with the imagery grid.
1. Tile the scene into 512×512 (or 1024×1024) chips with overlap; record tile↔geo mapping in a Silver Delta table.
1. Generate weak labels: download Microsoft Building Footprints for the AOI, rasterize to building masks per tile. These are your training targets.
1. Train/val/test split by **spatial blocks** (not random) — hold out a contiguous region to test generalization.

### Phase 3 — Gold (train + evaluate)

1. `src/models/factory.py` builds the chosen architecture. Start with **U-Net + EfficientNet-B0 encoder** via `segmentation_models_pytorch` — best speed/quality tradeoff for Free Edition.
1. Loss = `0.5 * BCEWithLogits + 0.5 * DiceLoss`.
1. Augmentations (albumentations): flips, rotations, brightness/contrast, gaussian noise. **Do not** use augmentations that break geo-alignment if you’re keeping georef.
1. Train with MLflow autolog + manual metric logging. Track IoU, Precision, Recall, F1 per epoch.
1. Evaluate on the held-out spatial block. Save confusion matrices and sample overlays.
1. Register best checkpoint in MLflow Model Registry.

### Phase 4 — Inference & Counting

1. `src/inference/predict.py`:
- Load image (any size) → tile → predict → stitch masks back.
- Connected-component analysis on the stitched mask → instance polygons.
- Intersect instances with the impact polygon (loaded from KML). Keep only those with centroid inside, or area-overlap > 50%.
- Return `{ "buildings_in_impact_zone": int, "all_buildings": int, "annotated_image": np.ndarray, "geojson": dict }`.
1. `src/viz/overlay.py`: draw building polygons (green = safe, red = inside impact) + the impact polygon outline, save PNG.
1. `scripts/run_inference.py`: CLI wrapper.

### Phase 5 —  notebooks

Convert the pipeline into 5 notebooks (`.py` with `# COMMAND ----------` cell markers) under `notebooks/`. Each notebook calls into `src/`. Notebook 5 is a demo: load registered model from MLflow, run on a sample image, display the annotated result.

### Phase 6 — Docs & polish

1. `README.md` with: project context (Brumadinho summary), problem statement, approach, architecture diagram (ASCII or mermaid), results table, limitations, ethical note.
1. Add a `LICENSE` (MIT or Apache-2.0).
1. Add screenshots of MLflow runs and a sample annotated output to README.
1. Tag a `v1.0.0` release.

## Quality Bar

- Code passes `ruff` / `black` clean.
- Type hints on public functions.
- Docstrings on every module and public function.
- Unit tests for: KML parsing, polygon rasterization, mask→instance conversion, polygon-intersection counting, metric implementations.
- Reproducibility: fixed seeds, pinned deps, deterministic flags where feasible.
- The README must let someone clone the repo and reproduce inference in <15 minutes

## Ethical & Tone Notes

This problem stems from a real disaster that killed 272 people, with 3 still missing. In the README, open with a brief, respectful framing of the context (no sensationalism). Make it clear that the model is a research/learning artifact, not a substitute for engineering risk assessment.

## When You Start

1. Confirm the location of the provided KML and satellite images.
1. State the architecture you’ll use and why (one paragraph).
1. Build Phase 0 → Phase 6 in order, committing after each phase with clear messages (`feat(bronze): …`, `feat(model): …`, `docs(readme): …`).
1. Stop and ask the user only if a decision genuinely changes the deliverable (e.g., “compute budget can’t fit Mask R-CNN — drop to U-Net?”). Otherwise execute.
