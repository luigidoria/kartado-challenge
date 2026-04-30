# Brumadinho Challenge — Intern Handoff

**Project:** Building detection pipeline for the Brumadinho dam collapse impact zone
**Date:** 2026-04-30
**Status:** Discovery complete. Phase 0 (setup) not yet started. Ready to build.

---

## What this is

Build an AI pipeline that processes pre-disaster satellite imagery and identifies every building inside the Brumadinho mudflow path (Dam I collapse, Jan 25, 2019). The output is: a building count + the image annotated with detected buildings and the impact zone polygon.

This is humanitarian context. Treat the data with care.

---

## Where everything lives

```
/Users/home/dev/kartado/challenges/brum/
├── system-prompt.md          ← Full technical spec. READ THIS FIRST.
├── EVALUATION_RUBRIC.md      ← How the submission will be evaluated. Read before coding.
├── Desafio - Cientista de Dados.pdf  ← Original brief (deprecated on one point: see below)
└── data/
    └── raw/
        ├── img0.png  through  img5.png   ← 6 satellite images, RGBA, ~2075×750px each
        └── limites_rejeitos_dia29_v2.kml ← Impact zone boundary
```

---

## One deprecated requirement

The PDF says **Databricks is mandatory**. It is not. Use a local environment instead.
Everything else in the PDF and in `system-prompt.md` stands.

---

## What was discovered in this session

**Images:**
- 6 PNG files, RGBA (4 bands), approximately 2075×750px each
- Pre-disaster satellite frames (~July 2018, ~6 months before collapse)
- The red impact boundary line is already burned into every image as visible red pixels
- These are plain PNGs — **no embedded georeferencing, no CRS metadata**

**KML:**
- Single `Placemark` containing a `LineString` (not a Polygon — you must close it)
- Coordinates in lon/lat around `(-44.15, -20.15)` — Brumadinho, Minas Gerais
- Style is red outline, no fill

**Critical constraint:**
There are no labeled building masks. You cannot train from scratch. You need either:
- A pretrained model (SpaceNet, INRIA, xBD weights)
- Weak labels from Microsoft Building Footprints or Google Open Buildings for this AOI
- Or zero-shot with SAM/SAM2

---

## Agreed architecture

**U-Net + EfficientNet-B0 encoder** via `segmentation_models_pytorch`.

Why: images tile into ~6–8 chips of 512×512px. EfficientNet-B0 is the lightest encoder with good aerial generalization. Integrates cleanly with the medallion pipeline and MLflow pyfunc.

**Spatial filtering strategy:** images have no georeferencing. Use pixel-space approach:
1. Extract the red boundary pixels from each image
2. Close them into a pixel-space polygon
3. Use the KML for display/overlay purposes only
4. Intersect predicted building instance centroids with the pixel-space polygon

---

## What you need to build — in order

Follow the phases in `system-prompt.md` exactly. Summary:

**Phase 0 — Repo setup**
- Init git repo, `.gitignore` (data/, mlruns/, *.tif, *.pth, .venv/)
- `requirements.txt` with pinned versions
- Create folder skeleton: `src/`, `notebooks/`, `conf/`, `scripts/`, `tests/`, `data/bronze/`, `data/silver/`, `data/gold/`

**Phase 1 — Bronze (ingest)**
- Copy raw assets to bronze with checksums
- Parse KML → GeoDataFrame
- Log metadata to a Bronze Delta table via PySpark

**Phase 2 — Silver (preprocess)**
- Extract red line from images → close into pixel-space polygon
- Tile images into 512×512 chips
- Download weak labels (Microsoft Building Footprints for AOI) → rasterize to masks
- Spatial train/val/test split (NOT random — hold out a contiguous geographic block)
- Record everything in Silver Delta table

**Phase 3 — Gold (train + evaluate)**
- U-Net + EfficientNet-B0, loss = 0.5 * BCEWithLogits + 0.5 * DiceLoss
- MLflow: log every run, every epoch metric, sample prediction artifacts
- Register best checkpoint in MLflow Model Registry

**Phase 4 — Inference**
- `src/inference/predict.py`: tile → predict → stitch → connected components → polygon filter → count
- `src/viz/overlay.py`: annotate image (green = safe, red = in impact zone) + polygon outline
- `scripts/run_inference.py`: CLI entrypoint

**Phase 5 — Notebooks**
- 5 notebooks under `notebooks/`, each orchestrating `src/` modules
- Notebook 5 is a demo: load from MLflow registry, run on sample image, display output

**Phase 6 — Docs**
- README: context, approach, architecture, results, limitations, ethical note
- MIT or Apache-2.0 license
- Tag `v1.0.0`

---

## Quality bar (non-negotiable)

- Code passes `ruff` and `black`
- Type hints on all public functions
- Docstrings on every module and public function
- Unit tests: KML parsing, polygon rasterization, mask→instance conversion, polygon-intersection counting, metric implementations
- Fixed seeds, pinned deps
- README must let someone reproduce inference in <15 minutes from a fresh clone

---

## How the submission will be evaluated

Read `EVALUATION_RUBRIC.md` in full. The three questions your reviewer will ask:

1. Does the pipeline close? Raw image in → count + annotated image out. No gaps.
2. Did you address the real constraint? 6 unlabeled PNGs with no georeferencing is the actual problem. Generic model application without addressing it fails.
3. Would you trust this in production? Reproducible, documented, tested, honest about limitations.

**The spatial trap to not fall into:** the KML coordinates are lon/lat. The images are plain PNGs. Do not compare KML coordinates directly to pixel coordinates. Use the pixel-space polygon approach described above.

---

## How to start

```bash
cd /Users/home/dev/kartado/challenges/brum
git init
# then follow Phase 0 in system-prompt.md
```

Good luck.
