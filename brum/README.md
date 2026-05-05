# Brumadinho Building Detection Pipeline

## Context

On January 25, 2019, the Córrego do Feijão iron ore tailings dam (Dam I) operated by Vale S.A. collapsed near Brumadinho, Minas Gerais, Brazil. The mudflow killed 270 people and contaminated the Paraopeba River. This pipeline reports how many buildings from a pre-disaster footprint catalog fall inside the confirmed mudflow impact zone, enabling retrospective exposure assessment.

## Problem

Given six RGBA satellite PNGs (~2075×750 px each) with an embedded red-pixel impact-zone boundary, and a KML file defining the boundary in geographic coordinates:

1. Project a pre-disaster building catalog into each image.
2. Classify each projected building as **inside** or **outside** the mudflow impact zone.
3. Produce an annotated image with the per-image counts.

## Approach

**Catalog projection.** The [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) catalog is downloaded for the AOI quadkey, filtered by confidence ≥ 0.90, and projected into pixel space using a per-image affine derived from the embedded red boundary. Each projected polygon is classified inside/outside the impact zone by centroid containment plus area-overlap fallback. No neural inference at runtime. The numbers reflect catalog coverage, not what was visually detected in the imagery.

## Architecture

```
Raw PNGs + KML
     │
     ▼
┌────────────────────────────────────────────────────────┐
│  INGEST (ingest.py)                                    │
│  MD5 checksums · KML → merged closed polygon           │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  FETCH + FILTER FOOTPRINTS (geo.py)                    │
│  MS Building Footprints (quadkey 211022203, Z9)        │
│  Filter to KML bbox · confidence ≥ 0.90                │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  PROJECT + CLASSIFY + ANNOTATE                         │
│  (geo.py · postprocess.py · overlay.py)                │
│  Red pixel extraction → per-image affine               │
│  geo→pixel projection of every footprint polygon       │
│  Centroid + 50% area-overlap test vs. red polygon      │
│  Mud-mask filter · annotated PNG with green/red overlays│
└────────────────────────────────────────────────────────┘
```

## Data Pipeline

| Stage  | Location                          | Format        | Key artifacts                                  |
|--------|-----------------------------------|---------------|------------------------------------------------|
| Bronze | `data/bronze/`                    | PNG + KML     | MD5-verified copies, KML polygon               |
| Silver | `data/silver/footprints/`         | GeoJSONL      | MS Building Footprints catalog (quadkey Z9)    |
| Gold   | `data/gold/annotated/`            | PNG           | Annotated outputs with per-image building counts|

## Results

Per-image counts produced by the pipeline (from `last-session.md`):

| Image | Total | Impact zone | Safe |
|-------|------:|------------:|-----:|
| img0  | 2352  | 852         | 1500 |
| img1  | 2302  | 1509        | 793  |
| img2  | 2327  | 1843        | 484  |
| img3  | 2353  | 1640        | 713  |
| img4  | 2353  | 743         | 1610 |
| img5  | 2335  | 1870        | 465  |

These numbers are how many catalog polygons project into each image after the confidence filter and the mud-mask filter, and how many of those land inside (or substantially overlap) the red impact polygon. They reflect catalog coverage of the AOI — not visual detection.

## Limitations

1. **Catalog, not visual detection.** Counts are how many MS Building Footprints (captured 2020–2021 from Bing imagery) project into the impact polygon. We cannot detect buildings missing from the catalog, nor distinguish ones buried by mud from ones that never existed.

2. **Affine approximation drift.** The pixel-to-geo transform is linear, fitted to the visible red pixels in each image; it can accumulate distortion near the edges, displacing the projection for buildings near the boundary of the zone.

3. **Incomplete rural coverage.** The Microsoft Global ML Building Footprints dataset (2020–2021 release) has uneven coverage in rural Brazilian areas; informal housing and small structures are underrepresented.

4. **Convex impact boundary.** `close_pixel_polygon()` uses a convex hull of the red pixels; the real mud boundary is non-convex (river-valley re-entrants), which can misclassify buildings in concave sections.

## Reproduction

```bash
pip install -r requirements.txt
cd brum
python scripts/run_pipeline.py

# Run unit tests
pytest tests/ -v
```

Expected output:
```
=== STAGE 3 — INFERENCE (footprint projection) ===
[footprint] Loading footprints from data/silver/footprints/211022203.geojsonl …
[footprint] After confidence filter (≥0.9): N
…
img0.png  2352   852  1500
img1.png  2302  1509   793
…
```

## Ethical Note

This pipeline is designed for retrospective exposure assessment and humanitarian planning — not surveillance. Building counts should be interpreted in conjunction with ground-truth field surveys. Automated counts should never be used as the sole basis for casualty estimates or insurance decisions. All data inputs are pre-disaster (January 2019), and no personal or identifying information is processed.

## License

MIT — see `LICENSE`.
