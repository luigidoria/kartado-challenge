# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: Preprocessing
# MAGIC
# MAGIC For each image: extract red pixels → estimate pixel-to-geo affine →
# MAGIC download MS Building Footprints → rasterize to mask → tile into 512×512 chips.
# MAGIC Logs chip metadata to a Silver Delta table.

# COMMAND ----------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from PIL import Image

with open("conf/config.yaml") as f:
    cfg = yaml.safe_load(f)

bronze_dir = Path(cfg["paths"]["bronze_dir"])
silver_dir = Path(cfg["paths"]["silver_dir"])
kml_bbox = cfg["kml_bbox"]
red_cfg = cfg["red_threshold"]

# COMMAND ----------
# MAGIC %md ## 1. Load bronze images and extract red pixels

from src.data.geo import (
    download_msft_building_footprints,
    estimate_pixel_to_geo_affine,
    extract_red_pixels,
    get_footprints_for_bbox,
    rasterize_buildings_to_mask,
)

image_paths = sorted(bronze_dir.glob("img*.png"))
print(f"Found {len(image_paths)} images in bronze layer")

affines = []
red_pixel_counts = []

for img_path in image_paths:
    img_array = np.array(Image.open(img_path))
    red_pixels = extract_red_pixels(
        img_array,
        r_min=red_cfg["r_min"],
        g_max=red_cfg["g_max"],
        b_max=red_cfg["b_max"],
    )
    red_pixel_counts.append(len(red_pixels))
    print(f"  {img_path.name}: {len(red_pixels)} red pixels "
          f"x=[{red_pixels[:, 0].min()},{red_pixels[:, 0].max()}]"
          if len(red_pixels) else f"  {img_path.name}: 0 red pixels")

    if len(red_pixels) >= 3:
        affine = estimate_pixel_to_geo_affine(red_pixels, kml_bbox)
    else:
        affine = {
            "px_x_min": 0, "px_x_max": img_array.shape[1],
            "px_y_min": 0, "px_y_max": img_array.shape[0],
            **kml_bbox,
            "scale_x": (kml_bbox["lon_max"] - kml_bbox["lon_min"]) / img_array.shape[1],
            "scale_y": (kml_bbox["lat_min"] - kml_bbox["lat_max"]) / img_array.shape[0],
        }
    affines.append(affine)

# COMMAND ----------
# MAGIC %md ## 2. Download MS Building Footprints for AOI

footprints_dir = silver_dir / "footprints"
buildings_gdf = get_footprints_for_bbox(
    lon_min=kml_bbox["lon_min"],
    lat_min=kml_bbox["lat_min"],
    lon_max=kml_bbox["lon_max"],
    lat_max=kml_bbox["lat_max"],
    output_dir=footprints_dir,
    zoom=9,
)
print(f"MS Building Footprints: {len(buildings_gdf)} polygons in AOI")

# COMMAND ----------
# MAGIC %md ## 3. Rasterize building masks for each image

mask_arrays = []
for img_path, affine in zip(image_paths, affines):
    img_array = np.array(Image.open(img_path))
    mask = rasterize_buildings_to_mask(buildings_gdf, img_array, affine)
    mask_arrays.append(mask)
    n_building_px = (mask > 0).sum()
    print(f"  {img_path.name}: {n_building_px} building pixels in mask")

# COMMAND ----------
# MAGIC %md ## 4. Tile images and masks → Silver chips

from src.data.tiling import tile_all_images, log_silver_metadata
from src.data.ingest import build_spark_session

silver_records = tile_all_images(image_paths, mask_arrays, silver_dir, affines, cfg)
print(f"\nChip split distribution:")
from collections import Counter
split_counts = Counter(r["split"] for r in silver_records)
for split, count in sorted(split_counts.items()):
    print(f"  {split:5s}: {count} chips")

# COMMAND ----------
# MAGIC %md ## 5. Write Silver metadata to Delta

spark = build_spark_session(cfg)
log_silver_metadata(silver_records, silver_dir, spark)

silver_delta_path = str((silver_dir / "delta" / "silver_metadata").resolve())
df = spark.read.format("delta").load(silver_delta_path)
df.groupBy("split").count().orderBy("split").show()
print(f"Total chips: {df.count()}")
