# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: Raw Data Ingest
# MAGIC
# MAGIC Copies raw PNGs and KML into the bronze layer, computes MD5 checksums,
# MAGIC parses the KML impact-zone boundary, and writes metadata to a Delta table.

# COMMAND ----------

import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

with open("conf/config.yaml") as f:
    cfg = yaml.safe_load(f)

raw_dir = Path(cfg["paths"]["raw_dir"])
bronze_dir = Path(cfg["paths"]["bronze_dir"])
kml_path = Path(cfg["paths"]["kml_path"])

print(f"Raw dir:    {raw_dir}")
print(f"Bronze dir: {bronze_dir}")
print(f"KML:        {kml_path}")

# COMMAND ----------
# MAGIC %md ## 1. Copy raw files to Bronze (immutable layer)

from src.data.ingest import copy_raw_to_bronze

records = copy_raw_to_bronze(raw_dir, bronze_dir, overwrite=False)
print(f"Ingested {len(records)} files:")
for r in records:
    print(f"  {r['tile_id']:12s}  {r['file_type']:4s}  "
          f"{r['image_width']}×{r['image_height']}  {r['checksum'][:8]}…")

# COMMAND ----------
# MAGIC %md ## 2. Parse KML impact-zone boundary

from src.data.ingest import parse_kml

kml_gdf = parse_kml(kml_path)
print(kml_gdf[["placemark_id", "n_coords", "geometry"]].to_string())

impact_polygon = kml_gdf[kml_gdf["placemark_id"] == "merged_polygon"].geometry.iloc[0]
print(f"\nMerged polygon: {impact_polygon.geom_type}, "
      f"bounds = {impact_polygon.bounds}")

# COMMAND ----------
# MAGIC %md ## 3. Write Bronze metadata to Delta

from src.data.ingest import build_spark_session, log_bronze_metadata

spark = build_spark_session(cfg)
image_records = [r for r in records if r["file_type"] != "kml"]
log_bronze_metadata(image_records, bronze_dir, spark)

# COMMAND ----------
# MAGIC %md ## 4. Query Delta table

bronze_delta_path = str((bronze_dir / "delta" / "bronze_metadata").resolve())
df = spark.read.format("delta").load(bronze_delta_path)
df.orderBy("tile_id").show(truncate=False)
print(f"Total records: {df.count()}")
