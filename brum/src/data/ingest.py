"""Bronze layer: raw data ingestion, KML parsing, Delta metadata logging."""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from lxml import etree
from PIL import Image
from shapely.geometry import LineString, Polygon

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".kml"}


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def compute_md5(file_path: str | Path) -> str:
    """Return hex MD5 digest of *file_path*."""
    h = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Bronze copy
# ---------------------------------------------------------------------------


def copy_raw_to_bronze(
    raw_dir: str | Path,
    bronze_dir: str | Path,
    overwrite: bool = False,
) -> list[dict]:
    """Copy raw files to bronze (immutable) and return metadata records.

    Returns a list of dicts with keys:
        tile_id, source_path, bronze_path, checksum, file_type,
        image_width, image_height, bands, file_size_bytes
    """
    raw_dir = Path(raw_dir)
    bronze_dir = Path(bronze_dir)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for src in sorted(raw_dir.iterdir()):
        if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        dst = bronze_dir / src.name
        if dst.exists() and not overwrite:
            raise FileExistsError(
                f"Bronze file already exists: {dst}. "
                "Use overwrite=True to allow re-ingestion."
            )

        shutil.copy2(src, dst)
        checksum = compute_md5(dst)
        file_size = dst.stat().st_size

        width, height, bands = 0, 0, 0
        if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            try:
                with Image.open(dst) as img:
                    width, height = img.size
                    bands = len(img.getbands())
            except Exception:
                pass

        records.append(
            {
                "tile_id": src.stem,
                "source_path": str(src.resolve()),
                "bronze_path": str(dst.resolve()),
                "checksum": checksum,
                "file_type": src.suffix.lower().lstrip("."),
                "image_width": width,
                "image_height": height,
                "bands": bands,
                "file_size_bytes": file_size,
            }
        )

    return records


# ---------------------------------------------------------------------------
# KML parsing
# ---------------------------------------------------------------------------


def _parse_coordinates_text(text: str) -> list[tuple[float, float]]:
    """Parse a KML coordinates string into a list of (lon, lat) tuples."""
    coords = []
    for token in text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            coords.append((lon, lat))
    return coords


def parse_kml(kml_path: str | Path) -> gpd.GeoDataFrame:
    """Parse a KML file with two LineString placemarks and return a GeoDataFrame.

    The two placemarks share their first coordinate (branch point).
    Row 0 = Placemark 1 (main ring), Row 1 = Placemark 2 (branch),
    Row 2 = merged closed polygon.

    Returns GDF with columns: placemark_id, geometry, n_coords
    """
    kml_path = Path(kml_path)
    tree = etree.parse(str(kml_path))
    root = tree.getroot()

    coord_nodes = root.findall(".//kml:LineString/kml:coordinates", KML_NS)
    if not coord_nodes:
        # Try without namespace (some KML files omit it)
        coord_nodes = root.findall(".//LineString/coordinates")

    if len(coord_nodes) < 2:
        raise ValueError(
            f"Expected at least 2 LineString/coordinates nodes, found {len(coord_nodes)}"
        )

    coords1 = _parse_coordinates_text(coord_nodes[0].text or "")
    coords2 = _parse_coordinates_text(coord_nodes[1].text or "")

    if not coords1 or not coords2:
        raise ValueError("One or both LineString coordinate lists are empty.")

    # Build individual geometries
    geom1 = LineString(coords1)
    geom2 = LineString(coords2)

    # Merge: Placemark 2 starts at same point as Placemark 1.
    # Append Placemark 2 coords (skip shared first point) to Placemark 1 coords,
    # then close by appending coords1[0].
    shared_start = coords1[0]
    if coords2[0] == shared_start:
        branch_coords = coords2[1:]
    else:
        branch_coords = coords2

    merged_coords = coords1 + branch_coords
    # Close the polygon
    merged_coords.append(merged_coords[0])
    merged_polygon = Polygon(merged_coords)

    records = [
        {"placemark_id": "placemark_1", "geometry": geom1, "n_coords": len(coords1)},
        {"placemark_id": "placemark_2", "geometry": geom2, "n_coords": len(coords2)},
        {
            "placemark_id": "merged_polygon",
            "geometry": merged_polygon,
            "n_coords": len(merged_coords),
        },
    ]

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------


def build_spark_session(cfg: dict):
    """Build a Delta-enabled SparkSession from config.

    Uses configure_spark_with_delta_pip for automatic Delta JAR management.
    """
    import os

    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    # Set JAVA_HOME if not already set
    if not os.environ.get("JAVA_HOME"):
        # Try common locations
        for candidate in [
            "/usr/lib/jvm/java-11-openjdk-amd64",
            "/usr/lib/jvm/java-11-openjdk",
            "/Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home",
            "/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home",
        ]:
            if Path(candidate).exists():
                os.environ["JAVA_HOME"] = candidate
                break

    spark_cfg = cfg.get("spark", {})
    builder = (
        SparkSession.builder.appName(
            spark_cfg.get("app_name", "BrumadinhoBuildingPipeline")
        )
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.executor.memory", spark_cfg.get("executor_memory", "4g"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "4g"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Bronze Delta metadata
# ---------------------------------------------------------------------------


def log_bronze_metadata(
    records: list[dict],
    bronze_dir: str | Path,
    spark,
) -> None:
    """Write bronze metadata records to a Delta table.

    Table location: {bronze_dir}/delta/bronze_metadata
    Schema:
        tile_id STRING, source_path STRING, bronze_path STRING,
        checksum STRING, file_type STRING, image_width INT,
        image_height INT, bands INT, file_size_bytes LONG,
        ingested_at TIMESTAMP
    """
    from pyspark.sql import Row
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    schema = StructType(
        [
            StructField("tile_id", StringType(), False),
            StructField("source_path", StringType(), True),
            StructField("bronze_path", StringType(), True),
            StructField("checksum", StringType(), True),
            StructField("file_type", StringType(), True),
            StructField("image_width", IntegerType(), True),
            StructField("image_height", IntegerType(), True),
            StructField("bands", IntegerType(), True),
            StructField("file_size_bytes", LongType(), True),
            StructField("ingested_at", TimestampType(), True),
        ]
    )

    now = datetime.now(tz=timezone.utc)
    rows = [
        Row(
            tile_id=r["tile_id"],
            source_path=r["source_path"],
            bronze_path=r["bronze_path"],
            checksum=r["checksum"],
            file_type=r["file_type"],
            image_width=int(r["image_width"]),
            image_height=int(r["image_height"]),
            bands=int(r["bands"]),
            file_size_bytes=int(r["file_size_bytes"]),
            ingested_at=now,
        )
        for r in records
    ]

    df = spark.createDataFrame(rows, schema=schema)
    table_path = str(Path(bronze_dir).resolve() / "delta" / "bronze_metadata")
    df.write.format("delta").mode("overwrite").save(table_path)
    print(f"[ingest] Bronze metadata written to {table_path} ({len(records)} records)")
