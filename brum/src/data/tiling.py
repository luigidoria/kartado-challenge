"""Silver layer: image and mask tiling with spatial block split."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------


def compute_chip_grid(
    image_width: int,
    image_height: int,
    chip_size: int = 512,
    stride: int = 384,
) -> list[tuple[int, int, int, int]]:
    """Return list of (x_start, y_start, x_end, y_end) chip bounding boxes.

    The last chip in each row/column is anchored to the image edge so no pixels
    are dropped, even if it results in extra overlap.

    For a typical 2075×750 image with chip_size=512, stride=384:
        ~6 columns × 2 rows = 12 chips per image.
    """
    xs = list(range(0, image_width - chip_size + 1, stride))
    if not xs or xs[-1] + chip_size < image_width:
        xs.append(image_width - chip_size)
    xs = sorted(set(max(0, x) for x in xs))

    ys = list(range(0, image_height - chip_size + 1, stride))
    if not ys or ys[-1] + chip_size < image_height:
        ys.append(image_height - chip_size)
    ys = sorted(set(max(0, y) for y in ys))

    grid = []
    for y_start in ys:
        for x_start in xs:
            x_end = min(x_start + chip_size, image_width)
            y_end = min(y_start + chip_size, image_height)
            grid.append((x_start, y_start, x_end, y_end))

    return grid


# ---------------------------------------------------------------------------
# Chip extraction
# ---------------------------------------------------------------------------


def chip_image(
    img_array: np.ndarray,
    bounds: tuple[int, int, int, int],
    drop_alpha: bool = True,
) -> np.ndarray:
    """Extract a chip from *img_array* at *bounds* = (x_start, y_start, x_end, y_end).

    If the array is RGBA and drop_alpha=True, returns (H, W, 3).
    """
    x0, y0, x1, y1 = bounds
    chip = img_array[y0:y1, x0:x1]
    if drop_alpha and chip.ndim == 3 and chip.shape[2] == 4:
        chip = chip[:, :, :3]
    return chip


def chip_mask(mask_array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a mask chip at *bounds* = (x_start, y_start, x_end, y_end).

    Returns (H, W) uint8.
    """
    x0, y0, x1, y1 = bounds
    return mask_array[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Spatial split
# ---------------------------------------------------------------------------


def assign_split(
    chip_col: int,
    total_cols: int,
    train_frac: float = 0.667,
    val_frac: float = 0.167,
) -> str:
    """Assign a spatial split based on chip column index.

    Leftmost train_frac of columns → train
    Next val_frac → val
    Remainder → test

    This prevents geographic data leakage caused by spatial autocorrelation.
    """
    train_end = int(total_cols * train_frac)
    val_end = int(total_cols * (train_frac + val_frac))
    if chip_col < train_end:
        return "train"
    elif chip_col < val_end:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_chip(chip_array: np.ndarray, output_path: str | Path) -> None:
    """Save a chip array as PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(chip_array)
    img.save(str(output_path))


# ---------------------------------------------------------------------------
# Main tiling orchestrator
# ---------------------------------------------------------------------------


def tile_all_images(
    image_paths: list[Path],
    mask_arrays: list[np.ndarray],
    silver_dir: str | Path,
    affines: list[dict],
    cfg: dict,
) -> list[dict]:
    """Tile all images and masks into chips and save to silver directory.

    Args:
        image_paths: List of paths to source images.
        mask_arrays: Corresponding binary building masks (H, W) uint8.
        silver_dir: Root silver directory.
        affines: Per-image affine dicts (used for metadata, not tiling).
        cfg: Config dict with chip_size, stride, training.train_frac/val_frac.

    Returns:
        List of dicts with keys:
            tile_id, source_image, image_chip_path, mask_chip_path,
            split, chip_col, chip_row, x_start, y_start, x_end, y_end
    """
    import numpy as np
    from PIL import Image as PILImage

    silver_dir = Path(silver_dir)
    img_chips_dir = silver_dir / "chips" / "images"
    msk_chips_dir = silver_dir / "chips" / "masks"
    img_chips_dir.mkdir(parents=True, exist_ok=True)
    msk_chips_dir.mkdir(parents=True, exist_ok=True)

    chip_size = cfg.get("chip_size", 512)
    stride = cfg.get("stride", 384)
    train_cfg = cfg.get("training", {})
    train_frac = train_cfg.get("train_frac", 0.667)
    val_frac = train_cfg.get("val_frac", 0.167)

    records: list[dict] = []

    for img_idx, (img_path, mask_array) in enumerate(zip(image_paths, mask_arrays)):
        img_path = Path(img_path)
        with PILImage.open(img_path) as pil_img:
            img_array = np.array(pil_img)

        h, w = img_array.shape[:2]
        grid = compute_chip_grid(w, h, chip_size, stride)

        # Determine number of distinct columns for split assignment
        xs_unique = sorted(set(x0 for x0, _, _, _ in grid))
        total_cols = len(xs_unique)
        col_index_map = {x0: i for i, x0 in enumerate(xs_unique)}
        ys_unique = sorted(set(y0 for _, y0, _, _ in grid))
        row_index_map = {y0: i for i, y0 in enumerate(ys_unique)}

        img_stem = img_path.stem

        for x0, y0, x1, y1 in grid:
            chip_col = col_index_map[x0]
            chip_row = row_index_map[y0]
            split = assign_split(chip_col, total_cols, train_frac, val_frac)

            tile_id = f"{img_stem}_col{chip_col:02d}_row{chip_row:02d}"

            img_chip = chip_image(img_array, (x0, y0, x1, y1), drop_alpha=True)
            msk_chip = chip_mask(mask_array, (x0, y0, x1, y1))

            img_chip_path = img_chips_dir / f"{tile_id}.png"
            msk_chip_path = msk_chips_dir / f"{tile_id}.png"

            save_chip(img_chip, img_chip_path)
            # Save mask as grayscale PNG
            PILImage.fromarray(msk_chip).save(str(msk_chip_path))

            records.append(
                {
                    "tile_id": tile_id,
                    "source_image": str(img_path.resolve()),
                    "image_chip_path": str(img_chip_path.resolve()),
                    "mask_chip_path": str(msk_chip_path.resolve()),
                    "split": split,
                    "chip_col": chip_col,
                    "chip_row": chip_row,
                    "x_start": x0,
                    "y_start": y0,
                    "x_end": x1,
                    "y_end": y1,
                }
            )

    logger.info(
        f"[tiling] Created {len(records)} chips from {len(image_paths)} images"
    )
    return records


def log_silver_metadata(
    records: list[dict],
    silver_dir: str | Path,
    spark,
) -> None:
    """Write silver chip metadata to a Delta table.

    Table location: {silver_dir}/delta/silver_metadata
    """
    from pyspark.sql import Row
    from pyspark.sql.types import (
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("tile_id", StringType(), False),
            StructField("source_image", StringType(), True),
            StructField("image_chip_path", StringType(), True),
            StructField("mask_chip_path", StringType(), True),
            StructField("split", StringType(), True),
            StructField("chip_col", IntegerType(), True),
            StructField("chip_row", IntegerType(), True),
            StructField("x_start", IntegerType(), True),
            StructField("y_start", IntegerType(), True),
            StructField("x_end", IntegerType(), True),
            StructField("y_end", IntegerType(), True),
        ]
    )

    rows = [
        Row(
            tile_id=r["tile_id"],
            source_image=r["source_image"],
            image_chip_path=r["image_chip_path"],
            mask_chip_path=r["mask_chip_path"],
            split=r["split"],
            chip_col=int(r["chip_col"]),
            chip_row=int(r["chip_row"]),
            x_start=int(r["x_start"]),
            y_start=int(r["y_start"]),
            x_end=int(r["x_end"]),
            y_end=int(r["y_end"]),
        )
        for r in records
    ]

    df = spark.createDataFrame(rows, schema=schema)
    table_path = str(Path(silver_dir).resolve() / "delta" / "silver_metadata")
    df.write.format("delta").mode("overwrite").save(table_path)
    logger.info(
        f"[tiling] Silver metadata written to {table_path} ({len(records)} records)"
    )
