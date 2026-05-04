# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Evaluate
# MAGIC
# MAGIC Loads the best checkpoint, evaluates on the test split,
# MAGIC displays IoU / Precision / Recall / F1, shows 2 prediction/truth comparisons,
# MAGIC and logs results to a Gold Delta table.

# COMMAND ----------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

with open("conf/config.yaml") as f:
    cfg = yaml.safe_load(f)

silver_dir = Path(cfg["paths"]["silver_dir"])
gold_dir = Path(cfg["paths"]["gold_dir"])
checkpoint_path = Path(cfg["paths"]["checkpoint_path"])

# COMMAND ----------
# MAGIC %md ## 1. Load test split records

from src.data.ingest import build_spark_session

spark = build_spark_session(cfg)
silver_delta_path = str((silver_dir / "delta" / "silver_metadata").resolve())
silver_df = spark.read.format("delta").load(silver_delta_path)
test_records = [r.asDict() for r in silver_df.filter("split = 'test'").collect()]
print(f"Test chips: {len(test_records)}")

# COMMAND ----------
# MAGIC %md ## 2. Load model and compute metrics

import torch
from torch.utils.data import DataLoader

from src.models.factory import build_model
from src.training.train import BrumadinhoBuildingDataset
from src.data.transforms import get_val_transforms
from src.training.metrics import compute_epoch_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg, checkpoint_path=str(checkpoint_path))
model.to(device)
model.eval()

transform = get_val_transforms(cfg.get("chip_size", 512))
test_ds = BrumadinhoBuildingDataset(test_records, transform=transform)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

metrics = compute_epoch_metrics(model, test_loader, device)
print("\n=== Test Set Metrics ===")
for k, v in metrics.items():
    print(f"  {k:12s}: {v:.4f}")

# COMMAND ----------
# MAGIC %md ## 3. Visualize 2 sample predictions

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

for i, rec in enumerate(test_records[:2]):
    img = np.array(Image.open(rec["image_chip_path"]).convert("RGB"))
    true_mask = (np.array(Image.open(rec["mask_chip_path"]).convert("L")) > 127).astype(np.uint8)

    aug = transform(image=img)
    img_t = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(img_t)
        prob = torch.sigmoid(logit).squeeze().cpu().numpy()
        pred_mask = (prob >= 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input chip")
    axes[1].imshow(true_mask, cmap="gray")
    axes[1].set_title("Ground truth mask")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Predicted mask")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Sample {i+1}: {Path(rec['tile_id']).name}")
    plt.tight_layout()
    plt.savefig(str(gold_dir / f"eval_sample_{i}.png"), dpi=150, bbox_inches="tight")
    plt.show()

# COMMAND ----------
# MAGIC %md ## 4. Log metrics to Gold Delta table

from pyspark.sql import Row
from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType
from datetime import datetime, timezone

schema = StructType([
    StructField("metric_name", StringType(), False),
    StructField("metric_value", DoubleType(), True),
    StructField("evaluated_at", TimestampType(), True),
    StructField("checkpoint_path", StringType(), True),
])
now = datetime.now(tz=timezone.utc)
rows = [
    Row(
        metric_name=k,
        metric_value=float(v),
        evaluated_at=now,
        checkpoint_path=str(checkpoint_path),
    )
    for k, v in metrics.items()
]
metrics_df = spark.createDataFrame(rows, schema=schema)
gold_delta_path = str((gold_dir / "delta" / "evaluation_metrics").resolve())
metrics_df.write.format("delta").mode("append").save(gold_delta_path)
print(f"\nMetrics logged to {gold_delta_path}")
