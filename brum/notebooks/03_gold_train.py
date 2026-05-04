# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold: Model Training
# MAGIC
# MAGIC Loads Silver chip records, trains U-Net + EfficientNet-B0 with combined
# MAGIC BCE + Dice loss, logs metrics to MLflow, saves best checkpoint.

# COMMAND ----------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

with open("conf/config.yaml") as f:
    cfg = yaml.safe_load(f)

silver_dir = Path(cfg["paths"]["silver_dir"])
gold_dir = Path(cfg["paths"]["gold_dir"])
gold_dir.mkdir(parents=True, exist_ok=True)

# COMMAND ----------
# MAGIC %md ## 1. Load Silver records from Delta table

from src.data.ingest import build_spark_session

spark = build_spark_session(cfg)
silver_delta_path = str((silver_dir / "delta" / "silver_metadata").resolve())
silver_df = spark.read.format("delta").load(silver_delta_path)
silver_records = [row.asDict() for row in silver_df.collect()]
print(f"Loaded {len(silver_records)} chip records from Silver Delta")

from collections import Counter
split_counts = Counter(r["split"] for r in silver_records)
for split, count in sorted(split_counts.items()):
    print(f"  {split:5s}: {count} chips")

# COMMAND ----------
# MAGIC %md ## 2. Train model

import mlflow

from src.training.train import train

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

best_checkpoint = train(cfg, silver_records, gold_dir)
print(f"\nBest checkpoint saved at: {best_checkpoint}")

# COMMAND ----------
# MAGIC %md ## 3. Review MLflow experiment

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_iou DESC"],
    max_results=5,
)
print(f"\nTop-5 runs by val_iou:")
for run in runs:
    m = run.data.metrics
    print(f"  run_id={run.info.run_id[:8]}  "
          f"val_iou={m.get('val_iou', 0):.4f}  "
          f"val_f1={m.get('val_f1', 0):.4f}  "
          f"epochs={int(m.get('training_step', 0))}")
