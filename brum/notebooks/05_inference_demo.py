# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Inference Demo
# MAGIC
# MAGIC Loads the registered model from MLflow Model Registry (Staging stage),
# MAGIC runs it on `img0.png`, and displays the annotated result.

# COMMAND ----------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

with open("conf/config.yaml") as f:
    cfg = yaml.safe_load(f)

raw_dir = Path(cfg["paths"]["raw_dir"])
gold_dir = Path(cfg["paths"]["gold_dir"])
mlflow_cfg = cfg["mlflow"]

import mlflow

mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

# COMMAND ----------
# MAGIC %md ## 1. Load model from MLflow Model Registry (Staging)

model_name = mlflow_cfg["model_name"]
model_uri = f"models:/{model_name}/Staging"
print(f"Loading model from: {model_uri}")

loaded_model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully.")

# COMMAND ----------
# MAGIC %md ## 2. Run inference on img0.png

import pandas as pd

img_path = str(raw_dir / "img0.png")
input_df = pd.DataFrame({"image_path": [img_path]})

result_df = loaded_model.predict(input_df)
print(result_df.to_string())

# COMMAND ----------
# MAGIC %md ## 3. Display annotated result

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

annotated_path = result_df.iloc[0]["annotated_image_path"]
annotated = np.array(Image.open(annotated_path))

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
axes[0].imshow(np.array(Image.open(img_path)))
axes[0].set_title("Original: img0.png")
axes[0].axis("off")

axes[1].imshow(annotated)
axes[1].set_title(
    f"Detected: {result_df.iloc[0]['all_buildings']} buildings  "
    f"(red={result_df.iloc[0]['all_buildings'] - result_df.iloc[0]['safe_buildings']}, "
    f"green={result_df.iloc[0]['safe_buildings']})"
)
axes[1].axis("off")
plt.tight_layout()
plt.savefig(str(gold_dir / "inference_demo.png"), dpi=150, bbox_inches="tight")
plt.show()

print(f"\nFull result:")
print(f"  Total buildings: {result_df.iloc[0]['all_buildings']}")
print(f"  In impact zone:  {result_df.iloc[0]['all_buildings'] - result_df.iloc[0]['safe_buildings']}")
print(f"  Safe:            {result_df.iloc[0]['safe_buildings']}")
print(f"  Annotated image: {annotated_path}")
