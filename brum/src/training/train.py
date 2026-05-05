"""Gold layer: training loop with MLflow integration and pyfunc model wrapper."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.factory import build_model
from src.training.losses import CombinedLoss
from src.training.metrics import compute_epoch_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BrumadinhoBuildingDataset(Dataset):
    """Dataset loading (image_chip, mask_chip) pairs from Silver records.

    Each item returns:
        image_tensor: (3, H, W) float32
        mask_tensor: (1, H, W) float32 in [0, 1]
    """

    def __init__(self, records: list[dict], transform=None) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        rec = self.records[idx]
        img = np.array(Image.open(rec["image_chip_path"]).convert("RGB"))
        msk_pil = Image.open(rec["mask_chip_path"]).convert("L")
        msk = (np.array(msk_pil) > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=msk)
            img_tensor = augmented["image"]  # (3, H, W) from ToTensorV2
            msk_tensor = augmented["mask"].unsqueeze(0)  # (1, H, W)
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            msk_tensor = torch.from_numpy(msk).unsqueeze(0)

        return img_tensor, msk_tensor.float()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _fix_seeds(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / max(len(dataloader.dataset), 1)


def _save_sample_predictions(
    model: nn.Module,
    val_records: list[dict],
    epoch: int,
    output_dir: Path,
    device: torch.device,
    n_samples: int = 4,
) -> list[str]:
    """Save sample prediction PNGs for a few validation chips."""
    import cv2
    from PIL import Image

    from src.data.transforms import get_val_transforms

    model.eval()
    transform = get_val_transforms()
    saved_paths = []

    for i, rec in enumerate(val_records[:n_samples]):
        img_arr = np.array(Image.open(rec["image_chip_path"]).convert("RGB"))
        msk_arr = (
            np.array(Image.open(rec["mask_chip_path"]).convert("L")) > 127
        ).astype(np.uint8) * 255

        aug = transform(image=img_arr, mask=msk_arr.astype(np.float32))
        img_t = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(img_t)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred = (prob >= 0.5).astype(np.uint8) * 255

        # Stack: original | true mask | predicted
        h, w = img_arr.shape[:2]
        composite = np.zeros((h, w * 3, 3), dtype=np.uint8)
        composite[:, :w] = img_arr
        composite[:, w : 2 * w] = cv2.cvtColor(msk_arr, cv2.COLOR_GRAY2RGB)
        composite[:, 2 * w :] = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

        save_path = output_dir / f"sample_epoch{epoch:03d}_{i}.png"
        Image.fromarray(composite).save(str(save_path))
        saved_paths.append(str(save_path))

    model.train()
    return saved_paths


# ---------------------------------------------------------------------------
# MLflow pyfunc wrapper
# ---------------------------------------------------------------------------


class BuildingSegmenterPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the building segmentation model.

    Input: pd.DataFrame with column "image_path"
    Output: pd.DataFrame with columns:
        count, all_buildings, safe_buildings, annotated_image_path
    """

    def load_context(self, context) -> None:
        import yaml

        self.cfg = yaml.safe_load(
            open(context.artifacts["config_path"], "r")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            self.cfg,
            checkpoint_path=context.artifacts["checkpoint_path"],
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from src.inference.predict import predict_full_image
        from src.inference.postprocess import run_full_postprocess
        from src.viz.overlay import draw_building_overlays, add_legend, save_annotated_image
        from src.data.transforms import get_val_transforms
        from PIL import Image

        transform = get_val_transforms(self.cfg.get("chip_size", 512))
        results = []

        for _, row in model_input.iterrows():
            img_path = row["image_path"]
            img_array = np.array(Image.open(img_path).convert("RGB"))

            prob_map = predict_full_image(
                img_array, self.model, transform, self.device, self.cfg
            )

            post = run_full_postprocess(prob_map, img_array, self.cfg)

            annotated = draw_building_overlays(
                img_array,
                post["safe_polygons"],
                post["in_impact_polygons"],
                impact_zone_polygon=post.get("impact_polygon"),
            )
            annotated = add_legend(
                annotated,
                post["buildings_safe"],
                post["buildings_in_impact_zone"],
            )

            out_path = str(Path(img_path).with_suffix("")) + "_annotated.png"
            save_annotated_image(annotated, out_path)

            results.append(
                {
                    "count": post["all_buildings"],
                    "all_buildings": post["all_buildings"],
                    "safe_buildings": post["buildings_safe"],
                    "annotated_image_path": out_path,
                }
            )

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(cfg: dict, silver_records: list[dict], output_dir: str | Path) -> str:
    """Full training loop with MLflow tracking.

    Args:
        cfg: Config dict.
        silver_records: List of Silver chip metadata dicts.
        output_dir: Directory for checkpoints and artifacts.

    Returns:
        Path to the best model checkpoint (.pth).
    """
    from src.data.transforms import get_train_transforms, get_val_transforms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg.get("training", {})
    seed = train_cfg.get("seed", 42)
    _fix_seeds(seed)  # Must happen BEFORE DataLoader construction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[train] Using device: {device}")

    # Split records
    train_records = [r for r in silver_records if r["split"] == "train"]
    val_records = [r for r in silver_records if r["split"] == "val"]
    test_records = [r for r in silver_records if r["split"] == "test"]
    logger.info(
        f"[train] Split: {len(train_records)} train, "
        f"{len(val_records)} val, {len(test_records)} test"
    )

    chip_size = cfg.get("chip_size", 512)
    train_ds = BrumadinhoBuildingDataset(
        train_records, transform=get_train_transforms(chip_size)
    )
    val_ds = BrumadinhoBuildingDataset(
        val_records, transform=get_val_transforms(chip_size)
    )

    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    n_epochs = train_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    criterion = CombinedLoss(
        bce_weight=train_cfg.get("bce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
    )

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(
        mlflow_cfg.get("experiment_name", "brumadinho_building_detection")
    )

    best_iou = -1.0
    best_ckpt_path = str(output_dir / "best_model.pth")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "encoder": cfg.get("model", {}).get("encoder_name", "efficientnet-b0"),
                "chip_size": chip_size,
                "batch_size": batch_size,
                "learning_rate": train_cfg.get("learning_rate", 3e-4),
                "epochs": n_epochs,
                "seed": seed,
                "bce_weight": train_cfg.get("bce_weight", 0.5),
                "dice_weight": train_cfg.get("dice_weight", 0.5),
                "n_train": len(train_records),
                "n_val": len(val_records),
            }
        )

        for epoch in range(1, n_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics = compute_epoch_metrics(model, val_loader, device)

            # Compute val loss manually
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    val_loss += criterion(logits, masks).item() * images.size(0)
            val_loss /= max(len(val_loader.dataset), 1)
            model.train()

            scheduler.step()

            # Manual per-epoch logging
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_iou": val_metrics.get("iou", 0.0),
                    "val_precision": val_metrics.get("precision", 0.0),
                    "val_recall": val_metrics.get("recall", 0.0),
                    "val_f1": val_metrics.get("f1", 0.0),
                },
                step=epoch,
            )

            logger.info(
                f"[train] Epoch {epoch}/{n_epochs} — "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_iou={val_metrics.get('iou', 0):.4f}"
            )

            # Save sample predictions every 10 epochs
            if epoch % 10 == 0 and val_records:
                sample_paths = _save_sample_predictions(
                    model, val_records, epoch, samples_dir, device
                )
                for p in sample_paths:
                    mlflow.log_artifact(p, artifact_path="samples")

            # Save best checkpoint
            if val_metrics.get("iou", 0.0) > best_iou:
                best_iou = val_metrics["iou"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_iou": best_iou,
                        "cfg": cfg,
                    },
                    best_ckpt_path,
                )
                logger.info(
                    f"[train] New best checkpoint saved (val_iou={best_iou:.4f})"
                )

        # Log best checkpoint as artifact
        if Path(best_ckpt_path).exists():
            mlflow.log_artifact(best_ckpt_path)

        # Log model as pyfunc with bundled src/
        pyfunc_artifacts = {
            "checkpoint_path": best_ckpt_path,
        }

        # Resolve config path
        import yaml

        config_path = str(output_dir / "config_logged.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
        pyfunc_artifacts["config_path"] = config_path

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=BuildingSegmenterPyfunc(),
            artifacts=pyfunc_artifacts,
            code_path=["src/"],
        )

        # Register and transition to Staging
        model_name = mlflow_cfg.get("model_name", "brumadinho_building_segmenter")
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            reg = mlflow.register_model(model_uri, model_name)
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=reg.version,
                stage="Staging",
            )
            logger.info(
                f"[train] Model registered as {model_name} v{reg.version} → Staging"
            )
        except Exception as exc:
            logger.warning(f"[train] Model registration skipped: {exc}")

        tracking_uri_abs = Path(mlflow_cfg.get("tracking_uri", "mlruns")).resolve()
        logger.info(
            f"[mlflow] Experiment logged. To view results:\n"
            f"         mlflow ui --backend-store-uri {tracking_uri_abs}\n"
            f"         → http://127.0.0.1:5000"
        )

    return best_ckpt_path
