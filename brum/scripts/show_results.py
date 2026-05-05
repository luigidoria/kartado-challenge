#!/usr/bin/env python3
"""Show MLflow training results without the UI.

Reads the latest run from the configured experiment and prints:
  - All logged metrics per epoch (table)
  - Model registry info (version, stage)
  - Artifact paths

Usage (from brum/):
    python scripts/show_results.py
    python scripts/show_results.py --experiment brumadinho_building_detection
    python scripts/show_results.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BRUM_ROOT = Path(__file__).resolve().parent.parent
if str(_BRUM_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRUM_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print MLflow training results (no browser needed)"
    )
    parser.add_argument(
        "--experiment",
        default="brumadinho_building_detection",
        help="MLflow experiment name (default: brumadinho_building_detection)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run ID to inspect (default: latest run in experiment)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: auto-detected from config.yaml)",
    )
    return parser.parse_args()


def _resolve_tracking_uri(args: argparse.Namespace) -> str:
    if args.tracking_uri:
        return args.tracking_uri

    cfg_path = _BRUM_ROOT / "conf" / "config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        uri = cfg.get("mlflow", {}).get("tracking_uri", "mlruns")
        if not Path(uri).is_absolute():
            uri = str((_BRUM_ROOT / uri).resolve())
        return uri

    return str((_BRUM_ROOT / "mlruns").resolve())


def _get_latest_run(client, experiment_name: str):
    import mlflow
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"ERROR: Experiment '{experiment_name}' not found.", file=sys.stderr)
        print(
            f"       Available experiments:",
            file=sys.stderr,
        )
        for e in client.search_experiments():
            print(f"         {e.name}", file=sys.stderr)
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"ERROR: No runs found in experiment '{experiment_name}'.", file=sys.stderr)
        sys.exit(1)
    return runs[0]


def _print_metrics_table(client, run_id: str) -> None:
    # Fetch all metric history
    metric_keys = [
        "train_loss", "val_loss",
        "val_iou", "val_precision", "val_recall", "val_f1",
    ]

    histories: dict[str, dict[int, float]] = {}
    for key in metric_keys:
        try:
            history = client.get_metric_history(run_id, key)
            histories[key] = {m.step: m.value for m in history}
        except Exception:
            histories[key] = {}

    epochs = sorted(
        set(step for h in histories.values() for step in h.keys())
    )
    if not epochs:
        print("  (no per-epoch metrics logged)")
        return

    # Header
    col_w = 12
    headers = ["Epoch"] + metric_keys
    header_line = f"{'Epoch':>6}" + "".join(f"{h:>{col_w}}" for h in metric_keys)
    sep = "-" * len(header_line)
    print(header_line)
    print(sep)

    for epoch in epochs:
        row = f"{epoch:>6}"
        for key in metric_keys:
            val = histories[key].get(epoch)
            row += f"{val:>{col_w}.4f}" if val is not None else f"{'—':>{col_w}}"
        print(row)

    print(sep)

    # Best row
    best_iou_epoch = max(
        epochs,
        key=lambda e: histories["val_iou"].get(e, -1),
        default=None,
    )
    if best_iou_epoch is not None:
        best_iou = histories["val_iou"].get(best_iou_epoch, 0)
        print(f"\n  Best val_iou: {best_iou:.4f}  (epoch {best_iou_epoch})")


def _print_model_registry(client, model_name: str) -> None:
    print(f"\n{'='*60}")
    print("  MODEL REGISTRY")
    print(f"{'='*60}")
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"  (no registered versions for '{model_name}')")
            return
        for v in versions:
            print(f"  {model_name} v{v.version}")
            print(f"    Stage   : {v.current_stage}")
            print(f"    Status  : {v.status}")
            print(f"    Run ID  : {v.run_id}")
            print(f"    Source  : {v.source}")
    except Exception as exc:
        print(f"  (model registry unavailable: {exc})")


def _print_artifacts(client, run_id: str) -> None:
    print(f"\n{'='*60}")
    print("  ARTIFACTS")
    print(f"{'='*60}")
    try:
        artifacts = client.list_artifacts(run_id)
        if not artifacts:
            print("  (no artifacts logged)")
            return
        for a in artifacts:
            print(f"  {'[DIR]' if a.is_dir else '     '} {a.path}")
    except Exception as exc:
        print(f"  (artifact listing failed: {exc})")


def main() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    args = parse_args()
    tracking_uri = _resolve_tracking_uri(args)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    print(f"\n{'='*60}")
    print("  BRUMADINHO — MLFLOW RESULTS")
    print(f"{'='*60}")
    print(f"  Tracking URI : {tracking_uri}")
    print(f"  Experiment   : {args.experiment}")
    print(f"\n  Tip — view in browser:")
    print(f"    mlflow ui --backend-store-uri {tracking_uri}")
    print(f"    → http://127.0.0.1:5000")

    if args.run_id:
        run = client.get_run(args.run_id)
    else:
        run = _get_latest_run(client, args.experiment)

    info = run.info
    params = run.data.params

    print(f"\n{'='*60}")
    print("  RUN INFO")
    print(f"{'='*60}")
    print(f"  Run ID    : {info.run_id}")
    print(f"  Status    : {info.status}")
    import datetime
    start_ms = info.start_time
    end_ms = info.end_time
    if start_ms:
        start_dt = datetime.datetime.fromtimestamp(start_ms / 1000)
        print(f"  Started   : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    if end_ms:
        end_dt = datetime.datetime.fromtimestamp(end_ms / 1000)
        duration_s = (end_ms - start_ms) / 1000
        print(f"  Ended     : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration  : {duration_s:.0f}s")

    print(f"\n{'='*60}")
    print("  PARAMETERS")
    print(f"{'='*60}")
    for k, v in sorted(params.items()):
        print(f"  {k:<25} {v}")

    print(f"\n{'='*60}")
    print("  METRICS (per epoch)")
    print(f"{'='*60}")
    _print_metrics_table(client, info.run_id)

    # Model registry
    import yaml
    cfg_path = _BRUM_ROOT / "conf" / "config.yaml"
    model_name = "brumadinho_building_segmenter"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg.get("mlflow", {}).get("model_name", model_name)

    _print_model_registry(client, model_name)
    _print_artifacts(client, info.run_id)
    print()


if __name__ == "__main__":
    main()
