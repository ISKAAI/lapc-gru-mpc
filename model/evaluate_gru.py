from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.gru import DisturbanceGRU
from model.train_gru import metrics, predict, speed_bin_metrics


def features_for_checkpoint(data, checkpoint, split):
    """Convert dataset-normalized features to the checkpoint's normalization."""
    data_names = data["feature_names"].tolist()
    checkpoint_names = checkpoint["feature_names"]
    if data_names != checkpoint_names:
        raise ValueError(
            f"Feature mismatch: dataset={data_names}, checkpoint={checkpoint_names}"
        )
    if int(data["window"]) != int(checkpoint["window"]):
        raise ValueError(
            f"Window mismatch: dataset={int(data['window'])}, "
            f"checkpoint={int(checkpoint['window'])}"
        )

    data_mean = np.asarray(data["mean"], dtype=np.float32)
    data_std = np.asarray(data["std"], dtype=np.float32)
    checkpoint_mean = checkpoint["feature_mean"].cpu().numpy()
    checkpoint_std = checkpoint["feature_std"].cpu().numpy()
    raw = np.asarray(data[f"X{split}"], dtype=np.float32) * data_std + data_mean
    return (raw - checkpoint_mean) / checkpoint_std


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen GRU checkpoint once.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="stress_final")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    data = np.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = DisturbanceGRU(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    X = features_for_checkpoint(data, checkpoint, args.split)
    Y = data[f"Y{args.split}"]
    pred = predict(model, X, device, args.batch_size)
    report = {
        "split": args.split,
        "samples": int(len(X)),
        "checkpoint": str(args.checkpoint),
        **metrics(pred, Y),
        "speed_bins": speed_bin_metrics(pred, Y, data[f"speed_{args.split}"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{args.split}: MSE={report['mean_mse']:.8f} "
        f"persist={report['mean_persist_mse']:.8f} "
        f"improve={100 * report['mean_improvement_vs_persist']:+.2f}% "
        f"R2={report['mean_r2']:+.4f}"
    )
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
