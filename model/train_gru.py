from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.gru import DisturbanceGRU


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(X, Y, batch_size, shuffle=False, generator=None):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0
    )


def predict(model, X, device, batch_size):
    loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def metrics(pred, target):
    squared_error = np.square(pred - target)
    mse = squared_error.mean(axis=0)
    persist_mse = np.square(target).mean(axis=0)
    improvement = np.divide(
        persist_mse - mse,
        persist_mse,
        out=np.zeros_like(mse),
        where=persist_mse > 1e-12,
    )
    target_mean = target.mean(axis=0)
    ss_res = squared_error.sum(axis=0)
    ss_tot = np.square(target - target_mean).sum(axis=0)
    r2 = 1.0 - np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan), where=ss_tot > 1e-12)
    return {
        "mse_per_step": mse.tolist(),
        "rmse_per_step": np.sqrt(mse).tolist(),
        "persist_mse_per_step": persist_mse.tolist(),
        "improvement_vs_persist_per_step": improvement.tolist(),
        "r2_per_step": r2.tolist(),
        "mean_mse": float(mse.mean()),
        "mean_persist_mse": float(persist_mse.mean()),
        "mean_improvement_vs_persist": float(1.0 - mse.mean() / persist_mse.mean()),
        "mean_r2": float(np.nanmean(r2)),
    }


def speed_bin_metrics(pred, target, speed):
    result = {}
    for lower, upper in [(0, 15), (15, 20), (20, 25), (25, 30), (30, float("inf"))]:
        mask = (speed >= lower) & (speed < upper)
        label = f"{lower}-{upper if np.isfinite(upper) else 'inf'}_mps"
        if mask.any():
            result[label] = {"samples": int(mask.sum()), **metrics(pred[mask], target[mask])}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "processed" / "combined_gru_h25")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "processed" / "highway_gru")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"device={device}")

    data = np.load(args.data)
    required = [
        "Xtr", "Ytr", "Xva", "Yva", "Xte", "Yte", "Xstress_dev", "Ystress_dev"
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset is missing highway splits: {missing}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        data["Xtr"], data["Ytr"], args.batch_size, shuffle=True, generator=generator
    )
    val_loader = make_loader(data["Xva"], data["Yva"], args.batch_size)
    model = DisturbanceGRU(
        n_feat=data["Xtr"].shape[-1], hidden=args.hidden, horizon=data["Ytr"].shape[-1]
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = {"train": [], "val": []}
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
            total_samples += len(xb)
        train_loss = total_loss / total_samples

        model.eval()
        val_total = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_total += loss.item() * len(xb)
                val_samples += len(xb)
        val_loss = val_total / val_samples
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"epoch={epoch + 1:03d} train={train_loss:.8f} val={val_loss:.8f}")

        if val_loss < best_loss - 1e-10:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": {
            "n_feat": int(data["Xtr"].shape[-1]),
            "hidden": args.hidden,
            "horizon": int(data["Ytr"].shape[-1]),
        },
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_loss,
        "feature_mean": torch.from_numpy(data["mean"]).float(),
        "feature_std": torch.from_numpy(data["std"]).float(),
        "feature_names": data["feature_names"].tolist(),
        "window": int(data["window"]),
        "data_path": str(args.data),
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")

    report = {
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_loss,
        "history": history,
        "splits": {},
    }
    predictions = {}
    for split in ["tr", "va", "te", "stress_dev"]:
        X, Y = data[f"X{split}"], data[f"Y{split}"]
        pred = predict(model, X, device, args.batch_size)
        predictions[f"pred_{split}"] = pred
        predictions[f"target_{split}"] = Y
        split_report = {"samples": int(len(X)), **metrics(pred, Y)}
        split_report["speed_bins"] = speed_bin_metrics(pred, Y, data[f"speed_{split}"])
        report["splits"][split] = split_report
        print(
            f"{split:6s}: MSE={split_report['mean_mse']:.8f} "
            f"persist={split_report['mean_persist_mse']:.8f} "
            f"improve={100 * split_report['mean_improvement_vs_persist']:+.2f}% "
            f"R2={split_report['mean_r2']:+.4f}"
        )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.axvline(best_epoch, color="C2", linestyle="--", label=f"best={best_epoch + 1}")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "loss.png", dpi=140)
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
