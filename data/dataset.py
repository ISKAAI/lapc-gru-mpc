from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import VehicleConfig
from data import extract, features
from data.channels import DT
from model.bicycle_model import BicycleModel


DATA_DIR = Path("/Users/kai/project/train")
HIGHWAY_LIST_DIR = PROJECT_ROOT / "processed" / "highway_data"
BASE_FEATURE_NAMES = ("d_e1dot", "e1", "e1dot", "e2", "e2dot", "front_steer")
FEATURE_SETS = {
    "base": BASE_FEATURE_NAMES,
    "vx": BASE_FEATURE_NAMES + ("vx",),
    "vx_kappa": BASE_FEATURE_NAMES + ("vx", "kappa"),
    "full": BASE_FEATURE_NAMES + ("vx", "kappa", "front_steer_rate"),
}


def make_windows(F, target, W, Np):
    """Convert a generic feature matrix into history/future sliding windows."""
    X, Y = [], []
    M = len(F)
    for k in range(W, M - Np):
        X.append(F[k - W:k])
        Y.append(target[k:k + Np] - target[k - 1])
    return np.asarray(X), np.asarray(Y)


def _segment_features(seg, vc, model, feature_set="base"):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    feat = features.compute_feature(seg, vc)
    disturbance = features.compute_disturbance(feat, model)
    disturbance_smooth = features.smooth_disturbance(disturbance)
    x, steer = feat["x"], feat["U"]
    columns = [
        disturbance_smooth[:, 1],
        x[:-1, 0],
        x[:-1, 1],
        x[:-1, 2],
        x[:-1, 3],
        steer[:-1],
    ]
    if feature_set in {"vx", "vx_kappa", "full"}:
        columns.append(feat["vx"][:-1])
    if feature_set in {"vx_kappa", "full"}:
        columns.append(feat["kappa"][:-1])
    if feature_set == "full":
        # Backward difference is causal: each value only uses current/past steering.
        steer_rate = np.diff(steer, prepend=steer[0]) / DT
        columns.append(steer_rate[:-1])
    F = np.column_stack(columns)
    return feat, disturbance_smooth, F


def segment_to_windows(seg, vc, model, W, Np, feature_set="base"):
    """One valid segment -> sliding-window samples (legacy-compatible API)."""
    _, disturbance_smooth, F = _segment_features(seg, vc, model, feature_set)
    return make_windows(F, disturbance_smooth[:, 1], W, Np)


def segment_to_windows_with_speed(seg, vc, model, W, Np, feature_set="base"):
    """As ``segment_to_windows``, plus mean historical speed for every sample."""
    feat, disturbance_smooth, F = _segment_features(seg, vc, model, feature_set)
    X, Y = make_windows(F, disturbance_smooth[:, 1], W, Np)
    M = len(F)
    speed = np.asarray(
        [np.mean(feat["vx"][k - W:k]) for k in range(W, M - Np)], dtype=float
    )
    return X, Y, speed


def _read_path_list(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing classified data list: {path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _process_split(paths, vc, model, W, Np, file_to_id, feature_set):
    xs, ys, speeds, file_ids, segment_ids = [], [], [], [], []
    segment_count = 0
    for path in paths:
        for seg in extract.extract_file(str(path)):
            X, Y, speed = segment_to_windows_with_speed(
                seg, vc, model, W, Np, feature_set=feature_set
            )
            if len(X) == 0:
                continue
            xs.append(X)
            ys.append(Y)
            speeds.append(speed)
            file_ids.append(np.full(len(X), file_to_id[str(path)], dtype=np.int32))
            segment_ids.append(np.full(len(X), segment_count, dtype=np.int32))
            segment_count += 1
    if not xs:
        raise ValueError("No windows were produced for a classified split")
    return {
        "X": np.concatenate(xs),
        "Y": np.concatenate(ys),
        "speed": np.concatenate(speeds),
        "file_id": np.concatenate(file_ids),
        "segment_id": np.concatenate(segment_ids),
        "segments": segment_count,
    }


def build_highway_dataset(
    W=20,
    Np=25,
    list_dir=HIGHWAY_LIST_DIR,
    out=None,
    feature_set="vx",
    include_legacy=False,
):
    """Build grouped normal/stress sets, optionally adding legacy logs to train."""
    list_dir = Path(list_dir)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set == "vx" and W == 20 and Np == 10:
        default_name = "combined_dataset.npz" if include_legacy else "highway_dataset.npz"
    else:
        default_name = f"highway_dataset_{feature_set}_w{W}_h{Np}.npz"
    out = Path(out) if out is not None else PROJECT_ROOT / "processed" / default_name
    feature_names = np.asarray(FEATURE_SETS[feature_set])
    split_lists = {
        "tr": _read_path_list(list_dir / "train_normal.txt"),
        "va": _read_path_list(list_dir / "val_normal.txt"),
        "te": _read_path_list(list_dir / "test_normal.txt"),
        "stress_dev": _read_path_list(list_dir / "stress_dev.txt"),
        "stress_final": _read_path_list(list_dir / "stress_final.txt"),
    }
    legacy_files = []
    if include_legacy:
        for category in ["lcc", "alc"]:
            legacy_files.extend(sorted((DATA_DIR / category).glob("*.MF4")))
        split_lists["tr"].extend(legacy_files)

    all_files = sorted({str(path) for paths in split_lists.values() for path in paths})
    file_to_id = {path: index for index, path in enumerate(all_files)}
    legacy_paths = {str(path) for path in legacy_files}

    vc = VehicleConfig()
    model = BicycleModel(VehicleConfig(Ts=0.05), 10.0)
    raw = {
        split: _process_split(paths, vc, model, W, Np, file_to_id, feature_set)
        for split, paths in split_lists.items()
    }

    mean = raw["tr"]["X"].reshape(-1, len(feature_names)).mean(axis=0)
    std = raw["tr"]["X"].reshape(-1, len(feature_names)).std(axis=0)
    std[std < 1e-8] = 1.0

    payload = {
        "mean": mean,
        "std": std,
        "feature_names": feature_names,
        "feature_set": np.asarray(feature_set),
        "files": np.asarray(all_files),
        "file_domain": np.asarray(
            ["legacy_low_speed" if path in legacy_paths else "new_recording" for path in all_files]
        ),
        "include_legacy": np.asarray(include_legacy),
        "window": np.asarray(W, dtype=np.int32),
        "horizon": np.asarray(Np, dtype=np.int32),
    }
    for split, values in raw.items():
        payload[f"X{split}"] = ((values["X"] - mean) / std).astype(np.float32)
        payload[f"Y{split}"] = values["Y"].astype(np.float32)
        payload[f"speed_{split}"] = values["speed"].astype(np.float32)
        payload[f"file_id_{split}"] = values["file_id"]
        payload[f"segment_id_{split}"] = values["segment_id"]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    for split in ["tr", "va", "te", "stress_dev", "stress_final"]:
        values = raw[split]
        print(
            f"{split:6s}: files={len(split_lists[split]):2d} segments={values['segments']:2d} "
            f"X{payload[f'X{split}'].shape} Y{payload[f'Y{split}'].shape} "
            f"speed=[{values['speed'].min():.2f}, {values['speed'].max():.2f}] m/s"
        )
    print(f"feature mean={mean.round(5)}")
    print(f"feature std ={std.round(5)}")
    print(f"saved -> {out}")
    return payload


def build_dataset(W=20, Np=5, val_ratio=0.2, out=None):
    """Legacy low-speed dataset builder retained for existing experiments."""
    out = Path(out) if out is not None else PROJECT_ROOT / "processed" / "dataset.npz"
    vc = VehicleConfig()
    model = BicycleModel(VehicleConfig(Ts=0.05), 10.0)
    seg_data = []
    for category in ["lcc", "alc"]:
        for path in sorted(glob.glob(str(DATA_DIR / category / "*.MF4"))):
            for seg in extract.extract_file(path):
                X, Y = segment_to_windows(seg, vc, model, W, Np)
                if len(X):
                    seg_data.append((X, Y))
    if len(seg_data) < 2:
        raise ValueError("Not enough valid segments to build train/validation datasets")

    rng = np.random.default_rng(0)
    rng.shuffle(seg_data)
    n_val = max(1, int(len(seg_data) * val_ratio))
    val, train = seg_data[:n_val], seg_data[n_val:]
    Xtr = np.concatenate([item[0] for item in train])
    Ytr = np.concatenate([item[1] for item in train])
    Xva = np.concatenate([item[0] for item in val])
    Yva = np.concatenate([item[1] for item in val])
    mean = Xtr.reshape(-1, len(BASE_FEATURE_NAMES)).mean(axis=0)
    std = Xtr.reshape(-1, len(BASE_FEATURE_NAMES)).std(axis=0)
    std[std < 1e-8] = 1.0
    Xtr, Xva = (Xtr - mean) / std, (Xva - mean) / std
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, mean=mean, std=std)
    print(f"train X{Xtr.shape} Y{Ytr.shape} | val X{Xva.shape} Y{Yva.shape}")
    print(f"saved -> {out}")
    return Xtr, Ytr, Xva, Yva


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["highway", "legacy"], default="highway")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="vx")
    parser.add_argument("--list-dir", type=Path, default=HIGHWAY_LIST_DIR)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Add raw MF4 logs from /train/lcc and /train/alc to the training split.",
    )
    args = parser.parse_args()
    if args.mode == "highway":
        build_highway_dataset(
            args.window,
            args.horizon,
            args.list_dir,
            args.out,
            args.feature_set,
            args.include_legacy,
        )
    else:
        build_dataset(args.window, args.horizon, out=args.out)


if __name__ == "__main__":
    main()
