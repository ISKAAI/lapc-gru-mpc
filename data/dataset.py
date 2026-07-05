import numpy as np
import sys, os, glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import extract
import features
from config import VehicleConfig
from model.bicycle_model import BicycleModel

DATA_DIR = "/Users/kai/project/train"


def make_windows(F, target, W, Np):
    """F:(M,6) 特征序列, target:(M,) 扰动 -> X:(样本,W,6), Y:(样本,Np)"""
    X, Y = [], []
    M = len(F)
    for k in range(W, M - Np):
        X.append(F[k - W:k])                      # 过去 W 步
        Y.append(target[k:k + Np] - target[k-1])  # 未来 Np 步"相对窗口末端 d_last 的增量"
    return np.array(X), np.array(Y)


def segment_to_windows(seg, vc, model, W, Np):
    """一段干净数据 -> (X, Y) 滑窗样本。"""
    f = features.compute_feature(seg, vc)
    d = features.compute_disturbance(f, model)
    d_smooth = features.smooth_disturbance(d)
    x, U = f["x"], f["U"]
    # 6 维特征: [d_e1dot, e1, e1dot, e2, e2dot, U]  (对齐到 N-1)
    F = np.column_stack([
        d_smooth[:, 1], x[:-1, 0], x[:-1, 1], x[:-1, 2], x[:-1, 3], U[:-1]
    ])
    return make_windows(F, d_smooth[:, 1], W, Np)


def build_dataset(W=20, Np=5, val_ratio=0.2, out=None):
    if out is None:
        out = os.path.join(PROJECT_ROOT, "processed", "dataset.npz")

    vc = VehicleConfig()
    model = BicycleModel(VehicleConfig(Ts=0.05), 10.0)   # 数据是 50ms

    # 1. 逐段切窗, 按段保存 (不立刻拼, 为了按段划分)
    seg_data = []
    for cat in ["lcc", "alc"]:
        for path in sorted(glob.glob(f"{DATA_DIR}/{cat}/*.MF4")):
            for seg in extract.extract_file(path):
                X, Y = segment_to_windows(seg, vc, model, W, Np)
                if len(X) > 0:
                    seg_data.append((X, Y))
    print(f"共 {len(seg_data)} 段产生窗口")

    # 2. 按"段"划分训练/验证 (防泄漏)
    np.random.seed(0)
    np.random.shuffle(seg_data)
    n_val = max(1, int(len(seg_data) * val_ratio))
    val, train = seg_data[:n_val], seg_data[n_val:]
    Xtr = np.concatenate([s[0] for s in train]); Ytr = np.concatenate([s[1] for s in train])
    Xva = np.concatenate([s[0] for s in val]);   Yva = np.concatenate([s[1] for s in val])

    # 3. 归一化特征 (只用 train 统计量)
    mean = Xtr.reshape(-1, 6).mean(axis=0)
    std = Xtr.reshape(-1, 6).std(axis=0)
    std[std < 1e-8] = 1.0                      # 防除 0
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    # 4. 存盘
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, mean=mean, std=std)
    print(f"train X{Xtr.shape} Y{Ytr.shape} | val X{Xva.shape} Y{Yva.shape}")
    print(f"特征 mean={mean.round(4)}")
    print(f"特征 std ={std.round(4)}")
    print(f"saved -> {out}")
    return Xtr, Ytr, Xva, Yva


if __name__ == "__main__":
    build_dataset()
