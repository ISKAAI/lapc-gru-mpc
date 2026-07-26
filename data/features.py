import os
import sys
import pandas as pd
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from data.channels import DT
from config import VehicleConfig
from model.bicycle_model import BicycleModel

def causal_filtered_derivative(values, dt=DT, tau=0.15):
    """Backward difference followed by a causal first-order low-pass filter."""
    values = np.asarray(values, dtype=float)
    raw = np.diff(values, prepend=values[0]) / dt
    alpha = dt / (tau + dt)
    filtered = np.empty_like(raw)
    filtered[0] = raw[0]
    for index in range(1, len(raw)):
        filtered[index] = alpha * raw[index] + (1.0 - alpha) * filtered[index - 1]
    return filtered


def compute_feature(seg, vehicle_config):
    e1 = -seg["c0"]
    e2 = -seg["c1"]
    k = 2*seg["c2"]
    U = seg["steer"] / vehicle_config.steering_ratio
    # Online-reproducible: never reads e1[k+1] to construct the feature at k.
    e1_dot = causal_filtered_derivative(e1)
    e2_dot = seg["yaw_rate"] - seg["vx"] * k

    x = np.stack([e1, e1_dot, e2, e2_dot], axis=1)
    return {"x": x, "U": U, "kappa": k, "vx": seg["vx"]}

def compute_disturbance(feat,model):
    x, U, kappa, vx = feat["x"], feat["U"], feat["kappa"], feat["vx"]
    N = len(x)
    d = np.zeros((N-1,4))
    for k in range(N-1):
        model.update_vx(vx[k])
        Ad, Bd, = model.Ad, model.Bd
        x_pred = Ad @ x[k] + Bd[:,0]*U[k] + Bd[:,1]*(vx[k]*kappa[k])
        d[k] = x[k+1] - x_pred
    return d

def smooth_disturbance(d,span=11):
    return pd.DataFrame(d).ewm(span=span).mean().to_numpy()

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try:
        from data import extract
    except ModuleNotFoundError:
        import extract
    seg = extract.extract_file("/Users/kai/project/train/lcc/xcp_2026-06-10_16-27-04.MF4")[0]
    f = compute_feature(seg, VehicleConfig())
    model = BicycleModel(VehicleConfig(Ts=0.05), 10.0)
    d = compute_disturbance(f, model)        # (N-1, 4)
    d_smooth = smooth_disturbance(d)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(d[:,1], alpha=0.4, label="raw d_e1dot")
    ax.plot(d_smooth[:,1], 'r', label="smoothed")
    ax.legend(); ax.grid()
    plt.savefig("smooth.png", dpi=110)
    print("raw std", round(d[:,1].std(),4), "-> smooth std", round(d_smooth[:,1].std(),4))
