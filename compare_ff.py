import numpy as np
import matplotlib.pyplot as plt
from config import VehicleConfig, MPCConfig
from model.bicycle_model import BicycleModel
from controller.mpc import MPCController
from controller.kalman import KalmanFilter

vx, N_sim = 15, 100
model = BicycleModel(VehicleConfig(), vx)
mpc = MPCController(model, MPCConfig())
Np, Nd = mpc.Np, mpc.cfg.Nd

s = np.arange(N_sim + Np + Nd)
kappa_full = 0.01 * np.sin(0.05 * s)
sigma = np.array([0.02, 0.01, 0.01, 0.01])
E, I4 = np.array([[0.],[1.],[0.],[0.]]), np.eye(4)

def true_dist(k):                      # 真实扰动(仅作用在 ė1)
    return 0.05 * np.sin(0.1 * k)

def run(mode):
    np.random.seed(0)                  # ★ 同一套噪声, 公平对比
    delay_buffer = [0.0]*Nd
    delta = applied = 0.0
    x_true = np.array([0.5, 0., 0., 0.])
    Ad_aug = np.block([[model.Ad, E], [np.zeros((1,4)), np.ones((1,1))]])
    Bd_aug = np.block([[model.Bd], [np.zeros((1,2))]])
    C_aug  = np.block([I4, np.zeros((4,1))])
    Q_aug  = np.eye(5)*1e-4; Q_aug[4:,4:] *= 100
    kf = KalmanFilter(Ad_aug, Bd_aug, C_aug, Q_aug, np.diag(sigma**2),
                      np.concatenate([x_true, [0.0]]))
    log_e1 = []
    for k in range(N_sim):
        x_meas = x_true + np.random.normal(0, sigma, size=4)
        kf.predict(applied, vx, kappa_full[k])
        z = kf.update(x_meas)
        x_est, d_scalar = z[:4], z[4]

        if   mode == "floor":   d_seq = np.zeros((Nd+Np, 4))                       # 无前馈
        elif mode == "dob":     d_seq = np.tile(E.flatten()*d_scalar, (Nd+Np, 1))  # 纯DOB(常值)
        elif mode == "oracle":  d_seq = np.outer(true_dist(k+np.arange(Nd+Np)), E.flatten())  # 完美预测

        kappa_seq = kappa_full[k:k+Np+Nd]
        delta = mpc.solve(x_est, kappa_seq, vx, delta, pending=delay_buffer, d_seq=d_seq)
        applied = delay_buffer.pop(0); delay_buffer.append(delta)
        x_true = (model.Ad @ x_true + model.Bd[:,0]*applied
                  + model.Bd[:,1]*(vx*kappa_full[k]) + np.array([0., true_dist(k), 0., 0.]))
        log_e1.append(x_true[0])
    return np.array(log_e1)

modes = {"floor (no FF)": "floor", "DOB (const)": "dob", "oracle (perfect)": "oracle"}
plt.figure(figsize=(9,4)); results = {}
for label, m in modes.items():
    e1 = run(m); results[m] = e1
    rms   = np.sqrt((e1**2).mean())            # 全程 RMS
    rms_ss= np.sqrt((e1[30:]**2).mean())       # ★ 稳态 RMS(跳过前30步收敛段)
    print(f"{label:18s} RMS={rms:.5f}  稳态RMS={rms_ss:.5f}  max|e1|={np.abs(e1).max():.5f}")
    plt.plot(e1, label=f"{label}  (稳态RMS={rms_ss:.4f})")
plt.axhline(0, color='k', lw=0.5); plt.legend(); plt.grid()
plt.xlabel("step"); plt.ylabel("e1 [m]"); plt.title("floor vs DOB vs oracle")
plt.tight_layout(); plt.savefig("compare_ff.png", dpi=120)
np.savez("compare_ff.npz", **results)          # 三条曲线也存下来
print("saved compare_ff.png / compare_ff.npz")