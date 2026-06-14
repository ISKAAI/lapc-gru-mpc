"""模式 B：用实车数据做开环反事实对比 —— 给定每一帧真实状态，看 MPC 会打多少方向盘，
与 TRPC 实际下发的方向盘角(ReqStrgWhlAngCmd)叠图对比。

跑法：conda activate ml  然后  python test_realdata.py
注意：本测试用 Ts=0.05（数据是 50ms），与项目 demo 的 Ts 不同。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from asammdf import MDF
from config import VehicleConfig, MPCConfig
from model.bicycle_model import BicycleModel
from controller.mpc import MPCController
from controller.kalman import KalmanFilter

# ───────────── 可调参数 ─────────────
MF4  = "/Users/kai/project/lapc_mpc_gru/xcp_2026-06-10_16-27-04.MF4"
SEG  = (0, 1500)          # LCC 连续激活片段 [a, b)
TS   = 0.05               # 数据采样周期 50ms
SIGN_EY   = -1.0          # e_y  = SIGN_EY * c0   (数据弱支持 -c0；若对比反向就翻这两个)
SIGN_EPSI = -1.0          # e_psi= SIGN_EPSI* c1
SMOOTH_N  = 5             # e_y 差分前的滑动平均窗口（抑制 50ms 差分噪声）

# ───────────── 读数据 ─────────────
m = MDF(MF4)
def g(name): return np.asarray(m.get(name).samples, float)
a, b = SEG
vx  = g("sm_req_ego_motion.velocity.x.value")[a:b]
c0  = g("lapc_req_lat_ctrl_lane_model.center_line.poly_line.c0")[a:b]
c1  = g("lapc_req_lat_ctrl_lane_model.center_line.poly_line.c1")[a:b]
c2  = g("lapc_req_lat_ctrl_lane_model.center_line.poly_line.c2")[a:b]
yaw = g("sm_req_can_in_chassis.yaw_rate")[a:b]
ang_trpc = g("TRPC_OnLineModel_Mix_Y.ReqStrgWhlAngCmd_sg")[a:b]   # 方向盘角 deg
N = len(vx)
kappa = 2.0 * c2                                                  # 道路曲率 1/m

# ───────────── 测量量（含噪，来自实车） ─────────────
e_y   = SIGN_EY   * c0                  # 横向误差测量
e_psi = SIGN_EPSI * c1                  # 航向误差测量
epsi_dot = yaw - vx * kappa             # 航向误差率（yaw、κ 均直接测量，干净）
# 朴素差分 ė_y：仅用于和 KF 对比画图，展示 KF 去噪效果（不再喂给 MPC）
e_y_s = np.convolve(e_y, np.ones(SMOOTH_N)/SMOOTH_N, mode="same")
ey_dot_naive = np.gradient(e_y_s, TS)

# ───────────── 建模型 + MPC + 卡尔曼（Ts=0.05） ─────────────
veh = VehicleConfig(); veh.Ts = TS
model = BicycleModel(veh, vx[0])
mpc = MPCController(model, MPCConfig())
Np = mpc.Np
ratio = veh.steering_ratio

# 量测矩阵：测 [e_y, e_psi, ė_ψ]，不测 ė_y（交给 KF 用模型推断）
C = np.array([[1., 0., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
R_kf = np.diag([0.02**2, 0.005**2, 0.005**2])       # 量测噪声方差（可调）
Q_kf = np.diag([1e-4, 1e-3, 1e-4, 1e-3])            # 过程噪声（速度态给大些，可调）
x0 = np.array([e_y[0], 0.0, e_psi[0], epsi_dot[0]])
kf = KalmanFilter(model.Ad, model.Bd, C, Q_kf, R_kf, x0=x0, P0=np.eye(4)*0.1)

# ───────────── 开环：KF 估状态 → MPC 反事实 δ ─────────────
delta_mpc_deg = np.full(N, np.nan)
ey_est    = np.full(N, np.nan)
eydot_est = np.full(N, np.nan)
for k in range(N - Np):
    # KF：用实车真实施加的 TRPC 转角做预测，再用含噪测量校正
    u_real = np.deg2rad(ang_trpc[k]) / ratio        # 实车实际施加的前轮转角(rad)
    model.update_vx(vx[k]); kf.Ad, kf.Bd = model.Ad, model.Bd
    kf.predict(u_real, vx[k], kappa[k])
    z = np.array([e_y[k], e_psi[k], epsi_dot[k]])
    x_est = kf.update(z)
    ey_est[k], eydot_est[k] = x_est[0], x_est[1]
    # MPC 反事实：给定 KF 估计状态，看 MPC 会打多少
    kseq = kappa[k:k+Np]
    delta_prev = np.deg2rad(ang_trpc[k-1]) / ratio if k > 0 else np.deg2rad(ang_trpc[0]) / ratio
    delta_front = mpc.solve(x_est, kseq, vx[k], delta_prev)
    delta_mpc_deg[k] = np.rad2deg(delta_front) * ratio

# ───────────── 指标 ─────────────
valid = ~np.isnan(delta_mpc_deg)
cc   = np.corrcoef(delta_mpc_deg[valid], ang_trpc[valid])[0, 1]
rmse = np.sqrt(np.nanmean((delta_mpc_deg - ang_trpc)**2))
print(f"δ:  corr(MPC,TRPC)={cc:+.3f}  RMSE={rmse:.2f}deg  (开环+卡尔曼)")

# ───────────── 画图 ─────────────
t = np.arange(N) * TS
fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
ax[0].plot(t, ang_trpc, label="TRPC (real)", color="C0")
ax[0].plot(t, delta_mpc_deg, label="MPC (KF)", color="C1", ls="--")
ax[0].set_ylabel("steering [deg]"); ax[0].legend(); ax[0].grid()
ax[0].set_title(f"δ: corr={cc:+.3f}  RMSE={rmse:.2f}deg  (with Kalman)")

ax[1].plot(t, e_y, label="meas (−c0)", color="C0", alpha=0.6)
ax[1].plot(t, ey_est, label="KF est", color="C1")
ax[1].set_ylabel("e_y [m]"); ax[1].legend(); ax[1].grid()

ax[2].plot(t, ey_dot_naive, label="naive diff (noisy)", color="C0", alpha=0.5)
ax[2].plot(t, eydot_est, label="KF est", color="C1")
ax[2].set_ylabel("ė_y [m/s]"); ax[2].legend(); ax[2].grid()

ax[3].plot(t, kappa, color="C4"); ax[3].set_ylabel("curvature [1/m]"); ax[3].grid()
ax[3].set_xlabel("time [s]")
plt.tight_layout(); plt.savefig("result_realdata.png", dpi=120)
print("已保存 result_realdata.png")
