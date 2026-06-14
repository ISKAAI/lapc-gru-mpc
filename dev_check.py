"""开发自检脚本：每完成一步就跑一次，确认没写崩。
用法:  conda activate ml  然后  python dev_check.py
"""
import numpy as np
from config import VehicleConfig, MPCConfig
from model.bicycle_model import BicycleModel
from controller.mpc import MPCController

# ---- 造一组假数据 ----
model = BicycleModel(VehicleConfig(), vx=15.0)
mpc = MPCController(model, MPCConfig())

x0 = np.array([0.3, 0.0, 0.05, 0.0])      # 横向偏 0.3m, 航向偏 0.05rad
kappa_seq = np.full(mpc.Np, 0.01)          # 未来曲率全 0.01
vx = 15.0

print(f"n={mpc.n}  m={mpc.m}  Np={mpc.Np}")

out = mpc.solve(x0, kappa_seq, vx)

if isinstance(out, tuple) and len(out) == 2:
    # Step 2~4 阶段: solve 返回 (constraints, cost)
    cons, cost = out
    expect = 1 + 2 * mpc.Np                 # 1初始 + Np动力学 + Np输入约束
    print(f"[约束数] {len(cons)}  (期望 {expect})")
    print(f"[cost ] 标量={cost.shape == ()}  合法凸={cost.is_dcp()}  (都应 True)")
else:
    # Step 5 之后: solve 返回 δ 标量
    print(f"[Step5] delta = {float(out):.5f} rad")

print("dev_check 跑完")
