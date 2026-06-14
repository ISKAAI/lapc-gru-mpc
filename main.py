import numpy as np
import matplotlib.pyplot as plt
from config import VehicleConfig, MPCConfig
from model.bicycle_model import BicycleModel
from controller.mpc import MPCController
from controller.kalman import KalmanFilter
vx = 15
N_sim = 100
#mpc
model = BicycleModel(VehicleConfig(),vx)
mpc = MPCController(model,MPCConfig())
Np = mpc.Np
Nd = mpc.cfg.Nd
delay_buffer = [0.0] * Nd

#initial position and ref
kappa_full = np.zeros(N_sim +Np)
s = np.arange(N_sim+Np)
kappa_full = 0.01 * np.sin(0.05 * s) 
delta = 0
x_true = np.array([0.5,0.,0.,0.])
log_x, log_u = [],[]

#noise
np.random.seed(0)
sigma = np.array([0.02,0.01,0.01,0.01])

#kalmer
C = np.eye(4)
R_kf = np.diag(sigma**2)
Q_kf = np.diag([1e-4,1e-4,1e-4,1e-4])
kf = KalmanFilter(model.Ad, model.Bd, C, Q_kf, R_kf, x0 = x_true.copy())
applied = 0.0

d_true = np.array([0.0,0.05,0.0,0.0])

for k in range(N_sim):

    noise = np.random.normal(0,sigma,size=4)
    x_meas = x_true + noise
    kf.predict(applied,vx,kappa_full[k])
    x_est = kf.update(x_meas)
    kappa_seq = kappa_full[k:k+Np]
    delta = mpc.solve(x_est,kappa_seq,vx,delta,pending=delay_buffer)

    applied = delay_buffer.pop(0)
    delay_buffer.append(delta)

    Ad, Bd = model.Ad,model.Bd
    x_true = Ad @ x_true + Bd[:,0]*applied + Bd[:,1] * ( vx * kappa_full[k]) + d_true
    log_x.append(x_true)
    log_u.append(applied)

log_x = np.array(log_x)
log_u = np.array(log_u)

fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

axes[0].plot(log_x[:,0], label='lateral error e1 [m]')
axes[0].plot(log_x[:,2], label='heading error e2 [rad]')
axes[0].legend(); axes[0].grid()

axes[1].plot( log_u, label='steering δ [rad]', color='C2') 
axes[1].legend(); axes[1].grid()

axes[2].plot(kappa_full[:N_sim], label='road curvature κ', color='C3')  
axes[2].legend(); axes[2].grid(); axes[2].set_xlabel('step')

plt.tight_layout()
plt.savefig('result.png', dpi=120)

