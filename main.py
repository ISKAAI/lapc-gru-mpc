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
s = np.arange(N_sim+Np+Nd)
kappa_full = 0.01 * np.sin(0.05 * s) 
delta = 0
x_true = np.array([0.5,0.,0.,0.])
log_x, log_u, log_d = [], [], []

#noise
np.random.seed(0)
sigma = np.array([0.02,0.01,0.01,0.01])

#kalmen
C = np.eye(4)
R_kf = np.diag(sigma**2)
Q_kf = np.diag([1e-4,1e-4,1e-4,1e-4])

applied = 0.0

d_true = np.array([0.0,0.05,0.0,0.0])

#增广
E = np.array([[0.],[1.],[0.],[0.]])
I4 = np.eye(4)

Ad_aug = np.block([[model.Ad,E],[np.zeros((1,4)),np.ones((1,1))]])
Bd_aug = np.block([[model.Bd],[np.zeros((1,2))]])
C_aug = np.block([I4,np.zeros((4,1))])
Q_aug = np.eye(5)*1e-4
Q_aug[4:,4:] *= 100
x0_aug = np.concatenate([x_true,[0.0]])
kf = KalmanFilter(Ad_aug, Bd_aug, C_aug, Q_aug, R_kf, x0_aug)

for k in range(N_sim):

    noise = np.random.normal(0,sigma,size=4)
    x_meas = x_true + noise
    kf.predict(applied,vx,kappa_full[k])
    z_est = kf.update(x_meas)
    x_est = z_est[:4]
    d_scalar = z_est[4]
    d_seq = np.tile(E.flatten() * d_scalar, (Nd + Np, 1))


    kappa_seq = kappa_full[k:k+Np+Nd]
    delta = mpc.solve(x_est,kappa_seq,vx,delta,pending=delay_buffer,d_seq=d_seq)

    applied = delay_buffer.pop(0)
    delay_buffer.append(delta)

    Ad, Bd = model.Ad,model.Bd
    x_true = Ad @ x_true + Bd[:,0]*applied + Bd[:,1] * ( vx * kappa_full[k]) + np.array([0.,0.05*np.sin(0.1*k),0.,0.])
    log_x.append(x_true)
    log_u.append(applied)
    log_d.append(d_scalar)

log_x = np.array(log_x)
log_u = np.array(log_u)
log_d = np.array(log_d)

fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

axes[0].plot(log_x[:,0], label='lateral error e1 [m]')
axes[0].plot(log_x[:,2], label='heading error e2 [rad]')
axes[0].legend(); axes[0].grid()

axes[1].plot( log_u, label='steering δ [rad]', color='C2') 
axes[1].legend(); axes[1].grid()

axes[2].plot(kappa_full[:N_sim], label='road curvature κ', color='C3')  
axes[2].legend(); axes[2].grid(); axes[2].set_xlabel('step')

axes[3].plot( log_d, label=' forward disturbance')
axes[3].legend(); axes[3].grid()
plt.tight_layout()
plt.savefig('result.png', dpi=120)

