import numpy as np

class KalmanFilter:
    def __init__(self,Ad,Bd,C,Q_kf,R_kf,x0,P0=None):
        self.Ad = Ad
        self.Bd = Bd
        self.C = C
        self.Q_kf = Q_kf
        self.R_kf = R_kf
        
        self.n = Ad.shape[0]
        self.x_est = x0.copy()
        self.P = P0 if P0 is not None else np.eye(self.n)

    def predict(self,u,vx,kappa):
        self.x_est = self.Ad @ self.x_est + self.Bd[:,0]*u + self.Bd[:,1] * (vx*kappa)
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q_kf

    def update(self,z):
        y = z - self.C @ self.x_est
        S = self.C @ self.P @ self.C.T + self.R_kf
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x_est = self.x_est + K @ y
        self.P = (np.eye(self.n) - K @ self.C) @ self.P
        return self.x_est