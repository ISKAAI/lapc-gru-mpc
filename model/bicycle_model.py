from dataclasses import dataclass
import numpy as np
from scipy.signal import cont2discrete

class BicycleModel:
    def __init__(self, config, vx):
        self.config = config
        self.vx = vx

        self.A = None
        self.B = None
        self.Ad = None
        self.Bd = None

        self.A, self.B = self.build_continuous(vx)
        self.Ad, self.Bd = self.discretize(self.A,self.B)

    def build_continuous(self, vx):
        m = self.config.mass
        Iz = self.config.Iz
        Caf = self.config.Caf
        Car = self.config.Car
        lf = self.config.lf
        lr = self.config.lr

        A = np.array([
            [0 ,1, 0, 0],
            [0, -( Caf + Car ) / ( m * vx ), ( Caf + Car ) / m, ( -Caf * lf + Car * lr) / ( m * vx)],
            [0, 0, 0, 1],
            [0, -( Caf * lf - Car * lr ) / ( Iz * vx), ( Caf * lf - Car * lr) / Iz, -( Caf * lf ** 2 + Car * lr ** 2) / (Iz * vx)]
        ])

        B = np.array([
            [0,0],
            [Caf/m,-( Caf * lf - Car * lr ) / ( m * vx) - vx],
            [0,0],
            [(Caf * lf) / Iz, -( Caf * lf ** 2 + Car * lr ** 2) / (Iz * vx)]
        ])
        return A , B

    def discretize(self, A, B):
        C = np.eye(4)
        D = np.zeros((4,2))

        Ad, Bd, _, _, _ = cont2discrete (
            (A, B, C, D),
            dt = self.config.Ts,
            method = 'zoh'
        )
        return Ad, Bd

    def update_vx(self, vx):
        self.vx = vx
        self.A, self.B = self.build_continuous(vx)
        self.Ad, self.Bd = self.discretize(self.A, self.B)

    def wheel_deg2front_rad(self, wheel_deg):
        ratio = self.config.steering_ratio
        wheel_rad = np.deg2rad( wheel_deg)
        front_rad = wheel_rad / ratio
        return front_rad
    
    def front_rad2wheel_deg(self,front_rad):
        ratio = self.config.steering_ratio
        front_deg = np.rad2deg(front_rad)
        wheel_deg = front_deg * ratio
        return wheel_deg