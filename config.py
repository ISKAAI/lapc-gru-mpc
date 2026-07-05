from dataclasses import dataclass, field
import numpy as np


@dataclass
class VehicleConfig:
    mass: float = 2350.0
    Iz: float = 4912.5
    lf: float = 1.573
    lr: float = 1.347
    Caf: float = 194129.0
    Car: float = 239985.0
    steering_ratio: float = 15.615
    Ts: float = 0.05      


@dataclass
class MPCConfig:
    Np: int = 20  
    Nc: int = 10
    Nd: int = 5            

    Q: np.ndarray = field(default_factory=lambda: np.diag([20.0, 0.5, 10.0, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))
    P: np.ndarray = field(default_factory=lambda: np.diag([20.0, 0.5, 10.0, 0.5]))
    S: float | None = 5.0

    delta_max: float = 0.1
    ddelta_max: float = 0.01

    
