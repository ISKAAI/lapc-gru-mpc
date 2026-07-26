"""Isolate MPC disturbance-feedforward value with an exact simulated state.

The previous version mixed Kalman-filter chronology errors with feedforward
quality, causing the perfect oracle to appear unstable. This benchmark keeps
the state identical across modes and varies only the disturbance sequence.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import VehicleConfig, MPCConfig
from model.bicycle_model import BicycleModel
from controller.mpc import MPCController


VX, N_SIM = 15.0, 100
E = np.array([0.0, 1.0, 0.0, 0.0])


def true_disturbance(step):
    return 0.05 * np.sin(0.1 * step)


def run(mode):
    model = BicycleModel(VehicleConfig(), VX)
    mpc = MPCController(model, MPCConfig())
    s = np.arange(N_SIM + mpc.Np + mpc.Nd)
    kappa = 0.01 * np.sin(0.05 * s)
    delay_buffer = [0.0] * mpc.Nd
    command = 0.0
    last_observed_disturbance = 0.0
    state = np.array([0.5, 0.0, 0.0, 0.0])
    log_e1 = []

    for step in range(N_SIM):
        if mode == "floor":
            scalar = np.zeros(mpc.Nd + mpc.Np)
        elif mode == "dob":
            scalar = np.full(mpc.Nd + mpc.Np, last_observed_disturbance)
        elif mode == "oracle":
            scalar = np.asarray(
                [true_disturbance(step + index) for index in range(mpc.Nd + mpc.Np)]
            )
        else:
            raise ValueError(mode)
        disturbance_sequence = np.outer(scalar, E)
        command = mpc.solve(
            state,
            kappa[step : step + mpc.Nd + mpc.Np],
            VX,
            command,
            pending=delay_buffer,
            d_seq=disturbance_sequence,
        )
        applied = delay_buffer.pop(0)
        delay_buffer.append(command)
        actual_disturbance = true_disturbance(step)
        state = (
            model.Ad @ state
            + model.Bd[:, 0] * applied
            + model.Bd[:, 1] * (VX * kappa[step])
            + E * actual_disturbance
        )
        last_observed_disturbance = actual_disturbance
        log_e1.append(state[0])
    return np.asarray(log_e1)


if __name__ == "__main__":
    modes = {
        "floor (no FF)": "floor",
        "DOB (last observed)": "dob",
        "oracle (perfect)": "oracle",
    }
    results = {}
    plt.figure(figsize=(9, 4))
    for label, mode in modes.items():
        e1 = run(mode)
        results[mode] = e1
        rms = np.sqrt(np.mean(np.square(e1)))
        steady_rms = np.sqrt(np.mean(np.square(e1[30:])))
        print(
            f"{label:22s} RMS={rms:.6f} steady_RMS={steady_rms:.6f} "
            f"max|e1|={np.abs(e1).max():.6f}"
        )
        plt.plot(e1, label=f"{label} (steady RMS={steady_rms:.4f})")
    plt.axhline(0, color="k", lw=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel("step")
    plt.ylabel("e1 [m]")
    plt.title("MPC disturbance feedforward sanity check")
    plt.tight_layout()
    plt.savefig("compare_ff.png", dpi=120)
    np.savez("compare_ff.npz", **results)
    print("saved compare_ff.png / compare_ff.npz")
