from pathlib import Path
from config import VehicleConfig, MPCConfig
from model.bicycle_model import BicycleModel
from controller.mpc import MPCController
from controller.gru_observer import GRUObserver

from data import extract


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    PROJECT_ROOT
    /"processed"
    /"combined_gru_h25"
    /"best_model.pt"
)

TEST_LIST_PATH = (
    PROJECT_ROOT
    /"processed"
    /"highway_data"
    /"test_normal.txt"
)

def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )
    initial_vx = 15.0
    vehicle_config = VehicleConfig()
    mpc_config = MPCConfig()

    mpc_model = BicycleModel(vehicle_config,initial_vx)
    residual_model = BicycleModel(vehicle_config,initial_vx)

    mpc = MPCController(mpc_model,mpc_config)
    observer = GRUObserver(
        checkpoint_path=CHECKPOINT_PATH,
        residual_model=residual_model,
        device="cpu",
        smooth_span=11
        )


    required_horizon = mpc.Nd + mpc.Np

    if observer.horizon != required_horizon:
        raise ValueError(
            f"Horizon mismatch: GRU outputs "
            f"{observer.horizon} steps, but MPC requires "
            f"{required_horizon} steps"
        )
    if mpc_model is residual_model:
        raise RuntimeError(
            "MPC and GRUObserver must use independent "
            "BicycleModel instances"
        )
    print("checkpoint:", CHECKPOINT_PATH)
    print("observer window:", observer.window)
    print("observer features:", observer.n_feat)
    print("observer horizon:", observer.horizon)
    print("mpc required horizon:", required_horizon)
    print(
        "independent models:",
        mpc_model is not residual_model
    )

    if not TEST_LIST_PATH.is_file():
        raise FileNotFoundError(
            f"test list path not found:{TEST_LIST_PATH}"
        )
    test_file_text = TEST_LIST_PATH.read_text(encoding = "utf-8")
    test_file_lines = test_file_text.splitlines()

    test_file_paths = []

    for raw_line in test_file_lines:
        clean_line = raw_line.strip()
        if not clean_line:
            continue
        file_path = Path(clean_line)
        test_file_paths.append(file_path)
    if not test_file_paths:
        raise ValueError(
            f"No test files found in: {TEST_LIST_PATH}"
        )
    test_file_path = test_file_paths[0]

    if not test_file_path.is_file():
        raise FileNotFoundError(
            f"Selected test file not found: {test_file_path}"
        )
    segments = extract.extract_file(test_file_path)
    if not segments:
        raise ValueError(
            f"No valid LCC segments extracted from: "
            f"{test_file_path}"
        )
    segment = segments[0]

    channel_lengths = {
        channel_name: len(channel_values)
        for channel_name, channel_values in segment.items()
    }
    unique_lengths = set(channel_lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Segment channels have inconsistent lengths: "
            f"{channel_lengths}"
        )
    num_samples = len(segment["vx"])
    minimum_samples = (
        observer.window
        + observer.horizon
        + 1
    )

    if num_samples < minimum_samples:
        raise ValueError(
            f"Segment is too short: got {num_samples} samples, "
            f"but at least {minimum_samples} are required"
    )

    vx_min = float(segment["vx"].min())
    vx_max = float(segment["vx"].max())
    print("test files found:", len(test_file_paths))
    print("selected test file:", test_file_path)
    print("valid segments:", len(segments))
    print("selected segment samples:", num_samples)
    print("segment channels:", sorted(segment.keys()))
    print("channel lengths:", channel_lengths)
    print(
        "vx range:",
        f"{vx_min:.2f} to {vx_max:.2f} m/s"
    )

if __name__ == "__main__":
    main()