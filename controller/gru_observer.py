from pathlib import Path
import numpy as np
import torch
from model.gru import DisturbanceGRU
from collections import deque

class GRUObserver:
    def __init__(
            self, 
            checkpoint_path, 
            residual_model, 
            device="cpu",
            smooth_span = 11):
        self.device = torch.device(device)
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        model_config = checkpoint["model_config"]
        self.model = DisturbanceGRU(**model_config)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()
        self.residual_model = residual_model

        self.window = int(checkpoint["window"])
        self.feature_mean = checkpoint["feature_mean"].cpu().numpy()
        self.feature_std = checkpoint["feature_std"].cpu().numpy()
        self.feature_names = checkpoint["feature_names"]

        self.n_feat = int(model_config["n_feat"])
        self.horizon = int(model_config["horizon"])
        self.history = deque(maxlen=self.window)
        self.smooth_span = int(smooth_span)
        if (self.smooth_span <= 0):
            raise ValueError(
                f"smooth span should be greater than 0"
            )
        self.alpha = 2.0/(self.smooth_span+1.0)
        self.reset()



    def predict_normalized(self, normalized_history):
        history = np.asarray( normalized_history, dtype=np.float32)
        expected_shape = (self.window, self.n_feat)

        if history.shape != expected_shape :
            raise ValueError(
                f"Expected history shape {expected_shape},"
                f"got {history.shape}"
            )
        x = torch.from_numpy(history)
        x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.inference_mode():
            prediction = self.model(x)
        prediction = (prediction.squeeze(0).cpu().numpy())
        return prediction

    def normalize_history(self, raw_history):
        x = np.asarray(raw_history, dtype=np.float32)
        expected_shape = (self.window, self.n_feat)
        if ( x.shape != expected_shape):
            raise ValueError(
                f"Expected raw history shape {expected_shape},"
                f"got {x.shape}"
            )

        x_norm = (x - self.feature_mean)/self.feature_std
        return x_norm

    def reset(self):
        self.history.clear()
        self.previous_x = None
        self.previous_front_steer = None
        self.previous_vx = None
        self.previous_kappa = None

        self.ewm_numerator = np.zeros(4,dtype=float)
        self.ewm_denominator = 0.0

    def append_raw_feature(self, raw_feature):
        feature = np.asarray(raw_feature, dtype=np.float32)
        expected_shape = (self.n_feat,)
        if feature.shape != expected_shape:
            raise ValueError(
                f"expceted shape {expected_shape},"
                f"get {feature.shape}"
            )
        self.history.append(feature.copy())

    @property
    def history_ready(self):
        return (len(self.history) == self.window)

    def predict_from_history(self):
        if not self.history_ready:
            return None
        raw_history = np.stack(self.history, axis=0)
        normalized_history = self.normalize_history(raw_history)
        prediction = self.predict_normalized(normalized_history)
        return prediction

    def observe_transition(
        self,
        current_x,
        current_front_steer,
        current_vx,
        current_kappa
    ):
        x = np.asarray(current_x, dtype=np.float32)
        if ( x.shape != (4,) ):
            raise ValueError(
                f"expected_shape (4,)",
                f"get shape {x.shape}"
            )
        front_steer = float(current_front_steer)
        vx = float(current_vx)
        kappa = float(current_kappa)

        if self.previous_x is None:
            self.previous_x = x.copy()
            self.previous_front_steer = front_steer
            self.previous_vx = vx
            self.previous_kappa = kappa

            return None

        self.residual_model.update_vx(self.previous_vx)
        Ad = self.residual_model.Ad
        Bd = self.residual_model.Bd

        x_prediction = Ad @ self.previous_x + Bd[:,0] * self.previous_front_steer + Bd[:,1] * ( self.previous_vx * self.previous_kappa)

        raw_residual = x - x_prediction
        smoothed_residual = self.smooth_residual(raw_residual)
        raw_feature = self.build_raw_feature(smoothed_residual)
        self.append_raw_feature(raw_feature)
        predicted_deltas = self.predict_from_history()
        if (predicted_deltas is None):
            d_seq = None
        else:
            d_seq = self.build_disturbance_sequence(predicted_deltas,smoothed_residual)

        self.previous_x = x.copy()
        self.previous_vx = vx
        self.previous_kappa = kappa
        self.previous_front_steer = front_steer

        return d_seq

    def smooth_residual(self,raw_residual):
        residual = np.asarray(raw_residual,dtype=float)
        if residual.shape != (4,):
            raise ValueError(
                f"expected residual shape (4,),"
                f"get residual shape{residual.shape}"
            )
        beta = 1.0 - self.alpha
        self.ewm_numerator = residual + beta * self.ewm_numerator
        self.ewm_denominator = 1 + beta * self.ewm_denominator

        smooth_residual = self.ewm_numerator / self.ewm_denominator
        return smooth_residual

    def build_raw_feature(self,smoothed_residual):
        residual = np.asarray(smoothed_residual, dtype=float)
        if (residual.shape != (4,)):
            raise ValueError(
                f"expected residual shape (4,),"
                f"got residual shape{residual.shape}"
            )
        if (self.previous_x is None):
            raise ValueError(
                f"State is not ready for the transition"
            )
        raw_feature = np.zeros(self.n_feat, dtype=np.float32)
        raw_feature[0] = residual[1]
        raw_feature[1:5] = self.previous_x
        raw_feature[5] = self.previous_front_steer
        raw_feature[6] = self.previous_vx

        expected_shape = (self.n_feat,)
        if raw_feature.shape != expected_shape:
            raise ValueError(
                f"the expected state shape {expected_shape}"
                f"current state shape is {raw_feature.shape}"
            )
        return raw_feature

    def build_disturbance_sequence(self, prediction_delta, current_smoothed_residual):
        predicted_deltas = np.asarray(prediction_delta, dtype=np.float32)
        if (predicted_deltas.shape != (self.horizon,)):
            raise ValueError(
                f"expected predicton deltas shape {(self.horizon,)},"
                f"get predicton deltas shape {predicted_deltas.shape}"
            )
        smoothed_residual = np.asarray(current_smoothed_residual, dtype=np.float32)
        if (smoothed_residual.shape != (4,)):
            raise ValueError(
                f"expected smoothed residual shape {(4)},"
                f"get current smoothed residual shape {smoothed_residual}"
            )
        current_d_e1dot = smoothed_residual[1]
        predicted_d_e1dot = current_d_e1dot + predicted_deltas
        d_seq = np.zeros((self.horizon,4),dtype=np.float32)
        d_seq[:,1] = predicted_d_e1dot

        return d_seq
