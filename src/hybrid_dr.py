import os
import numpy as np
import torch

class HybridDeadReckoning:
    def __init__(self, model_path="models/ai_adapter.pth", norm_path="models/norm_stats.npz"):
        self.pos = np.zeros(3)  # [x, y, z]
        self.vel = np.zeros(3)  # [vx, vy, vz]
        self.dt = 0.05
        self.total_distance = 0.0
        self.prev_pos = np.zeros(3)
        self.feature_history = []
        self.seq_len = 20

        # EKF Covariance Matrices
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.02
        self.R_gps = np.eye(3) * 4.0
        self.R_zupt = np.eye(3) * 0.005

        # AI Model Initialization
        self.ai_model = None
        self.mean = None
        self.std = None

        if os.path.exists(model_path) and os.path.exists(norm_path):
            try:
                from test import AIAdapter
                self.ai_model = AIAdapter(input_size=6, hidden_size=32)
                self.ai_model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.ai_model.eval()

                stats = np.load(norm_path)
                self.mean = stats["mean"]
                self.std = stats["std"]
            except Exception:
                self.ai_model = None

    def update(self, ax, ay, az, gx, gy, gz, dt=0.05, gps_pos=None, gps_valid=False):
        self.dt = max(0.001, min(dt, 0.2))

        # Stationary Detection
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        acc_diff = abs(acc_mag - 9.81)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        is_stationary = (gyro_mag < 0.06) and (acc_diff < 0.25)

        # 1. EKF State Prediction Step
        F = np.eye(6)
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt

        B = np.zeros((6, 3))
        B[3, 0] = self.dt
        B[4, 1] = self.dt
        B[5, 2] = self.dt

        linear_acc = np.array([ax, ay, az - 9.81])
        if abs(linear_acc[0]) < 0.12: linear_acc[0] = 0.0
        if abs(linear_acc[1]) < 0.12: linear_acc[1] = 0.0

        state = np.concatenate([self.pos, self.vel])
        state = F @ state + B @ linear_acc
        self.P = F @ self.P @ F.T + self.Q

        self.pos = state[:3]
        self.vel = state[3:]

        # 2. Measurement Updates
        if is_stationary:
            # ZUPT: velocity is zero
            H_zupt = np.zeros((3, 6))
            H_zupt[:, 3:] = np.eye(3)
            y_zupt = np.zeros(3) - self.vel
            S = H_zupt @ self.P @ H_zupt.T + self.R_zupt
            K = self.P @ H_zupt.T @ np.linalg.inv(S)

            state = np.concatenate([self.pos, self.vel]) + K @ y_zupt
            self.P = (np.eye(6) - K @ H_zupt) @ self.P
            self.pos = state[:3]
            self.vel = np.zeros(3)

        elif gps_valid and gps_pos is not None:
            # GPS Position Measurement
            H_gps = np.zeros((3, 6))
            H_gps[:3, :3] = np.eye(3)
            y_gps = gps_pos - self.pos
            S = H_gps @ self.P @ H_gps.T + self.R_gps
            K = self.P @ H_gps.T @ np.linalg.inv(S)

            state = np.concatenate([self.pos, self.vel]) + K @ y_gps
            self.P = (np.eye(6) - K @ H_gps) @ self.P
            self.pos = state[:3]
            self.vel = state[3:]

        # 3. AI Residual Correction (During GPS outages)
        if not gps_valid and self.ai_model is not None and not is_stationary:
            raw_features = np.array([ax, ay, az, gx, gy, gz])
            norm_feat = (raw_features - self.mean) / self.std
            self.feature_history.append(norm_feat)

            if len(self.feature_history) > self.seq_len:
                self.feature_history.pop(0)

            if len(self.feature_history) == self.seq_len:
                with torch.no_grad():
                    seq_tensor = torch.tensor([self.feature_history], dtype=torch.float32)
                    ai_residual = self.ai_model(seq_tensor).numpy()[0]
                    # Blend residual gently
                    self.pos += ai_residual * 0.05

        # 4. Distance Calculation (clamped to true movement)
        current_speed = np.linalg.norm(self.vel[:2])
        if current_speed > 0.10:
            step_disp = np.linalg.norm(self.pos[:2] - self.prev_pos[:2])
            self.total_distance += step_disp

        self.prev_pos = self.pos.copy()

        return {
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "z": float(self.pos[2]),
            "speed": float(current_speed if not is_stationary else 0.0),
            "distance": float(self.total_distance),
            "is_stationary": bool(is_stationary)
        }