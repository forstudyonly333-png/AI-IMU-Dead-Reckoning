import os
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs("results", exist_ok=True)

def generate_test_data(seed=123):
    np.random.seed(seed)
    t = np.arange(0, 100, 0.05)
    n = len(t)
    dt = 0.05

    gt_x = 20 * np.sin(0.05 * t)
    gt_y = 10 * np.sin(0.1 * t)
    gt_z = np.zeros(n)

    vx = np.gradient(gt_x, dt)
    vy = np.gradient(gt_y, dt)
    ax = np.gradient(vx, dt) + np.random.normal(0, 0.05, n)
    ay = np.gradient(vy, dt) + np.random.normal(0, 0.05, n)
    az = 9.81 + np.random.normal(0, 0.02, n)

    gx = np.random.normal(0, 0.002, n)
    gy = np.random.normal(0, 0.002, n)
    gz = 0.02 * np.cos(0.1 * t) + np.random.normal(0, 0.001, n)

    return pd.DataFrame({
        'timestamp': t,
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'gt_x': gt_x, 'gt_y': gt_y, 'gt_z': gt_z,
        'dt': dt
    })

def dead_reckoning(df):
    n = len(df)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    dt = df['dt'].values[0]

    for i in range(1, n):
        # Linear horizontal acceleration with noise rejection
        lax = df.iloc[i]['ax']
        lay = df.iloc[i]['ay']
        
        if abs(lax) < 0.05: lax = 0.0
        if abs(lay) < 0.05: lay = 0.0

        vel[i, 0] = (vel[i-1, 0] + lax * dt) * 0.99
        vel[i, 1] = (vel[i-1, 1] + lay * dt) * 0.99

        pos[i, 0] = pos[i-1, 0] + vel[i, 0] * dt
        pos[i, 1] = pos[i-1, 1] + vel[i, 1] * dt

    return pos, vel

class SimpleEKF:
    def __init__(self, gps_noise_std=3.0):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01
        self.R = np.eye(3) * (gps_noise_std ** 2)

    def predict(self, acceleration, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        B = np.zeros((6, 3))
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt

        state = np.concatenate([self.pos, self.vel])
        state = F @ state + B @ acceleration
        self.P = F @ self.P @ F.T + self.Q

        self.pos = state[:3]
        self.vel = state[3:]

    def update(self, gps_pos):
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        y = gps_pos - self.pos
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        state = np.concatenate([self.pos, self.vel]) + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.pos = state[:3]
        self.vel = state[3:]

    def get_position(self):
        return self.pos.copy()

def rmse(p1, p2):
    return np.sqrt(np.mean((p1 - p2) ** 2))

def main():
    df = generate_test_data()
    t = df['timestamp'].values
    dt = df['dt'].values[0]
    gt = df[['gt_x', 'gt_y', 'gt_z']].values

    dr_pos, _ = dead_reckoning(df)

    ekf = SimpleEKF()
    ekf_pos = np.zeros_like(gt)
    gps_available = (t < 20) | (t > 45)

    for i in range(len(t)):
        accel = np.array([df.iloc[i]['ax'], df.iloc[i]['ay'], df.iloc[i]['az'] - 9.81])
        ekf.predict(accel, dt)
        if gps_available[i]:
            gps_sample = gt[i] + np.random.normal(0, 1.5, 3)
            ekf.update(gps_sample)
        ekf_pos[i] = ekf.get_position()

    print(f"DR RMSE  : {rmse(dr_pos, gt):.3f} m")
    print(f"EKF RMSE : {rmse(ekf_pos, gt):.3f} m")
    print("✅ Evaluation complete.")

if __name__ == "__main__":
    main()