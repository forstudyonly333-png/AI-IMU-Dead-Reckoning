import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Model Definition (Matches test.py AIAdapter)
# ------------------------------------------------------------
class AIAdapter(nn.Module):
    def __init__(self, input_size=6, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ------------------------------------------------------------
# 2. Synthetic Dataset with Stationary & Dynamic Phases
# ------------------------------------------------------------
def generate_training_data(n_samples=4000, dt=0.05):
    np.random.seed(42)
    t = np.arange(0, n_samples * dt, dt)
    n = len(t)

    # Alternate between moving and resting every 20 seconds
    cycle = (t % 20.0) < 12.0  # True = moving, False = stationary

    # Ground truth trajectory
    gt_x = np.zeros(n)
    gt_y = np.zeros(n)
    gt_z = np.zeros(n)

    curr_x, curr_y = 0.0, 0.0
    for i in range(1, n):
        if cycle[i]:
            speed = 2.0 + 0.5 * np.sin(0.2 * t[i])
            heading = 0.1 * t[i]
            curr_x += speed * np.cos(heading) * dt
            curr_y += speed * np.sin(heading) * dt
        gt_x[i] = curr_x
        gt_y[i] = curr_y

    vx = np.gradient(gt_x, dt)
    vy = np.gradient(gt_y, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.full(n, 9.81)

    # Add realistic sensor bias and noise
    ax += np.random.normal(0.02, 0.04, n)
    ay += np.random.normal(-0.01, 0.04, n)
    az += np.random.normal(0, 0.03, n)

    gx = np.random.normal(0, 0.002, n)
    gy = np.random.normal(0, 0.002, n)
    gz = np.where(cycle, 0.1 + np.random.normal(0, 0.005, n), np.random.normal(0, 0.001, n))

    # Dead reckoning position calculation to determine residuals
    dr_x, dr_y, dr_z = np.zeros(n), np.zeros(n), np.zeros(n)
    vel = np.zeros(3)
    for i in range(1, n):
        linear_acc = np.array([ax[i], ay[i], az[i] - 9.81])
        if abs(linear_acc[0]) < 0.05: linear_acc[0] = 0.0
        if abs(linear_acc[1]) < 0.05: linear_acc[1] = 0.0
        vel = (vel + linear_acc * dt) * 0.98
        dr_x[i] = dr_x[i-1] + vel[0] * dt
        dr_y[i] = dr_y[i-1] + vel[1] * dt

    # Residual targets: ground truth minus dead reckoning
    target_res = np.column_stack([gt_x - dr_x, gt_y - dr_y, gt_z - dr_z])
    features = np.column_stack([ax, ay, az, gx, gy, gz])

    return features, target_res

# ------------------------------------------------------------
# 3. Main Training Execution
# ------------------------------------------------------------
def train_model():
    print("=" * 60)
    print("🧠 TRAINING AI-DR RESIDUAL ADAPTER")
    print("=" * 60)

    raw_features, targets = generate_training_data()

    # Calculate normalization statistics
    mean = np.mean(raw_features, axis=0)
    std = np.std(raw_features, axis=0) + 1e-8
    norm_features = (raw_features - mean) / std

    # Save normalization statistics for test.py / hybrid_dr.py
    stats_path = os.path.join(MODELS_DIR, "norm_stats.npz")
    np.savez(stats_path, mean=mean, std=std)
    print(f"✅ Normalization stats saved: {stats_path}")

    # Build sequence windows
    seq_len = 20
    x_seq, y_target = [], []
    for i in range(seq_len, len(norm_features)):
        x_seq.append(norm_features[i - seq_len:i])
        y_target.append(targets[i])

    x_tensor = torch.tensor(np.array(x_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_target), dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = AIAdapter(input_size=6, hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print("Training model across 30 epochs...")
    model.train()
    for epoch in range(1, 31):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        if epoch % 5 == 0 or epoch == 1:
            epoch_loss = total_loss / len(dataset)
            print(f"  Epoch [{epoch:02d}/30] - Loss (MSE): {epoch_loss:.6f}")

    model_path = os.path.join(MODELS_DIR, "ai_adapter.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model weights saved: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()