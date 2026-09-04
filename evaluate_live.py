import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_CSV = os.path.join(RESULTS_DIR, "live_run_log.csv")

if not os.path.exists(LOG_CSV):
    print(f"❌ Log file not found at: {LOG_CSV}.")
    sys.exit(1)

print("📄 Reading log file...")
try:
    df = pd.read_csv(LOG_CSV, on_bad_lines='skip')
except Exception as e:
    print(f"❌ Failed to parse CSV: {e}")
    sys.exit(1)

for col in ['x', 'y', 'speed']:
    if col not in df.columns:
        df[col] = 0.0

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

est_x = df["x"].values.astype(float)
est_y = df["y"].values.astype(float)
speeds = df["speed"].values.astype(float)

moving_samples = np.sum(speeds > 0.15)
total_samples = len(df)
movement_pct = (moving_samples / total_samples) * 100 if total_samples > 0 else 0

print("\n" + "=" * 55)
print("       LIVE RUN PERFORMANCE EVALUATION RESULTS")
print("=" * 55)
print(f"Total Samples Processed   : {total_samples}")
print(f"Movement Samples          : {moving_samples} ({movement_pct:.1f}%)")
print(f"Stationary Samples        : {total_samples - moving_samples} ({100 - movement_pct:.1f}%)")

if "distance" in df.columns and df["distance"].iloc[-1] > 0:
    total_dist = float(df["distance"].iloc[-1])
else:
    dt = 0.05
    total_dist = np.sum(np.where(speeds > 0.15, speeds * dt, 0.0))

print(f"Total Distance Navigated  : {total_dist:.2f} m")
print(f"Max Speed Recorded        : {float(np.max(speeds)):.2f} m/s")
print("=" * 55)

if len(df) > 10:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Trajectory
    axes[0, 0].plot(est_x, est_y, "b-", label="Estimated Path")
    axes[0, 0].scatter(est_x[0], est_y[0], color="green", s=100, label="Start")
    axes[0, 0].scatter(est_x[-1], est_y[-1], color="red", s=100, label="End")
    axes[0, 0].set_title("Trajectory")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Speed
    axes[0, 1].plot(speeds, "r-", label="Speed (m/s)")
    axes[0, 1].axhline(y=0.15, color="black", linestyle="--", label="Moving Cutoff")
    axes[0, 1].set_title("Speed Profile")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Distance
    dt = 0.05
    dist_curve = np.cumsum(np.where(speeds > 0.15, speeds * dt, 0.0))
    axes[1, 0].plot(dist_curve, "g-", label="Distance (m)")
    axes[1, 0].set_title("Distance Accumulation")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].axis("off")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "live_benchmark_comparison.png")
    plt.savefig(out_path, dpi=200)
    print(f"📊 Visualization saved to: {out_path}")