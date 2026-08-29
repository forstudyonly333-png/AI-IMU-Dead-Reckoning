import numpy as np
import pandas as pd
import torch

from src.data_loader import generate_demo_data
from src.preprocessing import preprocess_imu
from src.dead_reckoning import dead_reckoning
from src.ekf import SimpleEKF
from src.ai_adapter import AIAdapter
from src.evaluation import rmse
from src.visualization import (
    plot_error_over_time,
    plot_trajectory,
    plot_error_comparison,
    plot_gps_outage
)
from src.hybrid_dr import hybrid_position

print("=" * 60)
print("       AI-IMU DEAD RECKONING PROTOTYPE")
print("=" * 60)


# ============================================================
# 1. Generate IMU Dataset
# ============================================================

df = generate_demo_data()

print("\n[1] IMU data generated")
print("Shape:", df.shape)


# ============================================================
# 2. Preprocessing
# ============================================================

df = preprocess_imu(df)

print("[2] Preprocessing complete")


# ============================================================
# 3. Physics-Based Dead Reckoning
# ============================================================

dr_position, velocity = dead_reckoning(df)

print("[3] Physics Dead Reckoning complete")


# ============================================================
# 4. Ground Truth FROM DATASET
# ============================================================

t = df["timestamp"].values

ground_truth = df[
    ["gt_x", "gt_y", "gt_z"]
].values


print("[4] Ground truth loaded from dataset")


# ============================================================
# 5. GPS Simulation
# ============================================================

gps_position = ground_truth.copy()

outage_start = 20
outage_end = 40

gps_available = (
    (t < outage_start)
    |
    (t > outage_end)
)

print(
    f"[5] GPS outage simulated: "
    f"{outage_start}s - {outage_end}s"
)


# ============================================================
# 6. EKF
# ============================================================

ekf = SimpleEKF()

ekf_position = np.zeros(
    (len(t), 3)
)


for i in range(len(t)):

    acceleration = np.array([
        df.iloc[i]["ax"],
        df.iloc[i]["ay"],
        df.iloc[i]["az"] - 9.81
    ])

    dt = df.iloc[i]["dt"]

    # Prediction
    ekf.predict(
        acceleration,
        dt
    )

    # GPS correction
    if gps_available[i]:

        ekf.update(
            gps_position[i]
        )

    ekf_position[i] = (
        ekf.get_position()
    )


print(
    "[6] EKF prediction/correction complete"
)


# ============================================================
# 7. Load Trained LSTM
# ============================================================

model = AIAdapter(
    input_size=6,
    hidden_size=32
)

model.load_state_dict(
    torch.load(
        "models/ai_adapter.pth",
        map_location="cpu"
    )
)

model.eval()

print(
    "[7] Trained LSTM model loaded"
)


# ============================================================
# 8. Prepare IMU Features
# ============================================================

features = df[
    ["ax", "ay", "az", "gx", "gy", "gz"]
].values.astype(
    np.float32
)


# Normalization
feature_mean = features.mean(axis=0)

feature_std = (
    features.std(axis=0)
    + 1e-8
)

features = (
    features - feature_mean
) / feature_std


# ============================================================
# 9. LSTM Residual Prediction
# ============================================================

sequence_length = 20

ai_correction = np.zeros(
    (len(features), 3)
)


with torch.no_grad():

    for i in range(
        sequence_length,
        len(features)
    ):

        sequence = features[
            i-sequence_length:i
        ]

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        prediction = model(
            sequence
        )

        ai_correction[i] = (
            prediction.numpy()[0]
        )


print(
    "[8] LSTM residual prediction complete"
)


# ============================================================
# 10. AI-DR Position
# ============================================================

ai_position = (
    dr_position
    + ai_correction
)

# ============================================================
# Hybrid GPS + EKF + AI
# ============================================================

hybrid_position_result = hybrid_position(
    gps_position,
    ekf_position,
    ai_position,
    gps_available
)

print(
    "[9] AI-DR position calculated"
)

print(
    "[10] Hybrid GPS + EKF + AI position calculated"
)



# ============================================================
# 11. Evaluation
# ============================================================

dr_error = rmse(
    dr_position,
    ground_truth
)

ekf_error = rmse(
    ekf_position,
    ground_truth
)

ai_error = rmse(
    ai_position,
    ground_truth
)

hybrid_error = rmse(
    hybrid_position_result,
    ground_truth
)


# ============================================================
# 12. Improvements
# ============================================================

ekf_improvement = (
    (dr_error - ekf_error)
    / dr_error
) * 100


ai_improvement = (
    (dr_error - ai_error)
    / dr_error
) * 100


# ============================================================
# 13. Print Results
# ============================================================

print("\n")

print("=" * 60)
print("                    RESULTS")
print("=" * 60)

print(
    f"\nPhysics DR RMSE : "
    f"{dr_error:.4f} m"
)

print(
    f"EKF RMSE        : "
    f"{ekf_error:.4f} m"
)

print(
    f"Actual AI-DR RMSE: "
    f"{ai_error:.4f} m"
)

print(
    f"Hybrid RMSE      : "
    f"{hybrid_error:.4f} m"
)

print(
    f"\nEKF improvement: "
    f"{ekf_improvement:.2f}%"
)

print(
    f"AI-DR improvement: "
    f"{ai_improvement:.2f}%"
)


# ============================================================
# 14. Save Results
# ============================================================

results_df = pd.DataFrame({
    "timestamp": t,

    "ground_truth_x": ground_truth[:, 0],
    "ground_truth_y": ground_truth[:, 1],

    "dr_x": dr_position[:, 0],
    "dr_y": dr_position[:, 1],

    "ekf_x": ekf_position[:, 0],
    "ekf_y": ekf_position[:, 1],

    "ai_dr_x": ai_position[:, 0],
    "ai_dr_y": ai_position[:, 1],

    "hybrid_x": hybrid_position_result[:, 0],
    "hybrid_y": hybrid_position_result[:, 1],

    "gps_available": gps_available
})


results_df.to_csv(
    "results/final_results.csv",
    index=False
)

# ============================================================
# 15. Visualization
# ============================================================

print("\nGenerating visualizations...")

plot_trajectory(
    ground_truth,
    dr_position,
    ai_position,
    hybrid_position_result
)

print("[V1] Trajectory graph saved")


plot_error_comparison(
    dr_error,
    ekf_error,
    ai_error,
    hybrid_error
)

print("[V2] Error comparison graph saved")


plot_gps_outage(
    t,
    ground_truth,
    dr_position,
    ai_position,
    hybrid_position_result,
    outage_start,
    outage_end
)

print("[V3] GPS outage graph saved")


plot_error_over_time(
    t,
    ground_truth,
    dr_position,
    ai_position,
    hybrid_position_result,
    outage_start,
    outage_end
)

print("[V4] Error-over-time graph saved")


print("\n============================================")
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("============================================")

print("results/trajectory_comparison.png")
print("results/error_comparison.png")
print("results/gps_outage.png")
print("results/error_over_time.png")

print("\nPrototype completed successfully!")


print(
    "\nResults saved successfully!"
)

print(
    "results/final_results.csv"
)

print(
    "results/trajectory_comparison.png"
)

print(
    "results/error_comparison.png"
)


print(
    "results/gps_outage.png"
)

print(
    "results/error_over_time.png"
)

print(
    "\nPrototype completed successfully!"
)