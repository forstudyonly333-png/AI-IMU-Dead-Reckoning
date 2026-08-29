import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. TRAJECTORY COMPARISON
# ============================================================

def plot_trajectory(
    ground_truth,
    dr_position,
    ai_position,
    hybrid_position
):

    os.makedirs(
        "results",
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        label="Ground Truth",
        linewidth=2
    )

    plt.plot(
        dr_position[:, 0],
        dr_position[:, 1],
        label="Physics DR"
    )

    plt.plot(
        ai_position[:, 0],
        ai_position[:, 1],
        label="AI-DR"
    )

    plt.plot(
        hybrid_position[:, 0],
        hybrid_position[:, 1],
        label="Hybrid GPS + EKF + AI",
        linewidth=2
    )

    plt.xlabel(
        "X Position (m)"
    )

    plt.ylabel(
        "Y Position (m)"
    )

    plt.title(
        "Trajectory Comparison"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/trajectory_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 2. RMSE COMPARISON
# ============================================================

def plot_error_comparison(
    dr_error,
    ekf_error,
    ai_error,
    hybrid_error
):

    methods = [
        "Physics DR",
        "EKF",
        "AI-DR",
        "Hybrid"
    ]

    errors = [
        dr_error,
        ekf_error,
        ai_error,
        hybrid_error
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        methods,
        errors
    )

    plt.ylabel(
        "RMSE (meters)"
    )

    plt.title(
        "Position Estimation Error Comparison"
    )

    plt.grid(
        axis="y"
    )

    # Show value on top of each bar
    for bar, error in zip(
        bars,
        errors
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{error:.2f} m",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        "results/error_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 3. GPS OUTAGE ANALYSIS
# ============================================================

def plot_gps_outage(
    timestamps,
    ground_truth,
    dr_position,
    ai_position,
    hybrid_position,
    outage_start,
    outage_end
):

    plt.figure(figsize=(12, 6))

    plt.plot(
        timestamps,
        ground_truth[:, 0],
        label="Ground Truth X",
        linewidth=2
    )

    plt.plot(
        timestamps,
        dr_position[:, 0],
        label="Physics DR X"
    )

    plt.plot(
        timestamps,
        ai_position[:, 0],
        label="AI-DR X"
    )

    plt.plot(
        timestamps,
        hybrid_position[:, 0],
        label="Hybrid X",
        linewidth=2
    )

    # GPS outage region
    plt.axvspan(
        outage_start,
        outage_end,
        alpha=0.2,
        label="GPS Outage"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "X Position (meters)"
    )

    plt.title(
        "Position Estimation During GPS Outage"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/gps_outage.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 4. ERROR VS TIME
# ============================================================

def plot_error_over_time(
    timestamps,
    ground_truth,
    dr_position,
    ai_position,
    hybrid_position,
    outage_start,
    outage_end
):

    # Calculate Euclidean position error

    dr_error = np.linalg.norm(
        dr_position - ground_truth,
        axis=1
    )

    ai_error = np.linalg.norm(
        ai_position - ground_truth,
        axis=1
    )

    hybrid_error = np.linalg.norm(
        hybrid_position - ground_truth,
        axis=1
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        timestamps,
        dr_error,
        label="Physics DR Error"
    )

    plt.plot(
        timestamps,
        ai_error,
        label="AI-DR Error"
    )

    plt.plot(
        timestamps,
        hybrid_error,
        label="Hybrid Error",
        linewidth=2
    )

    # GPS outage
    plt.axvspan(
        outage_start,
        outage_end,
        alpha=0.2,
        label="GPS Outage"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Position Error (meters)"
    )

    plt.title(
        "Position Error Over Time"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/error_over_time.png",
        dpi=300
    )

    plt.close()