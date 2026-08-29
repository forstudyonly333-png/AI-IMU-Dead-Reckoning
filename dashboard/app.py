import streamlit as st
import pandas as pd
import numpy as np
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-IMU Dead Reckoning",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("🛰️ AI-IMU Dead Reckoning")

st.subheader(
    "GPS-Denied Position Estimation Prototype"
)

st.write(
    "Hybrid navigation system combining Physics Dead Reckoning, "
    "Extended Kalman Filter and LSTM-based AI correction."
)


# ============================================================
# LOAD DATA
# ============================================================

csv_path = "results/final_results.csv"

if not os.path.exists(csv_path):

    st.error(
        "Results file not found. Run 'python test.py' first."
    )

    st.stop()


df = pd.read_csv(csv_path)


# ============================================================
# CONSTANTS
# ============================================================

physics_rmse = 15.4797
ekf_rmse = 3.5162
ai_rmse = 11.0420
hybrid_rmse = 2.4575

outage_start = 20
outage_end = 40


# ============================================================
# GPS SIMULATION
# ============================================================

st.markdown("## 🛰️ GPS Simulation")

simulation_time = st.slider(
    "Select Simulation Time (seconds)",
    min_value=0.0,
    max_value=float(df["timestamp"].max()),
    value=10.0,
    step=0.5
)


# Find closest timestamp
index = (
    np.abs(df["timestamp"] - simulation_time)
).argmin()

current = df.iloc[index]


# ============================================================
# GPS STATUS
# ============================================================

if outage_start <= simulation_time <= outage_end:

    gps_status = "🔴 GPS SIGNAL LOST"

    st.error(
        f"GPS unavailable from {outage_start}s to {outage_end}s"
    )

    system_status = "EKF + AI Dead Reckoning Active"

else:

    gps_status = "🟢 GPS AVAILABLE"

    st.success(
        "GPS signal available"
    )

    system_status = "Hybrid GPS + EKF + AI Fusion"


# ============================================================
# STATUS DISPLAY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Current Time",
        f"{simulation_time:.1f} s"
    )

with col2:

    st.metric(
        "GPS Status",
        gps_status
    )

with col3:

    st.metric(
        "Navigation Mode",
        system_status
    )


# ============================================================
# CURRENT POSITION
# ============================================================

st.markdown("## 📍 Current Position Estimate")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Ground Truth X",
        f"{current['ground_truth_x']:.3f} m"
    )

with col2:

    st.metric(
        "Physics DR X",
        f"{current['dr_x']:.3f} m"
    )

with col3:

    st.metric(
        "AI-DR X",
        f"{current['ai_dr_x']:.3f} m"
    )

with col4:

    st.metric(
        "Hybrid X",
        f"{current['hybrid_x']:.3f} m"
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.markdown("## 📊 System Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Physics DR RMSE",
        f"{physics_rmse:.2f} m"
    )

with col2:

    st.metric(
        "EKF RMSE",
        f"{ekf_rmse:.2f} m"
    )

with col3:

    st.metric(
        "AI-DR RMSE",
        f"{ai_rmse:.2f} m"
    )

with col4:

    st.metric(
        "Hybrid RMSE",
        f"{hybrid_rmse:.2f} m"
    )


# ============================================================
# IMPROVEMENT
# ============================================================

improvement = (
    (physics_rmse - hybrid_rmse)
    / physics_rmse
) * 100


st.success(
    f"🚀 Hybrid system reduces RMSE by "
    f"{improvement:.2f}% compared with Physics DR."
)


# ============================================================
# TRAJECTORY
# ============================================================

st.markdown("## 📍 Live Trajectory Simulation")


visible_df = df[
    df["timestamp"] <= simulation_time
]


if len(visible_df) > 0:

    chart_df = visible_df[
        [
            "ground_truth_x",
            "ground_truth_y",
            "dr_x",
            "dr_y",
            "ai_dr_x",
            "ai_dr_y",
            "hybrid_x",
            "hybrid_y"
        ]
    ].copy()

    chart_df.columns = [
        "Ground Truth X",
        "Ground Truth Y",
        "Physics DR X",
        "Physics DR Y",
        "AI-DR X",
        "AI-DR Y",
        "Hybrid X",
        "Hybrid Y"
    ]

    st.line_chart(
        chart_df,
        x=None,
        use_container_width=True
    )


# ============================================================
# TRAJECTORY IMAGE
# ============================================================

st.markdown("## 🗺️ Complete Trajectory")

trajectory_path = (
    "results/trajectory_comparison.png"
)

if os.path.exists(trajectory_path):

    st.image(
        trajectory_path,
        use_container_width=True
    )


# ============================================================
# ERROR COMPARISON
# ============================================================

st.markdown("## 📉 RMSE Comparison")

error_path = (
    "results/error_comparison.png"
)

if os.path.exists(error_path):

    st.image(
        error_path,
        use_container_width=True
    )


# ============================================================
# GPS OUTAGE
# ============================================================

st.markdown("## 🛰️ GPS Outage Analysis")

gps_path = (
    "results/gps_outage.png"
)

if os.path.exists(gps_path):

    st.image(
        gps_path,
        use_container_width=True
    )


# ============================================================
# ERROR OVER TIME
# ============================================================

st.markdown("## 📈 Error Over Time")

error_time_path = (
    "results/error_over_time.png"
)

if os.path.exists(error_time_path):

    st.image(
        error_time_path,
        use_container_width=True
    )


# ============================================================
# DATA
# ============================================================

with st.expander("📋 View Dataset"):

    st.write(
        f"Total samples: {len(df)}"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-IMU Dead Reckoning Prototype | "
    "Physics DR + EKF + LSTM + Hybrid GPS Fusion"
)