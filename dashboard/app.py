import streamlit as st
import json
import os
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Live Smartphone Navigation",
    page_icon="📱",
    layout="wide"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
STATE_FILE = os.path.join(RESULTS_DIR, "live_state.json")
CONTROL_FILE = os.path.join(RESULTS_DIR, "outage_override.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. HELPERS
# ------------------------------------------------------------
def get_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def is_outage_forced():
    if os.path.exists(CONTROL_FILE):
        try:
            with open(CONTROL_FILE, "r") as f:
                return json.load(f).get("force_outage", False)
        except Exception:
            pass
    return False

def set_outage_forced(status: bool):
    try:
        with open(CONTROL_FILE, "w") as f:
            json.dump({"force_outage": status}, f)
    except Exception:
        pass

if "speeds" not in st.session_state:
    st.session_state.speeds = [0.0] * 25

# ------------------------------------------------------------
# 2. STATIC HEADER & CONTROL BUTTONS
# ------------------------------------------------------------
st.title("📱 Live Smartphone Navigation")
st.caption("Real-time smartphone IMU telemetry received by the Python engine via local TCP sockets.")

c1, c2, _ = st.columns([1.6, 1.6, 4])
with c1:
    if st.button("🚨 Simulate GPS Outage (Tunnel)", width="stretch", type="primary" if is_outage_forced() else "secondary"):
        set_outage_forced(True)
        st.rerun()

with c2:
    if st.button("🛰️ Restore GPS Lock", width="stretch", type="secondary" if is_outage_forced() else "primary"):
        set_outage_forced(False)
        st.rerun()

# Jaipur - Kota Highway Corridor Coordinates
start_lat, start_lon = 26.785892, 75.818937
highway_corridor = [
    [26.790500, 75.816000],
    [26.788500, 75.817800],
    [26.786500, 75.819800],
    [26.784000, 75.822200],
    [26.782000, 75.824200]
]

# ------------------------------------------------------------
# 3. UNIFIED SMOOTH REFRESH FRAGMENT (600ms)
# ------------------------------------------------------------
@st.fragment(run_every="600ms")
def render_live_dashboard():
    state = get_state()
    forced_outage = is_outage_forced()

    speed_ms = float(state.get("speed", 0.0))
    speed_kmh = speed_ms * 3.6
    dist = float(state.get("distance", 0.0))
    step_count = int(state.get("step_count", 0))
    heading_deg = float(state.get("yaw", 0.0))
    curr_lat = float(state.get("latitude", start_lat))
    curr_lon = float(state.get("longitude", start_lon))
    path = state.get("path", [])
    gps_valid = bool(state.get("gps_valid", True)) and not forced_outage

    # Maintain fixed 25-point rolling window
    st.session_state.speeds.append(round(speed_kmh, 2))
    st.session_state.speeds = st.session_state.speeds[-25:]

    # Status Banner
    if gps_valid:
        st.success("🛰️ **GPS LOCK CONFIRMED** — Standard Hybrid GPS + EKF Navigation Active.")
    else:
        st.error("🚨 **GPS OUTAGE DETECTED** — AI-ML Intelligent Dead Reckoning Active.")

    # 4 Main Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<p style='color: #8E95A5; margin-bottom: 2px; font-size: 14px;'>Navigation Mode</p>", unsafe_allow_html=True)
        if gps_valid:
            st.markdown("<h3 style='margin: 0; color: #00E676;'>🟢 GPS + EKF</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='margin: 0; color: #FF4B4B;'>🔴 AI-DR (Dead Reckoning)</h3>", unsafe_allow_html=True)

    with m2:
        st.metric("Current Speed", f"{speed_kmh:.1f} km/h", f"{speed_ms:.2f} m/s")

    with m3:
        st.metric("Heading (Yaw)", f"{heading_deg:.1f}°")

    with m4:
        st.metric("Distance Covered", f"{dist:.1f} m", f"{step_count} Steps")

    # Live Speed Graph with clean, positive 1..25 indices
    st.markdown("**⚡ Live Telemetry Speed Response (km/h)**")
    df_chart = pd.DataFrame(
        {"Speed": st.session_state.speeds},
        index=[f"T-{25-i}" for i in range(25)]
    )
    st.line_chart(df_chart, height=130, width="stretch")

    # Attitude Accordion
    with st.expander("📐 Sensor Orientation & Attitude Readouts", expanded=False):
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Roll", f"{float(state.get('roll', 0.0)):.2f}°")
        e2.metric("Pitch", f"{float(state.get('pitch', 0.0)):.2f}°")
        e3.metric("Local X Displacement", f"{float(state.get('x', 0.0)):.2f} m")
        e4.metric("Local Y Displacement", f"{float(state.get('y', 0.0)):.2f} m")

    # Folium Map
    m = folium.Map(
        location=[(start_lat + curr_lat) / 2.0, (start_lon + curr_lon) / 2.0],
        zoom_start=18,
        tiles="OpenStreetMap"
    )

    # 1. Orange Highway Corridor
    folium.PolyLine(
        highway_corridor,
        color="#FF7043",
        weight=20,
        opacity=0.45,
        tooltip="Jaipur - Kota Highway Corridor"
    ).add_to(m)

    folium.PolyLine(
        highway_corridor,
        color="#D84315",
        weight=3,
        dash_array="8, 10"
    ).add_to(m)

    # 2. Dotted Guidance Link to Highway Corridor
    folium.PolyLine(
        [[curr_lat, curr_lon], highway_corridor[2]],
        color="#8D6E63",
        weight=4,
        dash_array="4, 8",
        opacity=0.7
    ).add_to(m)

    # 3. Dedicated Walking Track between Start and Current Position
    active_track = path if (path and len(path) > 1) else [[start_lat, start_lon], [curr_lat, curr_lon]]
    folium.PolyLine(
        active_track,
        color="#E53935",
        weight=6,
        opacity=0.95,
        tooltip="Dead Reckoning Trajectory"
    ).add_to(m)

    # 4. Green Start Marker
    folium.Marker(
        [start_lat, start_lon],
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        popup="Starting Point"
    ).add_to(m)

    # 5. Red Current Position Marker
    folium.Marker(
        [curr_lat, curr_lon],
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
        popup="Current Dead Reckoning Position"
    ).add_to(m)

    folium.CircleMarker(
        [curr_lat, curr_lon],
        radius=12,
        color="#E53935",
        weight=2,
        fill=True,
        fill_color="#FFCDD2",
        fill_opacity=0.6
    ).add_to(m)

    st_folium(m, width=1200, height=480, key="nav_map_stream", returned_objects=[])

render_live_dashboard()