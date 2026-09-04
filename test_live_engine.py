import time
import json
import os
import numpy as np
from collections import deque

from src.live_receiver import IMUReceiver
from src.orientation import OrientationEstimator
from src.live_state import save_live_state

HOST = "0.0.0.0"
PORT = 5000

INITIAL_LAT = 26.7858633
INITIAL_LON = 75.8190191
METERS_PER_DEG_LAT = 111320.0

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONTROL_FILE = os.path.join(PROJECT_ROOT, "results", "outage_override.json")

receiver = IMUReceiver(host=HOST, port=PORT)
orientation = OrientationEstimator(alpha=0.98)

# ------------------------------------------------------------
# STEP DETECTOR STATE
# ------------------------------------------------------------
sample_count = 0
step_count = 0
x, y = 0.0, 0.0
total_distance = 0.0
current_speed = 0.0

# Dynamic thresholding parameters
is_in_step = False
last_step_timestamp = 0.0
step_cooldown = 0.28  # Max 3.5 steps/sec

# Low-pass filter deque
acc_window = deque(maxlen=4)
path = []

receiver.start()

print("\n" + "=" * 55)
print("🚶 ROBUST SMARTPHONE PDR STEP ENGINE ACTIVE")
print(f"📡 Listening on {HOST}:{PORT}")
print("=" * 55 + "\n")

try:
    while True:
        data = receiver.receive()
        if data is None:
            print("❌ Device disconnected.")
            break

        sample_count += 1
        curr_time = time.time()

        ax = float(data.get("ax", 0.0))
        ay = float(data.get("ay", 0.0))
        az = float(data.get("az", 9.81))
        gx = float(data.get("gx", 0.0))
        gy = float(data.get("gy", 0.0))
        gz = float(data.get("gz", 0.0))
        mx = float(data.get("mx", 0.0))
        my = float(data.get("my", 0.0))
        mz = float(data.get("mz", 0.0))
        timestamp = float(data.get("timestamp", curr_time * 1000.0))

        # Check manual outage toggle from dashboard
        gps_valid = bool(data.get("gps_valid", True))
        if os.path.exists(CONTROL_FILE):
            try:
                with open(CONTROL_FILE, "r") as f:
                    if json.load(f).get("force_outage", False):
                        gps_valid = False
            except Exception:
                pass

        # Update Orientation
        ori = orientation.update(ax, ay, az, gx, gy, gz, mx, my, mz, 0.05)
        yaw_deg = float(np.degrees(ori["yaw"]))
        yaw_rad = ori["yaw"]

        # Acceleration magnitude
        raw_mag = np.sqrt(ax**2 + ay**2 + az**2)
        acc_window.append(raw_mag)
        smooth_acc = float(np.mean(acc_window))

        # Dynamic variation from gravity (removes tilt dependence)
        dynamic_acc = smooth_acc - 9.81

        # ----------------------------------------------------
        # RELIABLE DYNAMIC STEP DETECTION
        # ----------------------------------------------------
        step_occurred = False
        motion_mode = "Stationary"

        # Rising edge (Foot strike impact)
        if not is_in_step:
            if dynamic_acc > 0.85 and (curr_time - last_step_timestamp) > step_cooldown:
                is_in_step = True
                step_occurred = True

                # Step duration for pacing
                if last_step_timestamp > 0:
                    step_duration = curr_time - last_step_timestamp
                else:
                    step_duration = 0.60
                last_step_timestamp = curr_time

                # Distinguish walk vs run
                if dynamic_acc > 3.8 or step_duration < 0.40:
                    motion_mode = "Running"
                    step_length = 0.85
                    current_speed = 4.5 / 3.6  # 4.5 km/h
                else:
                    motion_mode = "Walking"
                    step_length = 0.58
                    current_speed = 1.35 / 3.6 # 1.35 km/h

                step_count += 1
                total_distance += step_length

                # Advance coordinates along compass yaw
                x += step_length * np.sin(yaw_rad)
                y += step_length * np.cos(yaw_rad)

        # Falling edge (Foot swing reset)
        else:
            if dynamic_acc < 0.20:
                is_in_step = False

        # Speed decay when stationary
        if (curr_time - last_step_timestamp) > 1.3:
            current_speed = 0.0
            motion_mode = "Stationary"
        else:
            motion_mode = "Walking" if current_speed < 0.6 else "Running"

        latitude = INITIAL_LAT + (y / METERS_PER_DEG_LAT)
        lon_scale = METERS_PER_DEG_LAT * np.cos(np.radians(INITIAL_LAT))
        longitude = INITIAL_LON + (x / lon_scale)

        path.append([float(latitude), float(longitude)])
        if len(path) > 2500:
            path = path[-2500:]

        is_stationary = (current_speed == 0.0)

        save_live_state({
            "status": "LIVE",
            "timestamp": timestamp,
            "sample_count": sample_count,
            "step_count": step_count,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "x": float(x),
            "y": float(y),
            "speed": float(current_speed),
            "distance": float(total_distance),
            "is_stationary": bool(is_stationary),
            "motion_mode": motion_mode,
            "roll": float(np.degrees(ori["roll"])),
            "pitch": float(np.degrees(ori["pitch"])),
            "yaw": float(yaw_deg),
            "gps_valid": gps_valid,
            "path": path
        })

        if sample_count % 15 == 0:
            gps_lbl = "🟢 GPS:ON" if gps_valid else "🔴 AI-DR"
            print(f"[{sample_count:05d}] {motion_mode:<10s} | Steps: {step_count:3d} | "
                  f"Speed: {current_speed*3.6:3.1f} km/h | Dist: {total_distance:5.1f}m | {gps_lbl}")

except KeyboardInterrupt:
    print("\n🛑 Navigation engine stopped.")
finally:
    receiver.close()