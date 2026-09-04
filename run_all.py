import json
import os
import socket
import sys
import threading
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.live_state import save_live_state
from src.orientation import OrientationEstimator

HOST = "0.0.0.0"
PORT = 5000

INITIAL_LAT = 26.7858633
INITIAL_LON = 75.8190191
METERS_PER_DEG_LAT = 111320.0

CONTROL_FILE = os.path.join(PROJECT_ROOT, "results", "outage_override.json")
server_ready = threading.Event()
stop_event = threading.Event()

def engine_worker():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_sock.bind((HOST, PORT))
    except Exception as e:
        print(f"❌ Socket bind error: {e}")
        return

    server_sock.listen(5)
    server_sock.settimeout(1.0)
    server_ready.set()

    print(f"📡 Engine listening on 0.0.0.0:{PORT}")

    orientation = OrientationEstimator(alpha=0.98)
    sample_count = 0
    step_count = 0
    x, y = 0.0, 0.0
    total_distance = 0.0
    current_speed = 0.0
    last_step_time = 0.0
    path = []
    acc_history = []

    while not stop_event.is_set():
        try:
            conn, addr = server_sock.accept()
            print(f"\n✅ Device connected from: {addr[0]}:{addr[1]}")
        except socket.timeout:
            continue
        except Exception:
            break

        conn.settimeout(0.5)
        buffer = ""

        while not stop_event.is_set():
            try:
                data = conn.recv(2048).decode("utf-8")
                if not data:
                    break
                buffer += data
            except socket.timeout:
                continue
            except Exception:
                break

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue

                try:
                    telemetry = json.loads(line)
                except Exception:
                    continue

                sample_count += 1
                curr_time = time.time()

                ax = float(telemetry.get("ax", 0.0))
                ay = float(telemetry.get("ay", 0.0))
                az = float(telemetry.get("az", 9.81))
                gx = float(telemetry.get("gx", 0.0))
                gy = float(telemetry.get("gy", 0.0))
                gz = float(telemetry.get("gz", 0.0))
                mx = float(telemetry.get("mx", 0.0))
                my = float(telemetry.get("my", 0.0))
                mz = float(telemetry.get("mz", 0.0))
                timestamp = float(telemetry.get("timestamp", curr_time * 1000.0))

                # Check manual outage
                forced_outage = False
                if os.path.exists(CONTROL_FILE):
                    try:
                        with open(CONTROL_FILE, "r") as f:
                            forced_outage = json.load(f).get("force_outage", False)
                    except Exception:
                        pass

                gps_valid = not forced_outage

                ori = orientation.update(ax, ay, az, gx, gy, gz, mx, my, mz, 0.05)
                yaw_deg = float(np.degrees(ori["yaw"]))
                yaw_rad = ori["yaw"]

                acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
                acc_history.append(acc_mag)
                if len(acc_history) > 4:
                    acc_history.pop(0)
                smooth_acc = np.mean(acc_history)

                step_detected = False
                motion_mode = "Stationary"

                if (curr_time - last_step_time) > 0.32:
                    if smooth_acc > 13.5:
                        step_detected = True
                        step_len = 0.80
                        speed_kmh = 4.5
                        motion_mode = "Running"
                        last_step_time = curr_time
                    elif smooth_acc > 10.7:
                        step_detected = True
                        step_len = 0.52
                        speed_kmh = 1.3
                        motion_mode = "Walking"
                        last_step_time = curr_time

                if step_detected:
                    step_count += 1
                    current_speed = speed_kmh / 3.6
                    total_distance += step_len
                    x += step_len * np.sin(yaw_rad)
                    y += step_len * np.cos(yaw_rad)
                else:
                    if (curr_time - last_step_time) > 1.2:
                        current_speed = 0.0
                        motion_mode = "Stationary"
                    else:
                        motion_mode = "Walking" if current_speed < 0.6 else "Running"

                lat = INITIAL_LAT + (y / METERS_PER_DEG_LAT)
                lon_scale = METERS_PER_DEG_LAT * np.cos(np.radians(INITIAL_LAT))
                lon = INITIAL_LON + (x / lon_scale)

                path.append([float(lat), float(lon)])
                if len(path) > 2500:
                    path = path[-2500:]

                save_live_state({
                    "status": "LIVE",
                    "timestamp": timestamp,
                    "sample_count": sample_count,
                    "step_count": step_count,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "x": float(x),
                    "y": float(y),
                    "speed": float(current_speed),
                    "distance": float(total_distance),
                    "is_stationary": bool(current_speed == 0.0),
                    "motion_mode": motion_mode,
                    "roll": float(np.degrees(ori["roll"])),
                    "pitch": float(np.degrees(ori["pitch"])),
                    "yaw": float(yaw_deg),
                    "gps_valid": gps_valid,
                    "path": path
                })

                if sample_count % 20 == 0:
                    status_lbl = "STATIONARY" if current_speed == 0.0 else motion_mode.upper()
                    gps_lbl = "🟢 GPS:ON" if gps_valid else "🔴 OUTAGE"
                    print(f"[{sample_count:05d}] {status_lbl:<10s} | {gps_lbl:<13s} | Speed: {current_speed*3.6:4.1f} km/h | Dist: {total_distance:6.2f}m")

        conn.close()
    server_sock.close()

if __name__ == "__main__":
    print("=" * 65)
    print("🛰️ STARTING REAL SMARTPHONE PDR RECEIVER ENGINE")
    print("=" * 65)

    t_engine = threading.Thread(target=engine_worker, daemon=True)
    t_engine.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
        stop_event.set()
        time.sleep(0.5)