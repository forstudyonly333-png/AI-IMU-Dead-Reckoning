import json
import socket
import time
import sys
import numpy as np

HOST = "127.0.0.1"
PORT = 5000

WALKING_SPEED_RANGE = (1.5, 3.0)
RUNNING_SPEED_RANGE = (3.0, 5.0)
WALK_DURATION = 15
RUN_DURATION = 10
REST_DURATION = 10
CYCLE_DURATION = WALK_DURATION + RUN_DURATION + REST_DURATION

def generate_gait_pattern(t, mode, step_freq):
    step_phase = 2 * np.pi * step_freq * t
    if mode == "walking":
        az = 9.81 + 0.9 * np.sin(step_phase)
        ay = 0.2 * np.cos(step_phase)
        ax = 0.15 * np.sin(step_phase)
    elif mode == "running":
        az = 9.81 + 2.2 * np.sin(step_phase)
        ay = 0.5 * np.cos(step_phase)
        ax = 0.3 * np.sin(step_phase)
    else:
        az = 9.81
        ay = 0.0
        ax = 0.0
    
    ax += np.random.normal(0, 0.01)
    ay += np.random.normal(0, 0.01)
    az += np.random.normal(0, 0.01)
    return ax, ay, az

def main():
    print("=" * 60)
    print("🚶 TELEMETRY SIMULATOR STARTING...")
    print(f"📍 Target address: {HOST}:{PORT}")
    print("=" * 60)

    client = None
    max_retries = 10
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Connection attempt {attempt}/{max_retries}...")
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(3.0)
            client.connect((HOST, PORT))
            print(f"✅ Successfully connected to navigation engine at {HOST}:{PORT}!\n")
            break
        except Exception as err:
            print(f"   ⚠️ Could not connect: {err}")
            client = None
            time.sleep(1.5)

    if not client:
        print("\n❌ Failed to connect after multiple retries.")
        print("👉 Make sure 'python test_live_engine.py' is already running in another terminal window!")
        return

    start_time = time.time()
    base_lat, base_lon = 26.7858924, 75.8189368
    pos_x, pos_y = 0.0, 0.0
    total_distance = 0.0
    sample_idx = 0

    try:
        while True:
            t = time.time() - start_time
            sample_idx += 1
            cycle_time = t % CYCLE_DURATION

            if cycle_time < WALK_DURATION:
                current_mode = "walking"
                speed_kmh = 2.5
                step_freq = 1.0
                is_moving = True
            elif cycle_time < WALK_DURATION + RUN_DURATION:
                current_mode = "running"
                speed_kmh = 4.2
                step_freq = 1.8
                is_moving = True
            else:
                current_mode = "stationary"
                speed_kmh = 0.0
                step_freq = 0.0
                is_moving = False

            if is_moving:
                speed_ms = speed_kmh / 3.6
                step_length = speed_ms / step_freq
                ax, ay, az = generate_gait_pattern(t, current_mode, step_freq)
                gx = np.random.normal(0, 0.005)
                gy = np.random.normal(0, 0.005)
                gz = 0.02 * np.sin(0.1 * t)
                heading_deg = 15.0

                pos_x += step_length * np.sin(np.radians(heading_deg)) * (0.05 * step_freq)
                pos_y += step_length * np.cos(np.radians(heading_deg)) * (0.05 * step_freq)
                total_distance += speed_ms * 0.05
            else:
                speed_ms = 0.0
                ax = np.random.normal(0, 0.005)
                ay = np.random.normal(0, 0.005)
                az = 9.81 + np.random.normal(0, 0.005)
                gx = 0.0
                gy = 0.0
                gz = 0.0
                heading_deg = 0.0

            packet = {
                "timestamp": int(time.time() * 1000),
                "ax": float(ax), "ay": float(ay), "az": float(az),
                "gx": float(gx), "gy": float(gy), "gz": float(gz),
                "mx": 25.0, "my": 5.0, "mz": -35.0,
                "heading": float(heading_deg),
                "lat": float(base_lat + (pos_y / 111320.0)),
                "lon": float(base_lon + (pos_x / 111320.0)),
                "speed": float(speed_ms),
                "gps_valid": is_moving
            }

            try:
                client.sendall((json.dumps(packet) + "\n").encode("utf-8"))
            except Exception as send_err:
                print(f"\n❌ Lost connection to server: {send_err}")
                break

            if sample_idx % 20 == 0:
                print(f"[{t:05.1f}s] {current_mode.upper():<10s} | Speed: {speed_kmh:4.1f} km/h | Dist: {total_distance:5.1f}m")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Simulator stopped by user.")
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    main()