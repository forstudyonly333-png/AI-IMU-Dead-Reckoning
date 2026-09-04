import time
import numpy as np

from src.live_receiver import IMUReceiver
from src.orientation import OrientationEstimator
from src.live_navigation import LiveNavigator

receiver = IMUReceiver(host="0.0.0.0", port=5000)
orientation = OrientationEstimator(alpha=0.98)
navigator = LiveNavigator(stationary_threshold=0.35)

receiver.start()

print("\n" + "=" * 50)
print("🚀 LIVE PHONE NAVIGATION TEST")
print("=" * 50)
print("Listening on 0.0.0.0:5000...")
print("-" * 50)

previous_timestamp = None
sample_count = 0

try:
    while True:
        data = receiver.receive()
        if data is None:
            print("❌ Disconnected")
            break

        sample_count += 1

        ax = float(data.get("ax", 0.0))
        ay = float(data.get("ay", 0.0))
        az = float(data.get("az", 9.81))
        gx = float(data.get("gx", 0.0))
        gy = float(data.get("gy", 0.0))
        gz = float(data.get("gz", 0.0))
        mx = float(data.get("mx", 0.0))
        my = float(data.get("my", 0.0))
        mz = float(data.get("mz", 0.0))
        timestamp = float(data.get("timestamp", time.time() * 1000))

        if previous_timestamp is None:
            dt = 0.02
        else:
            dt = max(0.001, min((timestamp - previous_timestamp) / 1000.0, 0.2))
        previous_timestamp = timestamp

        ori = orientation.update(ax, ay, az, gx, gy, gz, mx, my, mz, dt)
        yaw_deg = np.degrees(ori["yaw"])

        state = navigator.update(ax, ay, az, dt=dt, heading=yaw_deg)

        if sample_count % 20 == 0:
            print(f"[{sample_count:05d}] {state['movement_mode']:<10s} | "
                  f"Speed: {state['speed_kmh']:5.2f} km/h | "
                  f"Dist: {state['distance']:6.2f} m | "
                  f"Yaw: {yaw_deg:6.1f}°")

except KeyboardInterrupt:
    print("\n🛑 Test stopped.")
finally:
    receiver.close()
    print("✅ Finished.")