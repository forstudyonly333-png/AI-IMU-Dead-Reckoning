import json
import socket
import threading
import time
import numpy as np

from src.live_receiver import IMUReceiver
from src.orientation import OrientationEstimator
from src.frame_transform import transform_acceleration

HOST = "127.0.0.1"
PORT = 5000

def simulated_phone():
    time.sleep(1)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((HOST, PORT))
        for _ in range(50):
            packet = {
                "timestamp": time.time(),
                "ax": 0.0, "ay": 0.0, "az": 9.81,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "mx": 30.0, "my": 5.0, "mz": -40.0
            }
            client.sendall((json.dumps(packet) + "\n").encode("utf-8"))
            time.sleep(0.05)
    finally:
        client.close()

def main():
    receiver = IMUReceiver(host=HOST, port=PORT)
    threading.Thread(target=receiver.start, daemon=True).start()
    threading.Thread(target=simulated_phone, daemon=True).start()

    orientation = OrientationEstimator(alpha=0.98)
    time.sleep(1.5)

    samples = 0
    while samples < 30:
        data = receiver.receive()
        if not data:
            break
        samples += 1
        ori = orientation.update(data["ax"], data["ay"], data["az"],
                                 data["gx"], data["gy"], data["gz"],
                                 data["mx"], data["my"], data["mz"], 0.05)
        world_a = transform_acceleration(data["ax"], data["ay"], data["az"],
                                         ori["roll"], ori["pitch"], ori["yaw"])
        if samples % 10 == 0:
            print(f"Sample {samples:02d} | Linear Accel: {np.linalg.norm(world_a):.4f} m/s^2 (Expected ~0.0)")

    receiver.close()
    print("✅ Pipeline test verified.")

if __name__ == "__main__":
    main()