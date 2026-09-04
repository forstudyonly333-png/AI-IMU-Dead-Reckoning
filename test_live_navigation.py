import os
import sys
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class LiveNavigator:
    def __init__(self, stationary_threshold=0.35):
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.distance = 0.0
        self.step_count = 0
        self.is_walking = False
        self.is_stationary = True
        self.latitude = 26.785892
        self.longitude = 75.818937
        self.path = []
        self.heading = 0.0
        self.movement_mode = "Stationary"
        self.sample_count = 0
        self._step_timer = 0.0
        self.stationary_threshold = stationary_threshold

    def initialize(self, latitude=None, longitude=None):
        if latitude is not None:
            self.latitude = latitude
        if longitude is not None:
            self.longitude = longitude
        return self

    def update(self, ax, ay, az, dt=0.05, heading=0.0):
        self.sample_count += 1
        self.heading = heading

        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        acc_diff = abs(acc_mag - 9.81)

        # Detect motion only when above stationary threshold
        if acc_diff > self.stationary_threshold:
            self.is_stationary = False
            self.is_walking = True
            self._step_timer += dt

            # Minimum time between steps to prevent false positive bursts
            if self._step_timer >= 0.40:
                self._step_timer = 0.0
                self.step_count += 1

                # Dynamic step length estimation
                step_len = 0.65 if acc_diff < 1.8 else 0.85
                self.speed = step_len / 0.45
                self.movement_mode = "Running" if acc_diff > 1.8 else "Walking"

                heading_rad = np.radians(heading)
                dx = step_len * np.sin(heading_rad)
                dy = step_len * np.cos(heading_rad)
                self.x += dx
                self.y += dy
                self.distance += step_len
        else:
            # Full ZUPT reset
            self.is_stationary = True
            self.is_walking = False
            self.movement_mode = "Stationary"
            self.speed = 0.0
            self._step_timer = 0.0

        # Update geographic coordinates
        self.latitude = 26.785892 + (self.y / 111320.0)
        self.longitude = 75.818937 + (self.x / (111320.0 * np.cos(np.radians(26.785892))))

        self.path.append([self.latitude, self.longitude])
        if len(self.path) > 2000:
            self.path = self.path[-2000:]

        return self.get_state()

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "speed_kmh": self.speed * 3.6,
            "distance": self.distance,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "step_count": self.step_count,
            "is_walking": self.is_walking,
            "is_stationary": self.is_stationary,
            "path": self.path,
            "movement_mode": self.movement_mode,
            "sample_count": self.sample_count,
            "gps_valid": self.sample_count > 10
        }

def simulate_walking_data(duration=10, sample_rate=20):
    data = []
    dt = 1.0 / sample_rate
    samples = int(duration * sample_rate)
    for i in range(samples):
        t = i * dt
        step_phase = 2 * np.pi * 1.8 * t
        az = 9.81 + 2.2 * np.sin(step_phase)
        ay = 0.5 * np.cos(step_phase)
        ax = 0.4 * np.sin(step_phase)
        data.append((ax, ay, az, dt))
    return data

def simulate_stationary_data(duration=5, sample_rate=20):
    data = []
    dt = 1.0 / sample_rate
    samples = int(duration * sample_rate)
    for _ in range(samples):
        ax = np.random.normal(0, 0.02)
        ay = np.random.normal(0, 0.02)
        az = 9.81 + np.random.normal(0, 0.02)
        data.append((ax, ay, az, dt))
    return data

def main():
    print("\n============================================================")
    print("🚶 LIVE NAVIGATION TEST (WITH PROPER DRIFT REJECTION)")
    print("============================================================\n")

    nav = LiveNavigator(stationary_threshold=0.35)
    nav.initialize(latitude=26.9124, longitude=75.7873)

    print("🚶 Testing Walking (10s)...")
    for i, (ax, ay, az, dt) in enumerate(simulate_walking_data(10)):
        nav.update(ax, ay, az, dt, heading=0.0)
    print(f"Dist after walk: {nav.distance:.2f}m (Expected > 0)")

    print("\n🧍 Testing Stationary (5s)... (Distance MUST NOT grow)")
    dist_before = nav.distance
    for i, (ax, ay, az, dt) in enumerate(simulate_stationary_data(5)):
        nav.update(ax, ay, az, dt, heading=0.0)
    dist_after = nav.distance
    print(f"Dist before: {dist_before:.2f}m | Dist after: {dist_after:.2f}m")

    assert np.isclose(dist_before, dist_after), "❌ Failed: Distance increased while stationary!"
    print("✅ Stationary drift rejection working perfectly.")

if __name__ == "__main__":
    main()