import json
import math
import os

import numpy as np

from src.orientation import OrientationEstimator
from src.frame_transform import transform_acceleration


class LiveNavigationEngine:
    """
    Real-time smartphone IMU navigation engine.

    Flow:
        IMU
        -> Orientation
        -> Gravity removal
        -> Stationary detection / ZUPT
        -> Velocity integration
        -> Position integration
        -> Local X/Y -> Latitude/Longitude
    """

    def __init__(self):
        # --------------------------------------------------
        # Orientation
        # --------------------------------------------------

        self.orientation = OrientationEstimator(alpha=0.98)

        # --------------------------------------------------
        # Navigation state
        # --------------------------------------------------

        self.initialized = False

        self.x = 0.0
        self.y = 0.0

        self.vx = 0.0
        self.vy = 0.0

        self.last_timestamp = None

        self.start_lat = None
        self.start_lon = None

        # --------------------------------------------------
        # Acceleration filtering
        # --------------------------------------------------

        self.accel_filter_x = 0.0
        self.accel_filter_y = 0.0

        self.filter_alpha = 0.20

        # --------------------------------------------------
        # Stationary / ZUPT
        # --------------------------------------------------

        self.stationary_count = 0

        # Slightly stricter than before.
        self.stationary_threshold = 0.30

        self.stationary_required = 5

        # Ignore small acceleration.
        self.acceleration_deadband = 0.20

        # Maximum pedestrian speed.
        self.max_speed = 2.5

        # --------------------------------------------------
        # Gyro threshold
        # --------------------------------------------------

        self.gyro_threshold = 0.20

        # Gravity tolerance.
        self.gravity_tolerance = 0.50

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.distance_m = 0.0
        self.sample_count = 0
        self.last_speed = 0.0

    # ======================================================
    # INITIALIZE FROM GPS
    # ======================================================

    def initialize(self, latitude, longitude):
        """Set starting GPS position and reset state."""

        self.start_lat = float(latitude)
        self.start_lon = float(longitude)

        self.x = 0.0
        self.y = 0.0

        self.vx = 0.0
        self.vy = 0.0

        self.distance_m = 0.0

        self.last_timestamp = None

        self.accel_filter_x = 0.0
        self.accel_filter_y = 0.0

        self.stationary_count = 0

        self.sample_count = 0
        self.last_speed = 0.0

        self.initialized = True

        print("\n========================================")
        print("📍 LIVE NAVIGATION INITIALIZED")
        print("========================================")
        print(f"Start Latitude : {self.start_lat:.7f}")
        print(f"Start Longitude: {self.start_lon:.7f}")
        print("")

    # ======================================================
    # LOCAL XY -> LAT/LON
    # ======================================================

    def xy_to_latlon(self):
        """Convert local displacement to latitude/longitude."""

        if self.start_lat is None or self.start_lon is None:
            return None, None

        meters_per_degree_lat = 111111.0

        meters_per_degree_lon = (
            111111.0
            * math.cos(math.radians(self.start_lat))
        )

        latitude = (
            self.start_lat
            + self.y / meters_per_degree_lat
        )

        longitude = (
            self.start_lon
            + self.x / meters_per_degree_lon
        )

        return latitude, longitude

    # ======================================================
    # PROCESS ONE IMU PACKET
    # ======================================================

    def process_packet(self, data):
        """Process one smartphone IMU packet."""

        if not self.initialized:
            return None

        self.sample_count += 1

        # --------------------------------------------------
        # Read IMU
        # --------------------------------------------------

        try:
            ax = float(data.get("ax", 0.0))
            ay = float(data.get("ay", 0.0))
            az = float(data.get("az", 0.0))

            gx = float(data.get("gx", 0.0))
            gy = float(data.get("gy", 0.0))
            gz = float(data.get("gz", 0.0))

            mx = float(data.get("mx", 0.0))
            my = float(data.get("my", 0.0))
            mz = float(data.get("mz", 0.0))

            timestamp = float(data.get("timestamp", 0.0))

        except (TypeError, ValueError):
            return None

        # --------------------------------------------------
        # Reject invalid sensor values
        # --------------------------------------------------

        values = [
            ax, ay, az,
            gx, gy, gz,
            mx, my, mz,
            timestamp,
        ]

        if not all(math.isfinite(v) for v in values):
            return None

        # --------------------------------------------------
        # Calculate dt
        # --------------------------------------------------

        if self.last_timestamp is None:

            dt = 0.02

        else:

            dt = (
                timestamp - self.last_timestamp
            ) / 1000.0

            if not math.isfinite(dt):
                dt = 0.02

            dt = max(0.005, min(dt, 0.10))

        self.last_timestamp = timestamp

        # --------------------------------------------------
        # Orientation
        # --------------------------------------------------

        orientation_result = self.orientation.update(
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
            mx,
            my,
            mz,
            dt,
        )

        roll = orientation_result["roll"]
        pitch = orientation_result["pitch"]
        yaw = orientation_result["yaw"]

        # --------------------------------------------------
        # Body -> World + gravity removal
        # --------------------------------------------------

        world_accel = transform_acceleration(
            ax,
            ay,
            az,
            roll,
            pitch,
            yaw,
        )

        wx = float(world_accel[0])
        wy = float(world_accel[1])
        wz = float(world_accel[2])

        # --------------------------------------------------
        # Low-pass filter
        # --------------------------------------------------

        self.accel_filter_x = (
            self.filter_alpha * wx
            + (1.0 - self.filter_alpha)
            * self.accel_filter_x
        )

        self.accel_filter_y = (
            self.filter_alpha * wy
            + (1.0 - self.filter_alpha)
            * self.accel_filter_y
        )

        ax_world = self.accel_filter_x
        ay_world = self.accel_filter_y

        # --------------------------------------------------
        # Horizontal acceleration
        # --------------------------------------------------

        horizontal_accel = math.hypot(
            ax_world,
            ay_world,
        )

        # --------------------------------------------------
        # Acceleration deadband
        # --------------------------------------------------

        if horizontal_accel < self.acceleration_deadband:

            ax_world = 0.0
            ay_world = 0.0
            horizontal_accel = 0.0

        # --------------------------------------------------
        # Raw accelerometer gravity check
        # --------------------------------------------------

        accel_norm = math.sqrt(
            ax * ax
            + ay * ay
            + az * az
        )

        gravity_error = abs(
            accel_norm - 9.81
        )

        # --------------------------------------------------
        # Gyroscope quiet check
        # --------------------------------------------------

        gyro_magnitude = math.sqrt(
            gx * gx
            + gy * gy
            + gz * gz
        )

        gyro_quiet = (
            gyro_magnitude
            < self.gyro_threshold
        )

        # --------------------------------------------------
        # Stationary detection
        # --------------------------------------------------

        stationary_sample = (
            horizontal_accel
            < self.stationary_threshold
            and gravity_error
            < self.gravity_tolerance
            and gyro_quiet
        )

        if stationary_sample:
            self.stationary_count += 1
        else:
            self.stationary_count = 0

        stationary = (
            self.stationary_count
            >= self.stationary_required
        )

        # --------------------------------------------------
        # IMPORTANT:
        # During startup, don't allow random IMU noise
        # to immediately create movement.
        # --------------------------------------------------

        if self.sample_count <= self.stationary_required:
            stationary = True

        # --------------------------------------------------
        # Acceleration safety limit
        # --------------------------------------------------

        ax_world = float(
            np.clip(
                ax_world,
                -5.0,
                5.0,
            )
        )

        ay_world = float(
            np.clip(
                ay_world,
                -5.0,
                5.0,
            )
        )

        # --------------------------------------------------
        # ZUPT
        # --------------------------------------------------

        if stationary:

            self.vx = 0.0
            self.vy = 0.0

            ax_world = 0.0
            ay_world = 0.0

            self.last_speed = 0.0

        else:

            # --------------------------------------------------
            # Velocity integration
            # --------------------------------------------------

            self.vx += ax_world * dt
            self.vy += ay_world * dt

            # --------------------------------------------------
            # Velocity damping
            # --------------------------------------------------

            self.vx *= 0.995
            self.vy *= 0.995

            # --------------------------------------------------
            # Speed limit
            # --------------------------------------------------

            speed = math.hypot(
                self.vx,
                self.vy,
            )

            if speed > self.max_speed:

                scale = (
                    self.max_speed
                    / speed
                )

                self.vx *= scale
                self.vy *= scale

        # --------------------------------------------------
        # Position integration
        # --------------------------------------------------

        old_x = self.x
        old_y = self.y

        if not stationary:

            self.x += self.vx * dt
            self.y += self.vy * dt

        # --------------------------------------------------
        # Distance
        # --------------------------------------------------

        if stationary:

            step_distance = 0.0

        else:

            step_distance = math.hypot(
                self.x - old_x,
                self.y - old_y,
            )

        self.distance_m += step_distance

        # --------------------------------------------------
        # Current speed
        # --------------------------------------------------

        self.last_speed = math.hypot(
            self.vx,
            self.vy,
        )

        self.last_speed = min(
            self.last_speed,
            self.max_speed,
        )

        # --------------------------------------------------
        # Lat/Lon
        # --------------------------------------------------

        latitude, longitude = (
            self.xy_to_latlon()
        )

        # --------------------------------------------------
        # GPS status
        # --------------------------------------------------

        gps_valid = bool(
            data.get(
                "gps_valid",
                False,
            )
        )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        return {
            "timestamp": timestamp,

            "x": self.x,
            "y": self.y,

            "vx": self.vx,
            "vy": self.vy,

            "speed": self.last_speed,

            "distance_m": self.distance_m,

            "latitude": latitude,
            "longitude": longitude,

            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),

            "ax_world": ax_world,
            "ay_world": ay_world,
            "az_world": wz,

            "stationary": stationary,

            "gps_valid": gps_valid,
        }

    # ======================================================
    # SAVE CURRENT STATE
    # ======================================================

    def save_state(
        self,
        result,
        filename="results/live_state.json",
    ):
        """Save latest navigation state safely."""

        if result is None:
            return

        directory = os.path.dirname(filename)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        temp_file = filename + ".tmp"

        with open(
            temp_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )

            f.flush()
            os.fsync(f.fileno())

        try:
            os.replace(
                temp_file,
                filename,
            )
        except PermissionError:
            # If Streamlit has the file open momentarily,
            # leave the previous valid state intact.
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass