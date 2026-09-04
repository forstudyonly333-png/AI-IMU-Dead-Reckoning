import numpy as np

class OrientationEstimator:
    def __init__(self, alpha=0.98):
        """
        Complementary Filter for IMU orientation tracking.
        alpha: Weight given to gyroscope integration (0.95 - 0.99 recommended).
        """
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.is_initialized = False

    def update(self, ax, ay, az, gx, gy, gz, mx, my, mz, dt):
        """
        Updates orientation using 9-DOF IMU data.
        dt: time step in seconds
        """
        # Pitch and Roll from Accelerometer
        acc_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        acc_roll = np.arctan2(ay, az)

        if not self.is_initialized:
            self.roll = acc_roll
            self.pitch = acc_pitch
            # Initial yaw from magnetometer tilt compensation
            mag_x = mx * np.cos(self.pitch) + mz * np.sin(self.pitch)
            mag_y = (mx * np.sin(self.roll) * np.sin(self.pitch) + 
                     my * np.cos(self.roll) - 
                     mz * np.sin(self.roll) * np.cos(self.pitch))
            self.yaw = np.arctan2(-mag_y, mag_x)
            self.is_initialized = True
            return {"roll": self.roll, "pitch": self.pitch, "yaw": self.yaw}

        # 1. Gyroscope Integration (rad/s * dt)
        self.roll += gx * dt
        self.pitch += gy * dt
        self.yaw += gz * dt

        # 2. Complementary Filter Correction (Roll & Pitch)
        self.roll = self.alpha * self.roll + (1.0 - self.alpha) * acc_roll
        self.pitch = self.alpha * self.pitch + (1.0 - self.alpha) * acc_pitch

        # 3. Tilt-Compensated Yaw from Magnetometer
        mag_x = mx * np.cos(self.pitch) + mz * np.sin(self.pitch)
        mag_y = (mx * np.sin(self.roll) * np.sin(self.pitch) + 
                 my * np.cos(self.roll) - 
                 mz * np.sin(self.roll) * np.cos(self.pitch))
        mag_yaw = np.arctan2(-mag_y, mag_x)

        # Fuse yaw with magnetometer
        self.yaw = self.alpha * self.yaw + (1.0 - self.alpha) * mag_yaw

        return {
            "roll": float(self.roll),
            "pitch": float(self.pitch),
            "yaw": float(self.yaw)
        }