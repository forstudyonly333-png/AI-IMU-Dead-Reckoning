"""
ekf.py

Extended Kalman Filter for position/velocity/heading estimation.

FIXES vs original version:
  1. The original state was [x, y, z, vx, vy, vz] and `predict()` only
     took raw acceleration -- it never used the gyroscope, so it was
     really a plain linear Kalman Filter wearing an "Extended" label.
     It implicitly assumed the vehicle's body frame never rotates.
  2. State is now [x, y, z, vx, vy, vz, theta] (heading included).
     `predict()` takes both acceleration AND yaw rate, rotates the
     horizontal acceleration into the world frame using the current
     heading estimate, and propagates heading. Because the rotation
     depends on the state itself (theta), the transition is now
     genuinely non-linear -- this is what makes it an "Extended" KF
     (its Jacobian, F, is computed and used to propagate covariance).
  3. GPS measurement noise R is configurable, so it can be set to
     match the noise actually injected into the simulated GPS
     (previously GPS was treated as a perfect, noise-free reading).
"""

import numpy as np


class SimpleEKF:

    def __init__(self, gps_noise_std=3.0):
        # State: [x, y, z, vx, vy, vz, theta]
        self.state = np.zeros(7)

        # State covariance
        self.P = np.eye(7) * 0.1

        # Process noise (heading gets a smaller process-noise term)
        self.Q = np.eye(7) * 0.01
        self.Q[6, 6] = 0.001

        # GPS measurement noise -- should match the real/simulated
        # GPS accuracy (std in metres). Default 3.0 m ~= typical
        # consumer GPS accuracy.
        self.R = np.eye(3) * (gps_noise_std ** 2)

    # ------------------------------------------------
    # Prediction Step (non-linear: rotation depends on theta)
    # ------------------------------------------------
    def predict(self, acceleration, yaw_rate, dt):
        x, y, z, vx, vy, vz, theta = self.state

        ax, ay, az = acceleration

        # Rotate body-frame horizontal acceleration into world frame
        ax_w = ax * np.cos(theta) - ay * np.sin(theta)
        ay_w = ax * np.sin(theta) + ay * np.cos(theta)
        az_w = az

        # State propagation
        x_new = x + vx * dt + 0.5 * ax_w * dt * dt
        y_new = y + vy * dt + 0.5 * ay_w * dt * dt
        z_new = z + vz * dt + 0.5 * az_w * dt * dt

        vx_new = vx + ax_w * dt
        vy_new = vy + ay_w * dt
        vz_new = vz + az_w * dt

        theta_new = theta + yaw_rate * dt

        self.state = np.array(
            [x_new, y_new, z_new, vx_new, vy_new, vz_new, theta_new]
        )

        # Jacobian of the transition w.r.t. state (only theta column
        # is non-trivial, since ax_w/ay_w depend on theta).
        F = np.eye(7)

        d_axw_dtheta = -ax * np.sin(theta) - ay * np.cos(theta)
        d_ayw_dtheta = ax * np.cos(theta) - ay * np.sin(theta)

        F[0, 6] = 0.5 * d_axw_dtheta * dt * dt   # dx/dtheta
        F[1, 6] = 0.5 * d_ayw_dtheta * dt * dt   # dy/dtheta
        F[3, 6] = d_axw_dtheta * dt              # dvx/dtheta
        F[4, 6] = d_ayw_dtheta * dt              # dvy/dtheta

        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------
    # GPS Correction Step (position-only measurement)
    # ------------------------------------------------
    def update(self, measurement):
        H = np.zeros((3, 7))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        predicted_measurement = H @ self.state
        error = measurement - predicted_measurement

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ error

        I = np.eye(7)
        self.P = (I - K @ H) @ self.P

        return self.state[0:3].copy()

    # ------------------------------------------------
    # Zero-Velocity Update (ZUPT)
    # ------------------------------------------------
    def zero_velocity_update(self):
        """
        Call this when the vehicle is known/detected to be stationary
        (e.g. stopped at a signal). Directly resets the velocity
        estimate to zero, which prevents accelerometer noise from
        integrating into a fake drift while the vehicle isn't moving.
        This was completely absent before.
        """
        self.state[3:6] = 0.0
        self.P[3:6, :] *= 0.1
        self.P[:, 3:6] *= 0.1

    # ------------------------------------------------
    # Get Position
    # ------------------------------------------------
    def get_position(self):
        return self.state[0:3].copy()

    def get_heading(self):
        return self.state[6]