import numpy as np


class SimpleEKF:

    def __init__(self):

        # State:
        # [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)

        # State covariance
        self.P = np.eye(6) * 0.1

        # Process noise
        self.Q = np.eye(6) * 0.01

        # GPS measurement noise
        self.R = np.eye(3) * 0.5

    # ------------------------------------------------
    # Prediction Step
    # ------------------------------------------------

    def predict(self, acceleration, dt):

        position = self.state[0:3]

        velocity = self.state[3:6]

        # Position update
        position = (
            position
            + velocity * dt
            + 0.5 * acceleration * dt * dt
        )

        # Velocity update
        velocity = (
            velocity
            + acceleration * dt
        )

        # Update state
        self.state[0:3] = position
        self.state[3:6] = velocity

        # Increase uncertainty
        self.P = self.P + self.Q

    # ------------------------------------------------
    # GPS Correction Step
    # ------------------------------------------------

    def update(self, measurement):

        # Measurement matrix
        H = np.zeros((3, 6))

        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        # Predicted GPS measurement
        predicted_measurement = (
            H @ self.state
        )

        # Measurement error
        error = (
            measurement
            - predicted_measurement
        )

        # Innovation covariance
        S = (
            H @ self.P @ H.T
            + self.R
        )

        # Kalman gain
        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        # Correct state
        self.state = (
            self.state
            + K @ error
        )

        # Update covariance
        I = np.eye(6)

        self.P = (
            (I - K @ H)
            @ self.P
        )

        return self.state.copy()

    # ------------------------------------------------
    # Get Position
    # ------------------------------------------------

    def get_position(self):

        return self.state[0:3]