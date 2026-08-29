import numpy as np
import pandas as pd


def generate_demo_data(seconds=60, dt=0.01):
    """
    Generates a physically consistent synthetic IMU dataset.

    Sampling rate: 100 Hz
    Duration: 60 seconds

    Ground truth trajectory is generated first.
    Velocity and acceleration are derived from that trajectory.
    IMU noise and small sensor bias are then added.
    """

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------

    time = np.arange(
        0,
        seconds,
        dt
    )

    n = len(time)

    # ---------------------------------------------------------
    # Ground Truth Trajectory
    # ---------------------------------------------------------

    # Forward motion with smooth walking-like oscillation
    speed = 0.8
    frequency_x = 0.18

    x = (
        speed * time
        - (speed / frequency_x)
        * np.sin(frequency_x * time)
    )

    # Side-to-side movement
    amplitude_y = 2.0
    frequency_y = 0.15

    y = (
        amplitude_y
        * (1 - np.cos(frequency_y * time))
    )

    z = np.zeros(n)

    # ---------------------------------------------------------
    # Ground Truth Velocity
    # ---------------------------------------------------------

    vx = (
        speed
        * (
            1
            - np.cos(frequency_x * time)
        )
    )

    vy = (
        amplitude_y
        * frequency_y
        * np.sin(frequency_y * time)
    )

    vz = np.zeros(n)

    # ---------------------------------------------------------
    # Ground Truth Acceleration
    # ---------------------------------------------------------

    ax_true = (
        speed
        * frequency_x
        * np.sin(frequency_x * time)
    )

    ay_true = (
        amplitude_y
        * frequency_y**2
        * np.cos(frequency_y * time)
    )

    az_true = np.zeros(n)

    # ---------------------------------------------------------
    # IMU Sensor Noise
    # ---------------------------------------------------------

    rng = np.random.default_rng(42)

    acceleration_noise = 0.03
    gyro_noise = 0.005

    # Small accelerometer bias
    bias_x = 0.015
    bias_y = -0.01
    bias_z = 0.008

    ax = (
        ax_true
        + bias_x
        + rng.normal(
            0,
            acceleration_noise,
            n
        )
    )

    ay = (
        ay_true
        + bias_y
        + rng.normal(
            0,
            acceleration_noise,
            n
        )
    )

    # Gravity + vertical acceleration
    az = (
        9.81
        + az_true
        + bias_z
        + rng.normal(
            0,
            acceleration_noise,
            n
        )
    )

    # ---------------------------------------------------------
    # Gyroscope
    # ---------------------------------------------------------

    gx = rng.normal(
        0,
        gyro_noise,
        n
    )

    gy = rng.normal(
        0,
        gyro_noise,
        n
    )

    gz = (
        0.02
        * np.sin(0.3 * time)
        + rng.normal(
            0,
            gyro_noise,
            n
        )
    )

    # ---------------------------------------------------------
    # Create DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame({

        "timestamp": time,

        "ax": ax,
        "ay": ay,
        "az": az,

        "gx": gx,
        "gy": gy,
        "gz": gz,

        # Ground truth
        "gt_x": x,
        "gt_y": y,
        "gt_z": z,

        "gt_vx": vx,
        "gt_vy": vy,
        "gt_vz": vz,

        "gt_ax": ax_true,
        "gt_ay": ay_true,
        "gt_az": az_true
    })

    return df


# -------------------------------------------------------------
# Test dataset generation
# -------------------------------------------------------------

if __name__ == "__main__":

    df = generate_demo_data()

    print(df.head())

    print(
        "\nShape:",
        df.shape
    )

    print(
        "\nColumns:"
    )

    print(
        df.columns.tolist()
    )

    df.to_csv(
        "data/processed/demo_imu.csv",
        index=False
    )

    print(
        "\nDemo IMU dataset created!"
    )