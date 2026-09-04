"""
data_loader.py

Generates a physically-consistent synthetic IMU + GPS dataset for a
GROUND VEHICLE (car), not a pedestrian.

FIXES vs original version:
  1. `seed` is now a parameter (was hardcoded to 42 inside the function).
     -> Lets train.py and test.py generate DIFFERENT trajectories, so the
        model is evaluated on data it has never seen (fixes the
        "train == test" data-leakage bug).
  2. Realistic vehicle speed profile (accelerate -> cruise -> stop ->
     turn -> cruise) instead of a 0.8 m/s "walking pace" profile.
  3. Motion is generated from a HEADING angle (theta), so the vehicle
     only ever moves in the direction it is pointing. This satisfies the
     Non-Holonomic Constraint (NHC) that the PPT claims to use -- the
     original generator moved sideways (y-axis) independent of heading,
     which a real car physically cannot do.
  4. Accelerometer bias is now a slow random walk instead of a fixed
     constant for all 60 seconds, which is closer to real IMU bias
     instability.
  5. Includes stationary ("stopped at a signal") segments so the
     pipeline has a realistic case for zero-velocity handling.
"""

import numpy as np
import pandas as pd


def generate_demo_data(seconds=60, dt=0.01, seed=42, cruise_speed_kmph=45):
    """
    Parameters
    ----------
    seconds : total simulated duration
    dt      : sample interval (100 Hz default)
    seed    : RNG seed. IMPORTANT -> use a DIFFERENT seed for train vs
              test so the model is validated on an unseen trajectory.
    cruise_speed_kmph : target cruising speed of the vehicle in km/h.
              Default 45 km/h is a realistic urban driving speed
              (the original 0.8 m/s ~= 2.9 km/h was pedestrian pace).
    """

    rng = np.random.default_rng(seed)

    time = np.arange(0, seconds, dt)
    n = len(time)

    cruise_speed = cruise_speed_kmph / 3.6  # km/h -> m/s

    # ---------------------------------------------------------
    # 1. Speed profile: accelerate -> cruise -> stop -> cruise
    #    (a real car does not move at constant speed for 60s)
    # ---------------------------------------------------------
    speed_profile = np.full(n, cruise_speed)

    # accelerate smoothly for the first 8 seconds
    accel_phase = time < 8
    speed_profile[accel_phase] = cruise_speed * (time[accel_phase] / 8.0)

    # simulate a stop at a signal between 28s-33s (tests zero-velocity)
    stop_mask = (time >= 28) & (time < 33)
    ramp_down = (time >= 25) & (time < 28)
    ramp_up = (time >= 33) & (time < 36)
    speed_profile[stop_mask] = 0.0
    speed_profile[ramp_down] = cruise_speed * (1 - (time[ramp_down] - 25) / 3.0)
    speed_profile[ramp_up] = cruise_speed * ((time[ramp_up] - 33) / 3.0)

    is_stationary = stop_mask.copy()

    # ---------------------------------------------------------
    # 2. Heading profile: gentle S-curve turns (yaw rate), so the
    #    car actually steers instead of sliding sideways.
    # ---------------------------------------------------------
    yaw_rate = 0.06 * np.sin(0.12 * time) + 0.03 * np.sin(0.4 * time)
    yaw_rate[is_stationary] = 0.0  # no turning while stopped

    heading = np.concatenate(([0.0], np.cumsum(yaw_rate[:-1]) * dt))

    # ---------------------------------------------------------
    # 3. Integrate heading + speed -> ground-truth (x, y) path.
    #    Motion is ALWAYS along the heading direction only
    #    (this is what "Non-Holonomic" means for a car).
    # ---------------------------------------------------------
    vx = speed_profile * np.cos(heading)
    vy = speed_profile * np.sin(heading)
    vz = np.zeros(n)

    x = np.concatenate(([0.0], np.cumsum(vx[:-1]) * dt))
    y = np.concatenate(([0.0], np.cumsum(vy[:-1]) * dt))
    z = np.zeros(n)

    # ground-truth acceleration (world frame), via numerical derivative
    ax_true_w = np.gradient(vx, dt)
    ay_true_w = np.gradient(vy, dt)
    az_true_w = np.zeros(n)

    # ---------------------------------------------------------
    # 4. Convert world-frame acceleration into BODY-frame
    #    accelerometer readings (this is what a real IMU measures).
    #    body_ax =  ax_w*cos(theta) + ay_w*sin(theta)
    #    body_ay = -ax_w*sin(theta) + ay_w*cos(theta)
    # ---------------------------------------------------------
    ax_body_true = ax_true_w * np.cos(heading) + ay_true_w * np.sin(heading)
    ay_body_true = -ax_true_w * np.sin(heading) + ay_true_w * np.cos(heading)
    az_body_true = az_true_w

    # ---------------------------------------------------------
    # 5. Sensor noise + SLOWLY DRIFTING bias (random walk), not a
    #    fixed constant -> closer to real accelerometer behaviour.
    # ---------------------------------------------------------
    acceleration_noise = 0.03
    gyro_noise = 0.005
    bias_walk_std = 0.0006  # how fast the bias wanders per sample

    bias_x = np.cumsum(rng.normal(0, bias_walk_std, n)) + 0.015
    bias_y = np.cumsum(rng.normal(0, bias_walk_std, n)) - 0.010
    bias_z = np.cumsum(rng.normal(0, bias_walk_std, n)) + 0.008

    ax = ax_body_true + bias_x + rng.normal(0, acceleration_noise, n)
    ay = ay_body_true + bias_y + rng.normal(0, acceleration_noise, n)
    az = 9.81 + az_body_true + bias_z + rng.normal(0, acceleration_noise, n)

    # ---------------------------------------------------------
    # 6. Gyroscope: z-axis carries the real yaw rate + noise.
    # ---------------------------------------------------------
    gx = rng.normal(0, gyro_noise, n)
    gy = rng.normal(0, gyro_noise, n)
    gz = yaw_rate + rng.normal(0, gyro_noise, n)

    df = pd.DataFrame({
        "timestamp": time,

        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,

        # Ground truth (used ONLY for evaluation, never fed to the model)
        "gt_x": x, "gt_y": y, "gt_z": z,
        "gt_vx": vx, "gt_vy": vy, "gt_vz": vz,
        "gt_heading": heading,
        "is_stationary": is_stationary,
    })

    return df


if __name__ == "__main__":
    df = generate_demo_data(seed=42)
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMax speed (km/h):",
          (np.hypot(df['gt_vx'], df['gt_vy']).max()) * 3.6)
    df.to_csv("data/processed/demo_imu.csv", index=False)
    print("\nDemo IMU dataset created!")