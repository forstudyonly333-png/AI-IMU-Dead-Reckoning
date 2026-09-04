"""
dead_reckoning.py

Physics-based dead reckoning.

FIX vs original version:
  The original code integrated raw accelerometer readings directly,
  treating the body frame as if it were always aligned with the world
  frame (i.e. it assumed the vehicle never turns). This ignored the
  gyroscope entirely.

  A real vehicle turns, so the accelerometer's x/y axes rotate with it.
  This version integrates the yaw rate (gz) to track a heading angle,
  then rotates the horizontal acceleration into the world frame BEFORE
  integrating velocity/position. This is what the PPT's "Calibration &
  Alignment" / heading-estimation claims actually require.

  Also vectorized (no more `df.iloc[i]` row-by-row loop for the
  cumulative sums), which matters for real-time / on-device performance.
"""

import numpy as np


def dead_reckoning(df):
    n = len(df)

    dt = df["dt"].values
    ax = df["ax"].values
    ay = df["ay"].values
    az = df["az"].values - 9.81
    gz = df["gz"].values

    # ------------------------------------------------------------
    # 1. Integrate yaw rate -> heading (this is the piece that was
    #    completely missing before).
    # ------------------------------------------------------------
    heading = np.concatenate(([0.0], np.cumsum(gz[:-1] * dt[:-1])))

    # ------------------------------------------------------------
    # 2. Rotate body-frame acceleration into the world frame using
    #    the estimated heading.
    # ------------------------------------------------------------
    ax_world = ax * np.cos(heading) - ay * np.sin(heading)
    ay_world = ax * np.sin(heading) + ay * np.cos(heading)
    az_world = az

    # ------------------------------------------------------------
    # 3. Integrate world-frame acceleration -> velocity -> position.
    # ------------------------------------------------------------
    velocity = np.zeros((n, 3))
    position = np.zeros((n, 3))

    vx = np.concatenate(([0.0], np.cumsum(ax_world[:-1] * dt[:-1])))
    vy = np.concatenate(([0.0], np.cumsum(ay_world[:-1] * dt[:-1])))
    vz = np.concatenate(([0.0], np.cumsum(az_world[:-1] * dt[:-1])))

    velocity[:, 0] = vx
    velocity[:, 1] = vy
    velocity[:, 2] = vz

    position[:, 0] = np.concatenate(([0.0], np.cumsum(vx[:-1] * dt[:-1])))
    position[:, 1] = np.concatenate(([0.0], np.cumsum(vy[:-1] * dt[:-1])))
    position[:, 2] = np.concatenate(([0.0], np.cumsum(vz[:-1] * dt[:-1])))

    return position, velocity