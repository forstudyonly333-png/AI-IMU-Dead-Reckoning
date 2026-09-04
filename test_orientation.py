import numpy as np

from src.orientation import OrientationEstimator


def main():

    estimator = OrientationEstimator(alpha=0.98)

    # Simulated stationary smartphone
    ax = 0.0
    ay = 0.0
    az = 9.81

    # No rotation
    gx = 0.0
    gy = 0.0
    gz = 0.0

    # Example magnetic field
    mx = 30.0
    my = 5.0
    mz = -40.0

    dt = 0.01

    for _ in range(100):

        orientation = estimator.update(
            ax, ay, az,
            gx, gy, gz,
            mx, my, mz,
            dt
        )

    print("\nOrientation test successful")
    print("--------------------------------")

    angles = estimator.get_orientation_degrees()

    print(f"Roll  : {angles['roll']:.2f}°")
    print(f"Pitch : {angles['pitch']:.2f}°")
    print(f"Yaw   : {angles['yaw']:.2f}°")


if __name__ == "__main__":
    main()