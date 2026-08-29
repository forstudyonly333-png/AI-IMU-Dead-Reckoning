import numpy as np


def dead_reckoning(df):

    n = len(df)

    position = np.zeros((n, 3))
    velocity = np.zeros((n, 3))

    for i in range(1, n):

        dt = df.iloc[i]["dt"]

        acceleration = np.array([
            df.iloc[i]["ax"],
            df.iloc[i]["ay"],
            df.iloc[i]["az"] - 9.81
        ])

        # Velocity integration
        velocity[i] = (
            velocity[i - 1]
            + acceleration * dt
        )

        # Position integration
        position[i] = (
            position[i - 1]
            + velocity[i] * dt
        )

    return position, velocity