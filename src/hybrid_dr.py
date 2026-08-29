import numpy as np


def hybrid_position(
    gps_position,
    ekf_position,
    ai_position,
    gps_available
):
    """
    Combines GPS, EKF and AI-DR estimates.

    GPS available:
        GPS position is trusted.

    GPS unavailable:
        AI-DR and EKF are blended.

    Returns:
        Hybrid estimated trajectory
    """

    n = len(gps_position)

    hybrid = np.zeros(
        (n, 3)
    )

    for i in range(n):

        if gps_available[i]:

            # GPS available
            hybrid[i] = (
                0.7 * gps_position[i]
                + 0.3 * ekf_position[i]
            )

        else:

            # GPS outage
            hybrid[i] = (
                0.6 * ai_position[i]
                + 0.4 * ekf_position[i]
            )

    return hybrid