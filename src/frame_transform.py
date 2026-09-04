import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Computes body-to-world rotation matrix using standard Tait-Bryan angles (Z-Y-X sequence).
    Angles must be in radians.
    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined Rotation Matrix (R_z * R_y * R_x)
    return R_z @ R_y @ R_x

def body_to_world(ax, ay, az, roll, pitch, yaw):
    """Rotates body-frame acceleration vector to the world reference frame."""
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    body_acc = np.array([ax, ay, az], dtype=float)
    return R @ body_acc

def remove_gravity(world_acc, g=9.81):
    """
    Subtracts earth gravity vector [0, 0, g] from the world-frame acceleration.
    """
    linear_acc = np.copy(world_acc)
    linear_acc[2] -= g
    return linear_acc

def transform_acceleration(ax, ay, az, roll, pitch, yaw, g=9.81):
    """
    Full pipeline: rotates body acceleration to world frame and extracts net linear acceleration.
    """
    world_acc = body_to_world(ax, ay, az, roll, pitch, yaw)
    linear_acc = remove_gravity(world_acc, g=g)
    return linear_acc