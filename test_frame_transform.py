import numpy as np
from src.frame_transform import body_to_world, remove_gravity, transform_acceleration

def main():
    print("\nValidating frame transformations...")
    ax, ay, az = 0.0, 0.0, 9.81
    roll, pitch, yaw = 0.0, 0.0, 0.0

    linear_acc = transform_acceleration(ax, ay, az, roll, pitch, yaw)
    assert np.allclose(linear_acc, [0.0, 0.0, 0.0], atol=1e-5), "Gravity removal failed!"
    print("✅ Frame transformation and gravity removal passed.")

if __name__ == "__main__":
    main()