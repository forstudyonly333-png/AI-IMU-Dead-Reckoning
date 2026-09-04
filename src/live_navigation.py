import numpy as np

class LiveNavigator:
    def __init__(self, stationary_threshold=0.35):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.distance = 0.0
        self.step_count = 0
        self.is_walking = False
        self.is_stationary = True
        self.latitude = 26.785892
        self.longitude = 75.818937
        self.path = []
        self.heading = 0.0
        self.movement_mode = "Stationary"
        self.sample_count = 0
        self._step_timer = 0.0
        self.stationary_threshold = stationary_threshold
        self.prev_x = 0.0
        self.prev_y = 0.0

    def initialize(self, latitude=None, longitude=None):
        if latitude is not None:
            self.latitude = latitude
        if longitude is not None:
            self.longitude = longitude
        return self

    def update(self, ax, ay, az, dt=0.05, heading=0.0):
        self.sample_count += 1
        self.heading = heading

        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        acc_diff = abs(acc_mag - 9.81)

        # Strict stationary check (ZUPT)
        if acc_diff < self.stationary_threshold:
            self.is_stationary = True
            self.is_walking = False
            self.movement_mode = "Stationary"
            self.speed = 0.0
            self.vx = 0.0
            self.vy = 0.0
            self._step_timer = 0.0
        else:
            self.is_stationary = False
            self.is_walking = True
            self._step_timer += dt

            # Vehicle / Walking step-rate governor
            if self._step_timer >= 0.40:
                self._step_timer = 0.0
                self.step_count += 1
                
                # Speed proportional to dynamic shock energy
                if acc_diff > 2.0:
                    self.speed = 1.6  # running / accelerating vehicle
                    self.movement_mode = "Fast Motion"
                else:
                    self.speed = 0.9  # moderate cruising / walking
                    self.movement_mode = "Steady Motion"

                heading_rad = np.radians(heading)
                step_dist = self.speed * 0.40
                self.x += step_dist * np.sin(heading_rad)
                self.y += step_dist * np.cos(heading_rad)
                self.distance += step_dist

        # Map local displacement to WGS84 Geodetic Coordinates
        self.latitude = 26.785892 + (self.y / 111320.0)
        self.longitude = 75.818937 + (self.x / (111320.0 * np.cos(np.radians(26.785892))))

        self.path.append([float(self.latitude), float(self.longitude)])
        if len(self.path) > 2000:
            self.path = self.path[-2000:]

        return self.get_state()

    def get_state(self):
        return {
            "x": float(self.x),
            "y": float(self.y),
            "speed": float(self.speed),
            "speed_kmh": float(self.speed * 3.6),
            "distance": float(self.distance),
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "heading": float(self.heading),
            "step_count": int(self.step_count),
            "is_walking": bool(self.is_walking),
            "is_stationary": bool(self.is_stationary),
            "path": self.path,
            "movement_mode": self.movement_mode,
            "sample_count": int(self.sample_count),
            "gps_valid": self.sample_count > 10
        }