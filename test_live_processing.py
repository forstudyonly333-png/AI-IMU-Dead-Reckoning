import json
import os
import sys
import time
import numpy as np
import socket
import threading
import csv
from datetime import datetime

# ============================================================
# USE APPDATA FOLDER - NOT WATCHED BY STREAMLIT
# ============================================================
APPDATA = os.environ.get('APPDATA', os.path.expanduser('~'))
RESULTS_DIR = os.path.join(APPDATA, "AI_IMU_DR_Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LIVE_STATE_PATH = os.path.join(RESULTS_DIR, "live_state.json")
LIVE_LOG_CSV = os.path.join(RESULTS_DIR, "live_run_log.csv")

print(f"📁 Results directory: {RESULTS_DIR}")

# ============================================================
# IMU RECEIVER - IMPROVED
# ============================================================
class IMUReceiver:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.data_queue = []
        self.buffer = ""
        self.connected = False
        self.client_address = None
        self._lock = threading.Lock()
        
    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)
            self.running = True
            
            print(f"📡 Listening on {self.host}:{self.port}...")
            print("   (Press Ctrl+C to stop)\n")
            
            def accept_loop():
                while self.running:
                    try:
                        if not self.connected:
                            print("⏳ Waiting for connection...")
                            self.client_socket, addr = self.server_socket.accept()
                            self.client_socket.settimeout(0.5)
                            self.connected = True
                            self.client_address = addr
                            print(f"✅ Connected to {addr}")
                            print("📊 Receiving data...\n")
                        
                        if self.client_socket:
                            try:
                                data = self.client_socket.recv(8192).decode('utf-8')
                                if data:
                                    self.buffer += data
                                    while '\n' in self.buffer:
                                        line, self.buffer = self.buffer.split('\n', 1)
                                        if line.strip():
                                            try:
                                                with self._lock:
                                                    self.data_queue.append(json.loads(line))
                                            except json.JSONDecodeError as e:
                                                pass
                                else:
                                    # Client disconnected
                                    print("⚠️ Client disconnected - waiting for reconnection...")
                                    self.connected = False
                                    self.client_socket = None
                            except socket.timeout:
                                pass
                            except ConnectionResetError:
                                print("⚠️ Connection reset - waiting for reconnection...")
                                self.connected = False
                                self.client_socket = None
                            except Exception as e:
                                print(f"⚠️ Connection error: {e}")
                                self.connected = False
                                self.client_socket = None
                    except socket.timeout:
                        pass
                    except Exception as e:
                        print(f"⚠️ Accept error: {e}")
                        time.sleep(0.1)
            
            threading.Thread(target=accept_loop, daemon=True).start()
            return True
        except Exception as e:
            print(f"❌ Failed to start receiver: {e}")
            return False
        
    def receive(self, timeout=0.5):
        """Receive data with timeout"""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self.data_queue:
                    return self.data_queue.pop(0)
            time.sleep(0.01)
        return None
    
    def has_data(self):
        with self._lock:
            return len(self.data_queue) > 0
    
    def get_queue_size(self):
        with self._lock:
            return len(self.data_queue)
    
    def close(self):
        self.running = False
        if self.client_socket:
            try: 
                self.client_socket.close()
            except: 
                pass
        if self.server_socket:
            try: 
                self.server_socket.close()
            except: 
                pass
        print("📡 Receiver closed")

# ============================================================
# LIVE NAVIGATOR
# ============================================================
class LiveNavigator:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
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
        self._step_timer = 0
        self._log_data = []
        self._origin_lat = 26.785892
        self._origin_lon = 75.818937
        self._last_heading = 0.0
        
    def initialize(self, latitude=None, longitude=None):
        if latitude is not None:
            self.latitude = latitude
            self._origin_lat = latitude
        if longitude is not None:
            self.longitude = longitude
            self._origin_lon = longitude
        print(f"📍 Initialized at: ({self.latitude:.6f}, {self.longitude:.6f})")
        return self
        
    def update(self, ax, ay, az, dt=0.05, heading=0.0, gps_valid=False, speed=0.0):
        self.sample_count += 1
        self.heading = heading
        self._last_heading = heading
        
        # Detect movement from acceleration magnitude
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        acc_diff = abs(acc_mag - 9.81)
        
        step_detected = 0
        
        # Use speed from simulator if provided
        if speed > 0.1:
            self.is_walking = True
            self.is_stationary = False
            self.movement_mode = "Moving"
            self.speed = speed
            
            # Update position using heading
            heading_rad = np.radians(heading)
            self.x += speed * dt * np.sin(heading_rad)
            self.y += speed * dt * np.cos(heading_rad)
            self.distance += speed * dt
            
            # Step detection
            self._step_timer += dt
            if self._step_timer > 0.4:
                self._step_timer = 0
                self.step_count += 1
                step_detected = 1
        else:
            # Use acceleration-based detection
            if acc_diff > 0.5:
                self.is_walking = True
                self.is_stationary = False
                self.movement_mode = "Walking"
                
                self._step_timer += dt
                if self._step_timer > 0.4:
                    self._step_timer = 0
                    self.step_count += 1
                    step_detected = 1
                    
                    step_len = 0.5 + np.random.random() * 0.2
                    self.speed = step_len / 0.4
                    
                    heading_rad = np.radians(heading)
                    self.x += step_len * np.sin(heading_rad)
                    self.y += step_len * np.cos(heading_rad)
                    self.distance += step_len
            else:
                self.is_walking = False
                self.is_stationary = True
                self.movement_mode = "Stationary"
                self.speed = 0.0
        
        # Update lat/lon
        self.latitude = self._origin_lat + (self.y / 111320.0)
        self.longitude = self._origin_lon + (self.x / (111320.0 * np.cos(np.radians(self._origin_lat))))
        
        # Path
        self.path.append([self.latitude, self.longitude])
        if len(self.path) > 1000:
            self.path = self.path[-1000:]
        
        # Log data for CSV
        self._log_data.append({
            'timestamp': time.time(),
            'sample_count': self.sample_count,
            'x': self.x,
            'y': self.y,
            'speed': self.speed,
            'distance': self.distance,
            'step_count': self.step_count,
            'step_detected': step_detected,
            'heading': heading,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'movement_mode': self.movement_mode,
            'is_stationary': 1 if self.is_stationary else 0,
            'is_walking': 1 if self.is_walking else 0,
            'gps_valid': 1 if gps_valid else 0,
            'ax': ax,
            'ay': ay,
            'az': az
        })
        
        return self.get_state()
        
    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "speed_kmh": self.speed * 3.6,
            "distance": self.distance,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "step_count": self.step_count,
            "is_walking": self.is_walking,
            "is_stationary": self.is_stationary,
            "path": self.path,
            "movement_mode": self.movement_mode,
            "sample_count": self.sample_count,
            "gps_valid": self.sample_count > 10
        }
        
    def get_log_data(self):
        return self._log_data
    
    def get_summary(self):
        return {
            "total_distance": self.distance,
            "step_count": self.step_count,
            "final_position": (self.x, self.y),
            "final_lat_lon": (self.latitude, self.longitude),
            "total_samples": self.sample_count
        }

# ============================================================
# SAVE FUNCTIONS
# ============================================================
def save_state(nav):
    """Save current state to JSON"""
    state = nav.get_state()
    state["origin_lat"] = 26.785892
    state["origin_lon"] = 75.818937
    state["timestamp"] = time.time()
    state["status"] = "LIVE"
    
    try:
        with open(LIVE_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ Could not save state: {e}")
        return False

def save_log_csv(nav):
    """Save log data to CSV"""
    log_data = nav.get_log_data()
    if not log_data:
        return False
    
    try:
        import pandas as pd
        df = pd.DataFrame(log_data)
        df.to_csv(LIVE_LOG_CSV, index=False)
        return True
    except Exception as e:
        print(f"⚠️ Could not save log: {e}")
        return False

# ============================================================
# MAIN
# ============================================================
def main():
    # Initialize receiver
    receiver = IMUReceiver(host="0.0.0.0", port=5000)
    if not receiver.start():
        print("❌ Failed to start receiver")
        return

    # Initialize navigator
    nav = LiveNavigator()
    nav.initialize(latitude=26.785892, longitude=75.818937)

    print("\n" + "=" * 60)
    print("🚗 LIVE DEAD RECKONING ENGINE")
    print("=" * 60)
    print(f"📂 Logging to: {RESULTS_DIR}")
    print("📡 Waiting for IMU data on port 5000...")
    print("-" * 60)

    sample_count = 0
    last_save_time = time.time()
    last_print_time = time.time()
    no_data_time = 0
    connected_once = False

    try:
        while True:
            # Try to receive data with timeout
            data = receiver.receive(timeout=0.1)
            
            if data is None:
                no_data_time += 0.1
                if no_data_time > 2.0 and not connected_once:
                    print("⏳ Waiting for data... (run simulate_smartphone.py in another terminal)")
                    connected_once = True
                    no_data_time = 0
                time.sleep(0.01)
                continue
            
            connected_once = True
            no_data_time = 0
            sample_count += 1
            
            # Extract IMU data
            ax = float(data.get("ax", 0.0))
            ay = float(data.get("ay", 0.0))
            az = float(data.get("az", 9.81))
            heading = float(data.get("heading", 0.0))
            gps_valid = data.get("gps_valid", False)
            speed = float(data.get("speed", 0.0))
            
            # Update navigation
            dt = 0.05
            state = nav.update(ax, ay, az, dt=dt, heading=heading, 
                             gps_valid=gps_valid, speed=speed)
            
            # Periodic save
            if sample_count % 10 == 0 or time.time() - last_save_time > 5:
                save_state(nav)
                save_log_csv(nav)
                last_save_time = time.time()
            
            # Print status
            if sample_count % 20 == 0 or time.time() - last_print_time > 2:
                speed_kmh = state.get("speed_kmh", 0)
                mode = state.get("movement_mode", "Unknown")
                steps = state.get("step_count", 0)
                dist = state.get("distance", 0)
                gps_status = "🟢" if gps_valid else "🔴"
                queue_size = receiver.get_queue_size()
                print(f"[{sample_count:05d}] {mode:12s} | "
                      f"Speed: {speed_kmh:5.2f} km/h | "
                      f"Steps: {steps:4d} | "
                      f"Dist: {dist:6.1f}m | "
                      f"GPS: {gps_status} | "
                      f"Q: {queue_size:3d}")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final save
        save_state(nav)
        save_log_csv(nav)
        receiver.close()
        
        # Print summary
        summary = nav.get_summary()
        print("\n" + "=" * 60)
        print("📊 SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Samples    : {summary['total_samples']}")
        print(f"Total Distance   : {summary['total_distance']:.2f}m")
        print(f"Total Steps      : {summary['step_count']}")
        print(f"Final Position   : ({summary['final_position'][0]:.2f}, {summary['final_position'][1]:.2f})m")
        print(f"Final Lat/Lon    : ({summary['final_lat_lon'][0]:.7f}, {summary['final_lat_lon'][1]:.7f})")
        print("=" * 60)
        print(f"📂 Results saved in: {RESULTS_DIR}")
        print(f"   - {LIVE_STATE_PATH}")
        print(f"   - {LIVE_LOG_CSV}")
        print("\n✅ Live processing complete!")

if __name__ == "__main__":
    main()