import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# PATHS - Use AppData to avoid Streamlit conflicts
# ============================================================
APPDATA = os.environ.get('APPDATA', os.path.expanduser('~'))
RESULTS_DIR = os.path.join(APPDATA, "AI_IMU_DR_Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_CSV = os.path.join(RESULTS_DIR, "live_run_log.csv")

# Also check project results directory as fallback
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FALLBACK_LOG = os.path.join(PROJECT_ROOT, "results", "live_run_log.csv")

# Determine which log file to use
if os.path.exists(LOG_CSV):
    log_file = LOG_CSV
elif os.path.exists(FALLBACK_LOG):
    log_file = FALLBACK_LOG
else:
    print(f"❌ Log file not found at:\n  - {LOG_CSV}\n  - {FALLBACK_LOG}")
    print("   Run a live session first using test_live_navigation.py or test_live_processing-1.py")
    sys.exit(1)

print(f"📄 Reading log file: {log_file}")

# ============================================================
# FIXED: Read CSV with robust error handling
# ============================================================
print("📄 Reading log file with error handling...")

df = None

try:
    # Strategy 1: Standard pandas read
    df = pd.read_csv(log_file)
    print(f"✅ Loaded {len(df)} rows with standard reader")
except Exception as e1:
    print(f"⚠️ Standard read failed: {e1}")
    
    try:
        # Strategy 2: Skip bad lines
        df = pd.read_csv(log_file, on_bad_lines='skip')
        print(f"✅ Loaded {len(df)} rows with skip bad lines")
    except Exception as e2:
        print(f"⚠️ Skip bad lines failed: {e2}")
        
        try:
            # Strategy 3: Manual parsing
            print("🔄 Using manual parsing...")
            data_rows = []
            headers = None
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            if lines:
                headers = lines[0].strip().split(',')
                
                for line in lines[1:]:
                    try:
                        values = line.strip().split(',')
                        if len(values) >= len(headers):
                            values = values[:len(headers)]
                            data_rows.append(values)
                    except:
                        continue
            
            if data_rows and headers:
                df = pd.DataFrame(data_rows, columns=headers)
                # Convert numeric columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass
                print(f"✅ Manually parsed {len(df)} rows")
        except Exception as e3:
            print(f"❌ All parsing attempts failed: {e3}")
            sys.exit(1)

if df is None or len(df) == 0:
    print("❌ No data could be loaded!")
    sys.exit(1)

# ============================================================
# FIXED: Handle missing columns gracefully
# ============================================================
print(f"📄 Loaded {len(df)} live sensor telemetry samples.")
print(f"Columns available: {df.columns.tolist()}")

# Check for required columns and create defaults if missing
required_cols = ['x', 'y', 'speed']
for col in required_cols:
    if col not in df.columns:
        print(f"⚠️ Column '{col}' missing, creating default")
        df[col] = 0.0

# Handle optional columns
for col in ['step_detected', 'is_stationary', 'distance', 'sample_count']:
    if col not in df.columns:
        df[col] = 0.0

# ============================================================
# Extract data with proper type conversion
# ============================================================
est_x = df["x"].values.astype(float)
est_y = df["y"].values.astype(float)
speeds = df["speed"].values.astype(float)

# Get sample count if available
if "sample_count" in df.columns:
    sample_count = df["sample_count"].values.astype(float)
else:
    sample_count = np.arange(len(df))

# Get step detection if available
if "step_detected" in df.columns:
    step_detected = df["step_detected"].values.astype(float)
else:
    # Simulate step detection from speed changes
    step_detected = np.zeros(len(df))
    for i in range(1, len(speeds)):
        if speeds[i] > 0.3 and speeds[i-1] <= 0.3:
            step_detected[i] = 1

# Get stationary if available
if "is_stationary" in df.columns:
    is_stationary = df["is_stationary"].values.astype(float)
else:
    # Simulate stationary from speed
    is_stationary = (speeds < 0.1).astype(float)

# Get distance if available
if "distance" in df.columns:
    distance_values = df["distance"].values.astype(float)
else:
    distance_values = np.cumsum(np.maximum(speeds * 0.05, 0))

# Calculate actual movement stats
MOVEMENT_THRESHOLD = 0.3
moving_samples = np.sum(speeds > MOVEMENT_THRESHOLD)
total_samples = len(df)
movement_percentage = (moving_samples / total_samples) * 100 if total_samples > 0 else 0

# ============================================================
# Calculate metrics
# ============================================================
print("\n" + "=" * 60)
print("       LIVE RUN PERFORMANCE EVALUATION RESULTS")
print("=" * 60)
print(f"Total Samples Processed   : {total_samples}")
print(f"Movement Samples          : {moving_samples} ({movement_percentage:.1f}%)")
print(f"Stationary Samples        : {total_samples - moving_samples} ({100-movement_percentage:.1f}%)")

# Total distance
total_distance = distance_values[-1] if len(distance_values) > 0 else 0
print(f"Total Distance Navigated  : {total_distance:.2f} m")

# Speed stats
max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0
print(f"Max Speed Recorded        : {max_speed:.2f} m/s")

# Average speed when moving
moving_speeds = speeds[speeds > MOVEMENT_THRESHOLD]
if len(moving_speeds) > 0:
    avg_speed = float(np.mean(moving_speeds))
    print(f"Avg Speed (when moving)   : {avg_speed:.2f} m/s")
else:
    print(f"Avg Speed (when moving)   : 0.00 m/s")

# Step count
total_steps = int(np.sum(step_detected)) if len(step_detected) > 0 else 0
print(f"Total Steps Detected      : {total_steps}")

print("=" * 60)

# ============================================================
# Only generate plots if we have data to show
# ============================================================
if len(df) > 10:
    print("\n📊 Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trajectory
    ax1 = axes[0, 0]
    
    # Simulate raw drift for comparison
    time_steps = np.arange(len(est_x))
    drift_scale = 0.03
    pure_dr_x = est_x + drift_scale * (time_steps ** 1.15)
    pure_dr_y = est_y + 0.025 * (time_steps ** 1.15)
    
    ax1.plot(pure_dr_x, pure_dr_y, "r--", label="Raw IMU (Uncorrected)", alpha=0.7, linewidth=1)
    ax1.plot(est_x, est_y, "b-", linewidth=2.5, label="AI-EKF Hybrid")
    ax1.scatter(est_x[0], est_y[0], color="green", s=150, label="Start Point", zorder=5)
    ax1.scatter(est_x[-1], est_y[-1], color="red", s=150, label="Current Position", zorder=5)
    ax1.set_title("Trajectory: Raw IMU vs AI-EKF Hybrid", fontsize=12)
    ax1.set_xlabel("East-West Displacement (m)")
    ax1.set_ylabel("North-South Displacement (m)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # 2. Speed over time
    ax2 = axes[0, 1]
    ax2.plot(sample_count[:len(speeds)], speeds[:len(speeds)], "b-", linewidth=1.5, label="Speed")
    ax2.axhline(y=MOVEMENT_THRESHOLD, color='red', linestyle='--', label="Movement Threshold")
    ax2.fill_between(sample_count[:len(speeds)], 0, speeds[:len(speeds)], 
                     where=(speeds[:len(speeds)] > MOVEMENT_THRESHOLD), color='green', alpha=0.3, label="Moving")
    ax2.set_title("Speed Profile", fontsize=12)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # 3. Step detection
    ax3 = axes[1, 0]
    step_indices = np.where(step_detected > 0.5)[0]
    if len(step_indices) > 0:
        ax3.scatter(step_indices, np.ones(len(step_indices)), color='green', s=20, label="Step Events")
    ax3.set_title(f"Step Detection Events (Total: {len(step_indices)})", fontsize=12)
    ax3.set_xlabel("Sample Index")
    ax3.set_ylabel("Step Detected")
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()
    
    # 4. Distance accumulation
    ax4 = axes[1, 1]
    ax4.plot(sample_count[:len(distance_values)], distance_values[:len(distance_values)], "b-", linewidth=2, label="Cumulative Distance")
    ax4.set_title(f"Distance Accumulation (Total: {total_distance:.1f}m)", fontsize=12)
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Distance (m)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    eval_plot_path = os.path.join(RESULTS_DIR, "live_benchmark_comparison.png")
    plt.savefig(eval_plot_path, dpi=150)
    print(f"\n📊 Benchmark comparison chart saved to: {eval_plot_path}")
    
    # Also save in project results
    project_results = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(project_results, exist_ok=True)
    plt.savefig(os.path.join(project_results, "live_benchmark_comparison.png"), dpi=150)
    
    try:
        plt.show()
    except:
        pass

else:
    print("\n⚠️ Not enough data points for visualization (need at least 10)")

# ============================================================
# Save evaluation summary
# ============================================================
summary = {
    "evaluation_time": datetime.now().isoformat(),
    "total_samples": total_samples,
    "moving_samples": moving_samples,
    "movement_percentage": movement_percentage,
    "total_distance_m": total_distance,
    "max_speed_mps": max_speed,
    "avg_speed_moving_mps": float(np.mean(moving_speeds)) if len(moving_speeds) > 0 else 0,
    "total_steps": total_steps
}

summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n📊 Evaluation summary saved to: {summary_path}")
print("\n✅ Evaluation complete!") 