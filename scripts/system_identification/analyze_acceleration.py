#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "rosbags",
# ]
# python = ">=3.10"
# ///
#
# Usage:
#   uv run scripts/system_identification/analyze_acceleration.py input.mcap
#
# Description:
#   MCAPファイルから加速度（速度の変化率）の最大値や分布を解析します。

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

def extract_velocity_data(mcap_path):
    print(f"Extracting data from {mcap_path}...")
    
    vel_times = []
    vel_vals = []
    
    vel_topic = "/localization/kinematic_state"
    
    with AnyReader([Path(mcap_path)]) as reader:
        connections = [x for x in reader.connections if x.topic == vel_topic]
        
        if not connections:
            print(f"No connections found for topic: {vel_topic}")
            return np.array([]), np.array([])
        
        # Check standard message type, might need adjustment if using different types
        # Usually nav_msgs/msg/Odometry or similar
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            t = timestamp / 1e9
            try:
                # localized kinematic state: twist.twist.linear.x
                val = msg.twist.twist.linear.x
                vel_times.append(t)
                vel_vals.append(val)
            except AttributeError:
                pass
                    
    return np.array(vel_times), np.array(vel_vals)

def analyze_acceleration(times, vals, output_dir, base_name):
    if len(times) < 2:
        print("Not enough data points.")
        return

    # Calculate derivative (acceleration)
    dt = np.diff(times)
    d_val = np.diff(vals)
    
    # Filter out zero time differences
    valid_mask = dt > 1e-6
    accel = d_val[valid_mask] / dt[valid_mask]
    accel_times = times[:-1][valid_mask]
    
    abs_accel = np.abs(accel)
    
    # Statistics
    max_accel = np.max(abs_accel)
    p99_accel = np.percentile(abs_accel, 99.9) 
    p99_normal = np.percentile(abs_accel, 99.0)
    mean_accel = np.mean(abs_accel)
    
    print("\n" + "="*40)
    print("ACCELERATION ANALYSIS")
    print("="*40)
    print(f"Max Absolute Accel:      {max_accel:.4f} m/s^2")
    print(f"99.9%ile Accel:          {p99_accel:.4f} m/s^2")
    print(f"99.0%ile Accel:          {p99_normal:.4f} m/s^2")
    print(f"Mean Absolute Accel:     {mean_accel:.4f} m/s^2")
    print("-" * 40)
    print(f"Suggested `max_acceleration` limit: ~{np.ceil(p99_accel):.0f} or {np.ceil(p99_accel*10)/10:.1f} m/s^2")
    
    # Plotting
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    
    # Velocity
    ax1.plot(times - times[0], vals, label='Velocity', color='orange')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity vs Time')
    ax1.grid(True)
    
    # Acceleration
    ax2.plot(accel_times - times[0], accel, label='Acceleration', color='blue')
    ax2.set_ylabel('Accel (m/s^2)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Acceleration vs Time')
    ax2.grid(True)
    ax2.axhline(y=max_accel, color='r', linestyle='--', label='Max')
    ax2.axhline(y=-max_accel, color='r', linestyle='--')
    ax2.legend()
    
    # Histogram
    ax3.hist(abs_accel, bins=50, log=True)
    ax3.set_xlabel('Absolute Acceleration (m/s^2)')
    ax3.set_ylabel('Count (Log Scale)')
    ax3.set_title('Acceleration Distribution')
    ax3.grid(True)
    ax3.axvline(x=p99_accel, color='g', linestyle='--', label='99.9%ile')
    ax3.legend()
    
    plt.tight_layout()
    output_png = output_dir / f"{base_name}_accel.png"
    plt.savefig(output_png)
    print(f"\nPlot saved to {output_png}")

def main():
    parser = argparse.ArgumentParser(description="Analyze acceleration limits.")
    parser.add_argument("file", help="Input MCAP file")
    
    args = parser.parse_args()
    mcap_path = args.file
    
    times, vals = extract_velocity_data(mcap_path)
    
    if len(times) == 0:
        print("No velocity data extracted.")
        return
        
    base_name = Path(mcap_path).stem
    output_dir = Path(__file__).parent / "results"
    
    analyze_acceleration(times, vals, output_dir, base_name)

if __name__ == "__main__":
    main()
