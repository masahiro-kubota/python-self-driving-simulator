#!/usr/bin/env python3
#
# Usage:
#   uv run scripts/system_identification/analyze_steering_rate.py scripts/system_identification/input.mcap
#
# Description:
#   MCAPファイルからステアリング角速度（変化率）の最大値や分布を解析します。

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.utils.mcap_utils import read_messages

def extract_steering_data(mcap_path):
    print(f"Extracting data from {mcap_path}...")
    
    status_times = []
    status_vals = []
    
    status_topic = "/vehicle/status/steering_status"
    
    for topic, msg, timestamp_ns in read_messages(mcap_path, [status_topic]):
        t = timestamp_ns / 1e9
        try:
            val = msg.steering_tire_angle
            status_times.append(t)
            status_vals.append(val)
        except AttributeError:
             pass
                    
    return np.array(status_times), np.array(status_vals)

def analyze_rate(times, vals, output_dir, base_name):
    if len(times) < 2:
        print("Not enough data points.")
        return

    # Calculate derivative
    dt = np.diff(times)
    d_val = np.diff(vals)
    
    # Filter out zero time differences to avoid division by zero
    valid_mask = dt > 1e-6
    rate = d_val[valid_mask] / dt[valid_mask]
    rate_times = times[:-1][valid_mask]
    
    abs_rate = np.abs(rate)
    
    # Statistics
    max_rate = np.max(abs_rate)
    p99_rate = np.percentile(abs_rate, 99.9) # 99.9%ile to exclude extreme impulse noise
    p99_normal = np.percentile(abs_rate, 99.0)
    mean_rate = np.mean(abs_rate)
    
    # Convert from rad/s to deg/s for easier reading
    max_rate_deg = np.rad2deg(max_rate)
    p99_rate_deg = np.rad2deg(p99_rate)
    p99_normal_deg = np.rad2deg(p99_normal)
    
    print("\n" + "="*40)
    print("STEERING RATE ANALYSIS")
    print("="*40)
    print(f"Max Absolute Rate:       {max_rate:.4f} rad/s ({max_rate_deg:.2f} deg/s)")
    print(f"99.9%ile Rate (Noise-free est.): {p99_rate:.4f} rad/s ({p99_rate_deg:.2f} deg/s)")
    print(f"99.0%ile Rate:           {p99_normal:.4f} rad/s ({p99_normal_deg:.2f} deg/s)")
    print(f"Mean Absolute Rate:      {mean_rate:.4f} rad/s")
    print("-" * 40)
    p99_rate_ceil = np.ceil(p99_rate * 10) / 10.0 # Round up to 1st decimal place in rad/s
    p99_rate_deg_conv = np.rad2deg(p99_rate_ceil)
    print(f"Suggested `max_steer_rate` limit: {p99_rate_ceil:.3f} rad/s (~{p99_rate_deg_conv:.1f} deg/s)")
    
    # Plotting
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time series
    ax1.plot(rate_times - rate_times[0], np.rad2deg(rate), label='Steering Rate')
    ax1.set_ylabel('Rate (deg/s)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Steering Rate vs Time')
    ax1.grid(True)
    ax1.axhline(y=max_rate_deg, color='r', linestyle='--', label='Max')
    ax1.axhline(y=-max_rate_deg, color='r', linestyle='--')
    ax1.legend()
    
    # Histogram
    ax2.hist(np.rad2deg(abs_rate), bins=50, log=True)
    ax2.set_xlabel('Absolute Rate (deg/s)')
    ax2.set_ylabel('Count (Log Scale)')
    ax2.set_title('Steering Rate Distribution')
    ax2.grid(True)
    ax2.axvline(x=p99_rate_deg, color='g', linestyle='--', label='99.9%ile')
    ax2.legend()
    
    plt.tight_layout()
    output_png = output_dir / f"{base_name}_rate.png"
    plt.savefig(output_png)
    print(f"\nPlot saved to {output_png}")

def main():
    parser = argparse.ArgumentParser(description="Analyze steering rate limits.")
    parser.add_argument("file", help="Input MCAP file")
    
    args = parser.parse_args()
    mcap_path = args.file
    
    times, vals = extract_steering_data(mcap_path)
    
    if len(times) == 0:
        print("No data extracted.")
        return
        
    base_name = Path(mcap_path).stem
    output_dir = Path(__file__).parent / "results"
    
    analyze_rate(times, vals, output_dir, base_name)

if __name__ == "__main__":
    main()
