import pandas as pd
import numpy as np
import math
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from track import Track
from vehicle import VehicleState, step
from controller_pp import pure_pursuit_control, pid_control
from obs_builder import build_obs

def main():
    # Load track
    csv_path = 'data/tracks/raceline_awsim_15km.csv'
    track = Track(csv_path)
    print(f"Generating dataset using {csv_path}...")
    
    # Simulation parameters
    dt = 0.1
    L = 2.5
    lookahead_dist = 5.0
    Kp = 1.0
    
    # Run for multiple laps to get enough data
    # AWSIM track is long, so maybe 1 lap is enough, but let's do 2 to be sure and capture loop closure well.
    # Or just run for a fixed time.
    T = 200.0 
    
    # Initial state
    state = VehicleState(x=track.x[0], y=track.y[0], yaw=track.yaw[0], v=0.0)
    
    data = []
    
    time = 0.0
    lap_count = 0
    start_dist_threshold = 5.0
    is_near_start = True # Flag to debounce lap counting
    
    print("Starting collection...")
    while time < T:
        # 1. Build observation (Input)
        # [e_lat, e_yaw, v, v_ref]
        obs = build_obs(state, track)
        
        # 2. Get control labels (Output)
        delta, target_ind = pure_pursuit_control(state, track, L, lookahead_dist)
        target_v = track.v_ref[target_ind]
        a = pid_control(target_v, state.v, Kp)
        
        # Store: e_lat, e_yaw, v, v_ref, delta, a
        row = obs + [delta, a]
        data.append(row)
        
        # 3. Update state
        state = step(state, delta, a, dt, L)
        time += dt
        
        # Lap counting logic
        dist_to_start = math.hypot(state.x - track.x[0], state.y - track.y[0])
        if dist_to_start > start_dist_threshold:
            is_near_start = False
        elif not is_near_start and dist_to_start < start_dist_threshold:
            lap_count += 1
            is_near_start = True
            print(f"Lap {lap_count} completed at t={time:.1f}")
            if lap_count >= 2: # Collect 2 laps
                break
                
    # Save to CSV
    columns = ['e_lat', 'e_yaw', 'v', 'v_ref', 'delta', 'a']
    df = pd.DataFrame(data, columns=columns)
    output_path = 'data/datasets/dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
    print(df.describe())

if __name__ == "__main__":
    main()
