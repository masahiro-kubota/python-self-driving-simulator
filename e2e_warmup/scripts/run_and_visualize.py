import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import argparse
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from track import Track
from vehicle import VehicleState, step
from controller_pp import pure_pursuit_control, pid_control

def run_simulation(track, T=100.0, dt=0.1, L=2.5, lookahead_dist=5.0, Kp=1.0):
    """Runs the Pure Pursuit simulation and returns history data."""
    
    # Initial state
    state = VehicleState(x=track.x[0], y=track.y[0], yaw=track.yaw[0], v=0.0)
    
    # History
    history = {
        't': [0.0],
        'x': [state.x],
        'y': [state.y],
        'yaw': [state.yaw],
        'v': [state.v],
        'delta': [0.0],
        'a': [0.0]
    }
    
    time = 0.0
    print("Starting simulation...")
    while time < T:
        delta, target_ind = pure_pursuit_control(state, track, L, lookahead_dist)
        target_v = track.v_ref[target_ind]
        a = pid_control(target_v, state.v, Kp)
        state = step(state, delta, a, dt, L)
        time += dt
        
        history['t'].append(time)
        history['x'].append(state.x)
        history['y'].append(state.y)
        history['yaw'].append(state.yaw)
        history['v'].append(state.v)
        history['delta'].append(delta)
        history['a'].append(a)
        
        # Check for lap completion
        dist_to_start = math.hypot(state.x - track.x[0], state.y - track.y[0])
        if time > 20.0 and dist_to_start < 5.0:
            print("Lap completed.")
            break
            
    print(f"Simulation finished in {time:.1f} seconds.")
    return history

def save_static_plot(history, track, filename):
    """Saves a static plot of the trajectory and velocity profile."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(track.x, track.y, 'k--', label='Track')
    plt.plot(history['x'], history['y'], 'b-', label='Trajectory')
    plt.scatter(track.x[0], track.y[0], color='g', label='Start')
    plt.scatter(history['x'][-1], history['y'][-1], color='r', label='End')
    plt.title('Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['t'], history['v'], label='Velocity')
    plt.title('Velocity Profile')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Static plot saved to {filename}")
    plt.close()

def save_animation(history, track, filename, fps=20):
    """Saves a dashboard-style animation of the simulation."""
    print(f"Creating animation with {len(history['x'])} frames...")
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_steer = fig.add_subplot(gs[0, 1])
    ax_accel = fig.add_subplot(gs[1, 1])
    
    # Map Plot
    ax_map.plot(track.x, track.y, 'k--', label='Track')
    vehicle_point, = ax_map.plot([], [], 'bo', markersize=8, label='Vehicle')
    vehicle_line, = ax_map.plot([], [], 'b-', linewidth=2)
    
    margin = 50
    ax_map.set_xlim(min(track.x) - margin, max(track.x) + margin)
    ax_map.set_ylim(min(track.y) - margin, max(track.y) + margin)
    ax_map.set_aspect('equal')
    ax_map.legend()
    ax_map.grid(True)
    ax_map.set_title('Simulation Replay')
    
    # Steering Plot
    ax_steer.set_xlim(0, max(history['t']))
    ax_steer.set_ylim(min(history['delta']) - 0.1, max(history['delta']) + 0.1)
    ax_steer.set_title('Steering Angle (rad)')
    ax_steer.grid(True)
    steer_line, = ax_steer.plot([], [], 'r-')
    
    # Acceleration Plot
    ax_accel.set_xlim(0, max(history['t']))
    ax_accel.set_ylim(min(history['a']) - 1.0, max(history['a']) + 1.0)
    ax_accel.set_title('Acceleration (m/s^2)')
    ax_accel.set_xlabel('Time (s)')
    ax_accel.grid(True)
    accel_line, = ax_accel.plot([], [], 'g-')
    
    # Skip frames
    skip = 2
    frames = range(0, len(history['x']), skip)
    
    def update(i):
        # Map update
        x = history['x'][i]
        y = history['y'][i]
        yaw = history['yaw'][i]
        vehicle_point.set_data([x], [y])
        line_len = 10.0
        vehicle_line.set_data([x, x + line_len * np.cos(yaw)], [y, y + line_len * np.sin(yaw)])
        
        # Graphs update
        current_t = history['t'][:i+1]
        current_delta = history['delta'][:i+1]
        current_a = history['a'][:i+1]
        
        steer_line.set_data(current_t, current_delta)
        accel_line.set_data(current_t, current_a)
        
        return vehicle_point, vehicle_line, steer_line, accel_line
    
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
    ani.save(filename, writer='pillow', fps=fps)
    print(f"Animation saved to {filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run Pure Pursuit simulation and visualize results.')
    parser.add_argument('--track', type=str, default='data/tracks/raceline_awsim_15km.csv', help='Path to track CSV file')
    parser.add_argument('--output_png', type=str, default='outputs/simulation_result.png', help='Output PNG filename')
    parser.add_argument('--output_gif', type=str, default='outputs/simulation_result.gif', help='Output GIF filename')
    args = parser.parse_args()
    
    # Load track
    track = Track(args.track)
    print(f"Loaded track from {args.track}")
    
    # Run simulation
    history = run_simulation(track)
    
    # Save visualizations
    save_static_plot(history, track, args.output_png)
    save_animation(history, track, args.output_gif)

if __name__ == "__main__":
    main()
