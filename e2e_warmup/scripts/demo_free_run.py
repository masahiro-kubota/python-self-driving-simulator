import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from vehicle import VehicleState, step

def main():
    # Simulation parameters
    T = 5.0  # seconds
    dt = 0.1
    steps = int(T / dt)
    
    # Initial state
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=0.0)
    
    # Inputs
    delta = 0.0  # Straight
    a = 1.0      # Accelerate
    
    # History
    x_hist = [state.x]
    y_hist = [state.y]
    v_hist = [state.v]
    
    print("Starting simulation...")
    for i in range(steps):
        state = step(state, delta, a, dt)
        x_hist.append(state.x)
        y_hist.append(state.y)
        v_hist.append(state.v)
        
    print("Simulation complete.")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_hist, y_hist, '-o', label='Trajectory')
    plt.title('Vehicle Trajectory (Free Run)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(v_hist)) * dt, v_hist, '-r', label='Velocity')
    plt.title('Velocity Profile')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    plt.tight_layout()
    output_path = 'outputs/demo_free_run.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
