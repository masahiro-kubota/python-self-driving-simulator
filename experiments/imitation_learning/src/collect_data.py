"""Data collection script."""

import time
from pathlib import Path

from components.control.pid import PIDController
from components.planning.pure_pursuit import PurePursuitPlanner
from core.data import SimulationLog, SimulationStep, VehicleState
from simulators.simple_2d import Simple2DSimulator
from track_loader import load_track_csv


def main() -> None:
    """Run pure pursuit tracking to collect data."""
    # Paths
    workspace_root = Path(__file__).parent.parent.parent.parent
    track_path = workspace_root / "e2e_warmup/data/tracks/raceline_awsim_15km.csv"
    output_dir = workspace_root / "experiments/imitation_learning/data/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / "log_pure_pursuit.json"
    
    print(f"Loading track from {track_path}...")
    track = load_track_csv(track_path)
    print(f"Loaded {len(track)} points.")

    # Initialize components
    initial_state = VehicleState(
        x=track[0].x,
        y=track[0].y,
        yaw=track[0].yaw,
        velocity=0.0,
    )
    
    simulator = Simple2DSimulator(initial_state=initial_state, dt=0.1)
    planner = PurePursuitPlanner(lookahead_distance=5.0)
    planner.set_reference_trajectory(track)
    controller = PIDController(kp=1.0, ki=0.1, kd=0.05)
    
    # Initialize log
    log = SimulationLog(metadata={
        "track": str(track_path),
        "planner": "PurePursuit",
        "controller": "PID",
        "timestamp": time.time(),
    })

    # Simulation loop
    max_steps = 2000
    print("Starting data collection...")
    
    start_time = time.time()
    
    for step in range(max_steps):
        current_state = simulator.current_state
        
        # 1. Plan
        target_trajectory = planner.plan(None, current_state) # type: ignore
        
        # 2. Control
        action = controller.control(target_trajectory, current_state)
        
        # 3. Simulate
        simulator.step(action)
        
        # 4. Log
        log.add_step(SimulationStep(
            timestamp=step * 0.1,
            vehicle_state=current_state,
            action=action,
        ))
        
        if step % 100 == 0:
            print(f"Step {step}: x={current_state.x:.2f}, y={current_state.y:.2f}, v={current_state.velocity:.2f}")
            
        # Check if reached end
        dist_to_end = ((current_state.x - track[-1].x)**2 + (current_state.y - track[-1].y)**2)**0.5
        if dist_to_end < 2.0 and step > 100:
            print("Reached goal!")
            break
            
    elapsed = time.time() - start_time
    print(f"Data collection finished in {elapsed:.2f}s")
    
    print(f"Saving log to {log_path}...")
    log.save(log_path)


if __name__ == "__main__":
    main()
