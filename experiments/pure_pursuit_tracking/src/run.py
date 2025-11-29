"""Main experiment script with MLflow tracking."""

import os
import sys
import time
from pathlib import Path

import mlflow
from components.control.pid import PIDController
from components.planning.pure_pursuit import PurePursuitPlanner
from core.data import SimulationLog, SimulationStep, VehicleState
from core.logging import MCAPLogger
from core.metrics import MetricsCalculator
from simulators.simple_2d import Simple2DSimulator
from track_loader import load_track_csv


def main() -> None:
    """Run pure pursuit tracking experiment with MLflow."""
    # Paths
    workspace_root = Path(__file__).parent.parent.parent.parent
    
    # Use existing log for track data (since CSV is missing)
    imitation_log_path = workspace_root / "experiments/imitation_learning/data/raw/log_pure_pursuit.json"
    
    if not imitation_log_path.exists():
        print("Error: Training log not found.")
        print("Please run imitation_learning data collection first.")
        return
    
    # MLflow setup
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("pure_pursuit_tracking")
    
    # Load track from existing log
    print(f"Loading track from {imitation_log_path}...")
    from core.data import SimulationLog as TempLog, Trajectory, TrajectoryPoint
    temp_log = TempLog.load(imitation_log_path)
    
    track_points = []
    for step in temp_log.steps:  # Use all points for better track
        track_points.append(TrajectoryPoint(
            x=step.vehicle_state.x,
            y=step.vehicle_state.y,
            yaw=step.vehicle_state.yaw,
            velocity=step.vehicle_state.velocity,
        ))
    track = Trajectory(track_points)
    print(f"Loaded {len(track)} points.")

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        config = {
            "planner": "PurePursuit",
            "controller": "PID",
            "lookahead_distance": 5.0,
            "kp": 1.0,
            "ki": 0.1,
            "kd": 0.05,
            "dt": 0.1,
        }
        mlflow.log_params(config)
        
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
        
        # Initialize log and MCAP
        log = SimulationLog(metadata=config)
        mcap_path = Path("/tmp/simulation_pp.mcap")
        
        # Simulation loop
        max_steps = 2000
        print("Starting simulation...")
        
        start_time = time.time()
        
        with MCAPLogger(mcap_path) as mcap_logger:
            for step in range(max_steps):
                current_state = simulator.current_state
                
                # Plan
                target_trajectory = planner.plan(None, current_state)  # type: ignore
                
                # Control
                action = controller.control(target_trajectory, current_state)
                
                # Simulate
                simulator.step(action)
                
                # Log
                sim_step = SimulationStep(
                    timestamp=step * 0.1,
                    vehicle_state=current_state,
                    action=action,
                )
                log.add_step(sim_step)
                mcap_logger.log_step(sim_step)
                
                if step % 100 == 0:
                    print(f"Step {step}: x={current_state.x:.2f}, y={current_state.y:.2f}, v={current_state.velocity:.2f}")
                    
                # Check if reached end
                dist_to_end = ((current_state.x - track[-1].x)**2 + (current_state.y - track[-1].y)**2)**0.5
                if dist_to_end < 2.0 and step > 100:
                    print("Reached goal!")
                    break
                    
        elapsed = time.time() - start_time
        print(f"Simulation finished in {elapsed:.2f}s")
        
        # Calculate metrics
        print("Calculating metrics...")
        calculator = MetricsCalculator(reference_trajectory=track)
        metrics = calculator.calculate(log)
        
        # Log metrics
        mlflow.log_metrics(metrics.to_dict())
        
        # Upload MCAP
        print("Uploading MCAP file...")
        mlflow.log_artifact(str(mcap_path))
        
        # Generate and upload dashboard
        print("Generating interactive dashboard...")
        dashboard_path = Path("/tmp/dashboard_pp.html")
        
        tools_scripts = workspace_root / "packages/tools/scripts"
        sys.path.insert(0, str(tools_scripts))
        from generate_dashboard import generate_dashboard
        
        generate_dashboard(log, dashboard_path)
        mlflow.log_artifact(str(dashboard_path))
        
        # Clean up
        mcap_path.unlink()
        dashboard_path.unlink()
        
        print("\nMetrics:")
        for key, value in metrics.to_dict().items():
            print(f"  {key}: {value}")
        
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"View results at: {mlflow_uri}")


if __name__ == "__main__":
    main()
