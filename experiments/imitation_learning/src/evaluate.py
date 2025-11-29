"""Evaluation script with MLflow tracking."""

import os
import time
from pathlib import Path

import mlflow
from components.control.neural_controller import NeuralController
from core.data import SimulationLog, SimulationStep, VehicleState
from core.logging import MCAPLogger
from core.metrics import MetricsCalculator
from simulators.simple_2d import Simple2DSimulator
from track_loader import load_track_csv


def main() -> None:
    """Evaluate neural controller with MLflow tracking."""
    # Paths
    workspace_root = Path(__file__).parent.parent.parent.parent
    experiment_root = workspace_root / "experiments/imitation_learning"
    data_dir = experiment_root / "data"
    model_path = data_dir / "models/nn_controller.pth"
    scaler_path = data_dir / "models/scaler.json"
    
    track_path = workspace_root / "e2e_warmup/data/tracks/raceline_awsim_15km.csv"
    
    # Check model
    if not model_path.exists() or not scaler_path.exists():
        print("Error: Model files not found.")
        print("Please run 'uv run python src/train.py' first.")
        return

    # MLflow setup
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Set MinIO credentials for artifact storage
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("imitation_learning")
    
    # Load track from training data log (since CSV file is missing)
    training_log_path = data_dir / "raw/log_pure_pursuit.json"
    if not training_log_path.exists():
        print("Error: Training log not found.")
        print("Please run 'uv run python src/collect_data.py' first.")
        return
    
    print(f"Loading track from training log: {training_log_path}...")
    from core.data import SimulationLog as TempLog
    temp_log = TempLog.load(training_log_path)
    
    # Reconstruct track from vehicle states
    from core.data import TrajectoryPoint
    track_points = []
    for step in temp_log.steps[::10]:  # Sample every 10th point
        track_points.append(TrajectoryPoint(
            x=step.vehicle_state.x,
            y=step.vehicle_state.y,
            yaw=step.vehicle_state.yaw,
            velocity=step.vehicle_state.velocity,
        ))
    from core.data import Trajectory
    track = Trajectory(track_points)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        config = {
            "controller": "NeuralController",
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "track": str(track_path),
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
        
        # Neural Controller
        controller = NeuralController(
            model_path=model_path,
            scaler_path=scaler_path,
        )
        controller.set_reference_trajectory(track)
        
        # Initialize log and MCAP logger
        log = SimulationLog(metadata=config)
        mcap_path = Path("/tmp/simulation_nn.mcap")
        
        # Simulation loop
        max_steps = 2000
        print("Starting evaluation...")
        
        start_time = time.time()
        
        with MCAPLogger(mcap_path) as mcap_logger:
            for step in range(max_steps):
                current_state = simulator.current_state
                
                # Control
                action = controller.control(None, current_state)  # type: ignore
                
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
                if dist_to_end < 5.0 and step > 100:
                    print("Reached goal!")
                    break
                    
        elapsed = time.time() - start_time
        print(f"Evaluation finished in {elapsed:.2f}s")
        
        # Calculate metrics
        print("Calculating metrics...")
        calculator = MetricsCalculator(reference_trajectory=track)
        metrics = calculator.calculate(log)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics.to_dict())
        
        # Log MCAP file as artifact
        print(f"Uploading MCAP file to MLflow...")
        mlflow.log_artifact(str(mcap_path))
        
        # Generate and upload interactive dashboard
        print("Generating interactive dashboard...")
        dashboard_path = Path("/tmp/dashboard.html")
        
        # Import dashboard generator
        import sys
        tools_scripts = workspace_root / "packages/tools/scripts"
        sys.path.insert(0, str(tools_scripts))
        from generate_dashboard import generate_dashboard
        
        generate_dashboard(log, dashboard_path)
        mlflow.log_artifact(str(dashboard_path))
        
        # Clean up temp files
        mcap_path.unlink()
        dashboard_path.unlink()
        
        print("\nMetrics:")
        for key, value in metrics.to_dict().items():
            print(f"  {key}: {value}")
        
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"View results at: {mlflow_uri}")


if __name__ == "__main__":
    main()
