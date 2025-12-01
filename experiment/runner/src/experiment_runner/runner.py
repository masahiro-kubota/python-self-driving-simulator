"""Experiment runner implementation."""

import importlib
import os
import time
from pathlib import Path
from typing import Any

import mlflow
from core.data import SimulationStep, VehicleState
from core.interfaces import Controller, Planner, Simulator

from experiment_runner.config import ExperimentConfig, ExperimentType
from experiment_runner.logging import MCAPLogger
from experiment_runner.metrics import MetricsCalculator


class ExperimentRunner:
    """Unified experiment runner."""

    def __init__(self, config: ExperimentConfig, config_path: str | Path | None = None) -> None:
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            config_path: Path to the config file (for MLflow logging)
        """
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self.simulator: Simulator | None = None
        self.planner: Planner | None = None
        self.controller: Controller | None = None

    def _instantiate_component(
        self, module_path: str, class_name: str, params: dict[str, Any]
    ) -> Any:
        """Dynamically instantiate a component.

        Args:
            module_path: Module path (e.g., "components.planning")
            class_name: Class name (e.g., "PurePursuitPlanner")
            params: Component parameters

        Returns:
            Instantiated component
        """
        # Resolve special parameters
        resolved_params = {}
        path_keys = {"track_path", "model_path", "scaler_path"}

        for key, value in params.items():
            if key in path_keys and isinstance(value, str):
                # User specified custom path
                resolved_params[key] = self.workspace_root / value
            else:
                resolved_params[key] = value

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**resolved_params)

    def _setup_components(self) -> None:
        """Set up all components based on configuration."""
        # Planning
        planning_type = self.config.components.planning.type
        planning_params = self.config.components.planning.params.copy()

        # Handle track_path specially for PurePursuitPlanner
        track_path = None
        if "track_path" in planning_params:
            # User specified custom track path
            workspace_root = Path(__file__).parent.parent.parent.parent.parent
            track_path = workspace_root / planning_params.pop("track_path")
        elif planning_type == "PurePursuitPlanner":
            # Use default track from component directory
            # __file__ is in src/experiment_runner/runner.py
            # Go to components_packages/planning/pure_pursuit/src/pure_pursuit/data/tracks/
            components_root = (
                Path(__file__).parent.parent.parent.parent.parent / "component_packages"
            )
            default_track = (
                components_root
                / "planning/pure_pursuit/src/pure_pursuit/data/tracks"
                / "raceline_awsim_15km.csv"
            )
            if default_track.exists():
                track_path = default_track

        # Determine module path based on type
        if planning_type == "PurePursuitPlanner":
            planning_module = "pure_pursuit"
        else:
            planning_module = "components.planning"  # Fallback/TODO

        self.planner = self._instantiate_component(planning_module, planning_type, planning_params)

        # Load track if specified and store for artifact logging
        self.track_path = None
        if track_path is not None:
            from planning_utils import load_track_csv

            track = load_track_csv(track_path)
            self.planner.set_reference_trajectory(track)  # type: ignore
            self.track_path = track_path  # Store for artifact logging

        # Control
        control_type = self.config.components.control.type
        control_params = self.config.components.control.params

        if control_type == "PIDController":
            control_module = "pid"
        elif control_type == "NeuralController":
            control_module = "neural_controller"
        else:
            control_module = "components.control"  # Fallback

        self.controller = self._instantiate_component(control_module, control_type, control_params)

        # Simulator
        sim_type = self.config.simulator.type
        sim_params = self.config.simulator.params.copy()

        # Handle initial_state from track if specified
        if sim_params.get("initial_state", {}).get("from_track"):
            if hasattr(self.planner, "reference_trajectory"):
                track = self.planner.reference_trajectory  # type: ignore
                sim_params["initial_state"] = VehicleState(
                    x=track[0].x,
                    y=track[0].y,
                    yaw=track[0].yaw,
                    velocity=0.0,
                    timestamp=0.0,
                )
            else:
                raise ValueError("Planner does not have reference_trajectory")

        # Determine simulator module based on type
        if sim_type == "KinematicSimulator":
            sim_module = "simulator_kinematic"
        elif sim_type == "DynamicSimulator":
            sim_module = "simulator_dynamic"
        else:
            # Fallback for backward compatibility
            sim_module = "simulators"

        self.simulator = self._instantiate_component(sim_module, sim_type, sim_params)

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        # Skip MLflow in CI environments
        if os.getenv("CI"):
            print("CI environment detected - skipping MLflow setup")
            return

        if not self.config.logging.mlflow.enabled:
            return

        mlflow_uri = self.config.logging.mlflow.tracking_uri
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(self.config.experiment.name)

    def run(self) -> None:
        """Run the experiment."""
        exp_type = self.config.experiment.type

        if exp_type == ExperimentType.DATA_COLLECTION:
            self._setup_components()
            self._setup_mlflow()
            self._run_data_collection()
        elif exp_type == ExperimentType.TRAINING:
            self._setup_mlflow()
            self._run_training()
        elif exp_type == ExperimentType.EVALUATION:
            self._setup_components()
            self._setup_mlflow()
            self._run_evaluation()
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

    def _run_evaluation(self) -> None:
        """Run evaluation mode."""
        assert self.simulator is not None
        assert self.planner is not None
        assert self.controller is not None

        # Check if running in CI environment
        is_ci = bool(os.getenv("CI"))

        # Get reference trajectory for metrics
        if hasattr(self.planner, "reference_trajectory"):
            reference_trajectory = self.planner.reference_trajectory  # type: ignore
        else:
            reference_trajectory = None

        # Context manager for MLflow (no-op in CI)
        if is_ci:
            # Use a dummy context manager in CI
            from contextlib import nullcontext

            mlflow_context = nullcontext()
            params = {}  # Empty params for CI
        else:
            mlflow_context = mlflow.start_run()

        with mlflow_context:
            # Log parameters (skip in CI)
            if not is_ci:
                params = {
                    "planner": self.config.components.planning.type,
                    "controller": self.config.components.control.type,
                    **self.config.components.planning.params,
                    **self.config.components.control.params,
                }
                mlflow.log_params(params)

                # Log input data files as artifacts for reproducibility
                if self.track_path is not None:
                    mlflow.log_artifact(str(self.track_path), artifact_path="input_data")

            # Initialize log
            from datetime import datetime

            params["execution_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # log = SimulationLog(metadata=params)  # Handled by simulator
            mcap_path = Path(self.config.logging.mcap.output_dir) / "simulation.mcap"

            print("Starting simulation...")
            start_time = time.time()

            # Initialize state from simulator
            current_state = self.simulator.reset()

            with MCAPLogger(mcap_path) as mcap_logger:
                max_steps = (
                    self.config.execution.max_steps_per_episode if self.config.execution else 2000
                )
                for step in range(max_steps):
                    # Plan
                    target_trajectory = self.planner.plan(None, current_state)  # type: ignore

                    # Control
                    action = self.controller.control(target_trajectory, current_state)

                    # Simulate and get next state
                    next_state, observation, done, info = self.simulator.step(action)

                    # Log to MCAP (still done here for now, or could be moved to simulator too?)
                    # For now, we keep MCAP logging here as it might depend on runner-specifics,
                    # but we remove the SimulationLog accumulation.
                    # Actually, we need to construct SimulationStep for MCAP logging.
                    sim_step = SimulationStep(
                        timestamp=step * self.simulator.dt,  # type: ignore
                        vehicle_state=current_state,
                        action=action,
                    )
                    mcap_logger.log_step(sim_step)

                    if step % 100 == 0:
                        print(
                            f"Step {step}: x={current_state.x:.2f}, "
                            f"y={current_state.y:.2f}, v={current_state.velocity:.2f}"
                        )

                    # Check if reached end
                    if reference_trajectory is not None:
                        dist_to_end = (
                            (current_state.x - reference_trajectory[-1].x) ** 2
                            + (current_state.y - reference_trajectory[-1].y) ** 2
                        ) ** 0.5
                        # Use time threshold instead of step threshold to handle different dt
                        elapsed_time = step * self.simulator.dt
                        if dist_to_end < 5.0 and elapsed_time > 20.0:
                            break

                    # Update state for next iteration
                    current_state = next_state

                    # Check done flag from simulator
                    if done:
                        print("Simulation done.")
                        break

            end_time = time.time()
            print(f"Simulation finished in {end_time - start_time:.2f}s")

            # Retrieve log from simulator
            log = self.simulator.get_log()
            log.metadata = params

            # Calculate metrics
            if reference_trajectory is not None:
                print("Calculating metrics...")
                calculator = MetricsCalculator(reference_trajectory=reference_trajectory)
                metrics = calculator.calculate(log)
                if not is_ci:
                    mlflow.log_metrics(metrics.to_dict())

                print("\nMetrics:")
                for key, value in metrics.to_dict().items():
                    print(f"  {key}: {value}")

            # Upload MCAP (skip in CI)
            if self.config.logging.mcap.enabled and not is_ci:
                print("Uploading MCAP file...")
                mlflow.log_artifact(str(mcap_path))

            # Generate dashboard
            if self.config.logging.dashboard.enabled:
                print("Generating interactive dashboard...")
                dashboard_path = Path("/tmp/dashboard.html")

                # Use dashboard package
                from dashboard import HTMLDashboardGenerator

                # Find OSM file in dashboard assets
                workspace_root = Path(__file__).parent.parent.parent.parent.parent
                osm_path = workspace_root / "dashboard" / "assets" / "lanelet2_map.osm"
                if not osm_path.exists():
                    osm_path = None
                    print(
                        "Warning: lanelet2_map.osm not found in dashboard/assets, "
                        "dashboard will not include map data"
                    )

                generator = HTMLDashboardGenerator()
                generator.generate(log, dashboard_path, osm_path)
                if not is_ci:
                    mlflow.log_artifact(str(dashboard_path))
                else:
                    # In CI, save simulation log as JSON for dashboard injection
                    ci_log_path = Path("simulation_log.json")
                    log.save(ci_log_path)
                    print(f"Simulation log saved to {ci_log_path} for CI dashboard injection")

                    # Also save dashboard to persistent location for artifact upload
                    ci_dashboard_path = Path("dashboard.html")
                    if dashboard_path.exists():
                        import shutil

                        shutil.copy(dashboard_path, ci_dashboard_path)
                        print(f"Dashboard saved to {ci_dashboard_path} for CI artifact upload")

            # Clean up
            if mcap_path.exists():
                mcap_path.unlink()
            if dashboard_path.exists() and not is_ci:
                dashboard_path.unlink()

            # Print MLflow links (skip in CI)
            if not is_ci:
                run_info = mlflow.active_run().info  # type: ignore
                run_id = run_info.run_id
                experiment_id = run_info.experiment_id

                print(f"\n{'='*70}")
                print("MLflow Tracking")
                print(f"{'='*70}")
                print(f"Run ID: {run_id}")
                print(f"Experiment: {self.config.experiment.name}")
                print("\nView this run:")
                print(
                    f"  {self.config.logging.mlflow.tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
                )
                print("\nView artifacts (dashboard, MCAP):")
                print(
                    f"  {self.config.logging.mlflow.tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}/artifacts"
                )
                print("\nView all runs in this experiment:")
                print(f"  {self.config.logging.mlflow.tracking_uri}/#/experiments/{experiment_id}")
                print(f"{'='*70}\n")

    def _run_data_collection(self) -> None:
        """Run data collection mode."""
        assert self.simulator is not None
        assert self.planner is not None
        assert self.controller is not None
        assert self.config.data_collection is not None
        assert self.config.execution is not None

        # Check if running in CI environment
        is_ci = bool(os.getenv("CI"))

        # Context manager for MLflow (no-op in CI)
        if is_ci:
            from contextlib import nullcontext

            mlflow_context = nullcontext()
        else:
            mlflow_context = mlflow.start_run()

        with mlflow_context:
            # Log parameters (skip in CI)
            if not is_ci:
                params = {
                    "planner": self.config.components.planning.type,  # type: ignore
                    "controller": self.config.components.control.type,  # type: ignore
                    "num_episodes": self.config.execution.num_episodes,
                }
                mlflow.log_params(params)

            # Create output directory
            output_dir = Path(self.config.data_collection.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Starting data collection for {self.config.execution.num_episodes} episodes...")
            print(f"Output directory: {output_dir}")

            collected_files = []

            for episode in range(self.config.execution.num_episodes):
                print(f"\nEpisode {episode + 1}/{self.config.execution.num_episodes}")

                # Initialize state from simulator
                current_state = self.simulator.reset()

                # Run episode
                for step in range(self.config.execution.max_steps_per_episode):
                    # Plan
                    target_trajectory = self.planner.plan(None, current_state)  # type: ignore

                    # Control
                    action = self.controller.control(target_trajectory, current_state)

                    # Simulate and get next state
                    next_state, observation, done, info = self.simulator.step(action)

                    # Update state for next iteration
                    current_state = next_state

                    # Check done flag from simulator
                    if done:
                        print(f"  Episode completed at step {step}")
                        break

                # Save episode data
                if (episode + 1) % self.config.data_collection.save_frequency == 0:
                    log = self.simulator.get_log()

                    if self.config.data_collection.format == "json":
                        log_path = output_dir / f"episode_{episode:04d}.json"
                        log.save(log_path)
                        collected_files.append(log_path)
                        print(f"  Saved: {log_path}")
                    elif self.config.data_collection.format == "mcap":
                        # TODO: Implement MCAP saving
                        print("  MCAP format not yet implemented")

            print(
                f"\nData collection completed. {self.config.execution.num_episodes} episodes saved to {output_dir}"
            )

            # Upload collected data and config to MLflow (skip in CI)
            if not is_ci and collected_files:
                print("\nUploading data to MLflow...")

                # Upload all collected data files
                for data_file in collected_files:
                    mlflow.log_artifact(str(data_file), artifact_path="training_data")

                # Upload the config file used for data collection
                # This allows reproducing the data collection experiment
                if hasattr(self, "config_path") and self.config_path:
                    mlflow.log_artifact(str(self.config_path), artifact_path="config")

                print(f"Uploaded {len(collected_files)} data files to MLflow")
                print("Data can be retrieved from MLflow even if local data/ directory is deleted")

    def _run_training(self) -> None:
        """Run training mode."""
        # Check if running in CI environment
        is_ci = bool(os.getenv("CI"))

        # Context manager for MLflow
        if is_ci:
            from contextlib import nullcontext

            mlflow_context = nullcontext()
        else:
            mlflow_context = mlflow.start_run()

        with mlflow_context:
            print("Starting training...")

            # Get model type and training config
            model_type = self.config.model.type if self.config.model else "NeuralController"
            training_config = self.config.training.dict() if self.config.training else {}

            # Add model architecture to training config
            if self.config.model and self.config.model.architecture:
                training_config.update(self.config.model.architecture)

            # Choose trainer based on model type
            if model_type == "MLP":
                # Simple function approximation
                from experiment_training.function_trainer import FunctionTrainer

                trainer = FunctionTrainer(config=training_config)

                # Find data file
                data_dir = Path(self.config.training.data_dir)
                if not data_dir.exists():
                    print(f"Warning: Data directory {data_dir} does not exist")
                    return

                data_files = list(data_dir.glob("*.json"))
                if not data_files:
                    print(f"No training data (.json) found in {data_dir}")
                    return

                # Use first data file for function approximation
                trainer.train(data_files[0])

            elif model_type == "NeuralController":
                # Imitation learning for controller
                from experiment_training.trainer import Trainer

                # Load reference trajectory from config
                if not self.config.training or not self.config.training.reference_trajectory_path:
                    raise ValueError(
                        "training.reference_trajectory_path is required for NeuralController training"
                    )

                from planning_utils import load_track_csv

                workspace_root = Path(__file__).parent.parent.parent.parent.parent
                ref_traj_path = workspace_root / self.config.training.reference_trajectory_path
                reference_trajectory = load_track_csv(ref_traj_path)

                # Initialize Trainer
                trainer = Trainer(
                    config=training_config,
                    reference_trajectory=reference_trajectory,
                    workspace_root=workspace_root,
                )

                # Find data files
                data_dir = Path(self.config.training.data_dir)
                if not data_dir.exists():
                    print(f"Warning: Data directory {data_dir} does not exist")
                    return

                # Find all .json files
                data_paths = list(data_dir.glob("*.json"))

                if not data_paths:
                    print(f"No training data (.json) found in {data_dir}")
                    return

                # Run training
                trainer.train(data_paths)

            else:
                raise ValueError(f"Unknown model type: {model_type}")
