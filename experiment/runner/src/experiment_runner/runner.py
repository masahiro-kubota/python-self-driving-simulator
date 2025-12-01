"""Experiment runner implementation."""

import importlib
import os
import time
from pathlib import Path
from typing import Any

import mlflow
from core.data import SimulationStep, VehicleState
from core.interfaces import Controller, Planner, Simulator

from experiment_runner.config import ExperimentConfig
from experiment_runner.logging import MCAPLogger
from experiment_runner.metrics import MetricsCalculator


class ExperimentRunner:
    """Unified experiment runner."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
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
        self._setup_components()
        self._setup_mlflow()

        if self.config.execution.mode == "inference":
            self._run_inference()
        elif self.config.execution.mode == "training":
            self._run_training()
        else:
            raise ValueError(f"Unknown execution mode: {self.config.execution.mode}")

    def _run_inference(self) -> None:
        """Run inference mode."""
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
                for step in range(self.config.execution.max_steps):
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

            # Import Trainer here to avoid circular imports or import errors if training pkg not installed
            from experiment_training.trainer import Trainer

            # Get training config
            training_config = (
                self.config.training.dict() if hasattr(self.config, "training") else {}
            )

            # Setup reference trajectory (needed for feature calculation)
            # We assume the planner has been setup and has the reference trajectory
            if self.planner is None:
                self._setup_components()

            if (
                not hasattr(self.planner, "reference_trajectory")
                or self.planner.reference_trajectory is None
            ):  # type: ignore
                raise ValueError("Planner must have a reference trajectory for training")

            reference_trajectory = self.planner.reference_trajectory  # type: ignore

            # Initialize Trainer
            trainer = Trainer(
                config=training_config,
                reference_trajectory=reference_trajectory,
                workspace_root=Path(__file__).parent.parent.parent.parent.parent,
            )

            # Find data files
            # For now, we look for JSON logs in the data directory specified in config or default
            # Assuming config.logging.mcap.output_dir contains the raw data
            data_dir = Path(self.config.logging.mcap.output_dir)
            if not data_dir.exists():
                print(f"Warning: Data directory {data_dir} does not exist")
                return

            # Find all .json files (using JSON for now as per Dataset implementation)
            data_paths = list(data_dir.glob("*.json"))

            if not data_paths:
                print(f"No training data (.json) found in {data_dir}")
                # Fallback to check for mcap if implemented later
                return

            # Run training
            trainer.train(data_paths)
