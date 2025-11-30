"""Experiment runner implementation."""

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
from core.data import SimulationLog, SimulationStep, VehicleState
from core.interfaces import ControlComponent, PlanningComponent, Simulator
from core.logging import MCAPLogger
from core.metrics import MetricsCalculator

from experiment_runner.config import ExperimentConfig


class ExperimentRunner:
    """Unified experiment runner."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.simulator: Simulator | None = None
        self.planner: PlanningComponent | None = None
        self.controller: ControlComponent | None = None

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
                # __file__ is in src/experiment_runner/runner.py
                # Go up 3 levels: runner.py -> experiment_runner -> src -> experiment_runner (package) -> workspace
                workspace_root = Path(__file__).parent.parent.parent.parent
                resolved_params[key] = workspace_root / value
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
            workspace_root = Path(__file__).parent.parent.parent.parent
            track_path = workspace_root / planning_params.pop("track_path")
        elif planning_type == "PurePursuitPlanner":
            # Use default track from component directory
            # __file__ is in src/experiment_runner/runner.py
            # Go to components_packages/planning/pure_pursuit/src/pure_pursuit/data/tracks/
            components_root = Path(__file__).parent.parent.parent.parent / "components_packages"
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
                )
            else:
                raise ValueError("Planner does not have reference_trajectory")

        self.simulator = self._instantiate_component("simulators", sim_type, sim_params)

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
            log = SimulationLog(metadata=params)
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

                    # Log current step (before state update)
                    sim_step = SimulationStep(
                        timestamp=step * self.simulator.dt,  # type: ignore
                        vehicle_state=current_state,
                        action=action,
                    )
                    log.add_step(sim_step)
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
                        if dist_to_end < 2.0 and step > 100:
                            print("Reached goal!")
                            break

                    # Update state for next iteration
                    current_state = next_state

                    # Check done flag from simulator
                    if done:
                        print("Simulation done.")
                        break

            end_time = time.time()
            print(f"Simulation finished in {end_time - start_time:.2f}s")

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

                # Workspace root is 4 levels up from this file
                workspace_root = Path(__file__).parent.parent.parent.parent
                tools_scripts = workspace_root / "tools/scripts"
                sys.path.insert(0, str(tools_scripts))
                from generate_dashboard import generate_dashboard

                generate_dashboard(log, dashboard_path)
                if not is_ci:
                    mlflow.log_artifact(str(dashboard_path))
                else:
                    # In CI, save to a persistent location for artifact upload
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
        raise NotImplementedError("Training mode not yet implemented")
