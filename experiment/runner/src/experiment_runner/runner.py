"""Experiment runner implementation."""

import importlib
import os
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
from core.data import VehicleParameters
from core.data.experiment import Artifact, ExperimentResult
from core.data.simulator import SimulationLog
from core.interfaces import ADComponent, Simulator
from core.nodes import PhysicsNode
from core.utils import get_project_root
from experiment_runner.config import ExperimentConfig
from experiment_runner.executor import (
    SimulationContext,
    SingleProcessExecutor,
)
from experiment_runner.mcap_logger import MCAPLogger
from experiment_runner.metrics import EvaluationMetrics, MetricsCalculator
from experiment_runner.mlflow_logger import MLflowExperimentLogger


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
        self.vehicle_params: VehicleParameters | None = None
        self.ad_component: ADComponent | None = None
        self.logger = MLflowExperimentLogger(
            tracking_uri=self.config.logging.mlflow.tracking_uri,
            experiment_name=self.config.experiment.name,
        )

    def _instantiate_component(
        self,
        component_type: str,
        params: dict[str, Any],
        vehicle_params: VehicleParameters | None = None,
    ) -> Any:
        """Dynamically instantiate a component.

        Args:
            component_type: Component type in "module.ClassName" format
            params: Component parameters
            vehicle_params: Vehicle parameters to inject if provided

        Returns:
            Instantiated component
        """
        import inspect

        # Resolve special parameters
        resolved_params = {}
        path_keys = {"track_path", "model_path", "scaler_path"}

        for key, value in params.items():
            if key in path_keys and isinstance(value, str):
                # User specified custom path
                resolved_params[key] = get_project_root() / value

            # Recursive instantiation for sub-components (e.g. planner/controller in stack)
            elif isinstance(value, dict) and "type" in value and "params" in value:
                sub_type = value["type"]
                sub_params = value["params"]
                # Recursively instantiate, propagating vehicle_params
                resolved_params[key] = self._instantiate_component(
                    sub_type, sub_params, vehicle_params
                )
            else:
                resolved_params[key] = value

        # Inject vehicle_params if provided -> logic moved to filtering below
        # We hold it in a separate logical scope, but add to resolved_params for filtering check
        if vehicle_params is not None:
            # Only add if not already present (avoid overwrite if recursion passed it?)
            # Actually recursion handles it in the call.
            resolved_params["vehicle_params"] = vehicle_params

        cls = None
        # 1. Try resolving via Entry Points
        if "." not in component_type:
            for group in ["ad_components", "simulators"]:
                # Python 3.10+ usage
                eps = metadata.entry_points(group=group)
                # Filter by name
                matches = [ep for ep in eps if ep.name == component_type]
                if matches:
                    cls = matches[0].load()
                    break

        # 2. Fallback to module path
        if cls is None:
            try:
                module_name, class_name = component_type.rsplit(".", 1)
            except ValueError:
                raise ValueError(
                    f"Invalid component type: {component_type}. "
                    "Must be in 'Entry Point' or 'module.ClassName' format."
                ) from None

            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

        # Filter arguments based on __init__ signature
        sig = inspect.signature(cls.__init__)
        valid_params = {}

        for param_name in sig.parameters:
            if param_name == "self":
                continue
            if param_name in resolved_params:
                valid_params[param_name] = resolved_params[param_name]

            # Special case: var_keyword (**kwargs)
            if sig.parameters[param_name].kind == inspect.Parameter.VAR_KEYWORD:
                # Pass all remaining params?
                # For safety, let's just pass what we resolved that matches known needs?
                # Or pass everything remaining?
                # To be safe against "unexpected argument", we usually strictly filter unless **kwargs is there.
                # If **kwargs is there, we can pass everything.
                valid_params.update(resolved_params)
                break

        return cls(**valid_params)

    def _setup_components(self) -> None:
        """Set up all components based on configuration."""
        from core.data import VehicleParameters

        workspace_root = get_project_root()
        sim_params = self.config.simulator.params.copy()

        # 1. Load Vehicle Parameters
        if "vehicle_config" in sim_params:
            from experiment_runner.yaml_vehicle_repository import YamlVehicleParametersRepository

            config_path = sim_params.pop("vehicle_config")
            full_path = workspace_root / config_path

            if not full_path.exists():
                raise FileNotFoundError(f"Vehicle config not found: {full_path}")

            repository = YamlVehicleParametersRepository()
            self.vehicle_params = repository.load(full_path)
            sim_params["vehicle_params"] = self.vehicle_params
        else:
            self.vehicle_params = VehicleParameters()
            if "vehicle_params" not in sim_params:
                sim_params["vehicle_params"] = self.vehicle_params

        # 2. Setup ADComponent
        comp_config = self.config.components
        ad_component_type = comp_config.ad_component.type
        ad_component_params = comp_config.ad_component.params.copy()

        # Instantiate ADComponent with vehicle_params
        ad_component_params["vehicle_params"] = self.vehicle_params
        self.ad_component = self._instantiate_component(ad_component_type, ad_component_params)

        # 3. Setup Simulator
        sim_type = self.config.simulator.type

        # FIXME: Temporary fix/override to ensure correct initial state for Pure Pursuit
        # ConfigLoader merging issue prevents overrides from working correctly
        # We need to update self.config as well so that _generate_dashboard can see the map path
        initial_state_fix = {
            "x": 89630.067,
            "y": 43130.695,
            "yaw": 2.2,
            "velocity": 0.0,
        }
        map_path_fix = "dashboard/assets/lanelet2_map.osm"

        sim_params["initial_state"] = initial_state_fix
        sim_params["map_path"] = map_path_fix

        # Update config object for dashboard generator
        if hasattr(self.config.simulator, "params"):
            self.config.simulator.params["map_path"] = map_path_fix
            self.config.simulator.params["initial_state"] = initial_state_fix

        # Remove scene_config if present (not used by simulator)
        if "scene_config" in sim_params:
            sim_params.pop("scene_config")

        # Pass map_path if available
        if "map_path" in sim_params:
            config_path = sim_params.pop("map_path")
            sim_params["map_path"] = str(workspace_root / config_path)

        self.simulator = self._instantiate_component(
            sim_type, sim_params, vehicle_params=self.vehicle_params
        )

    def run(self) -> None:
        """Run the experiment."""
        # Only EVALUATION mode is supported
        self._setup_components()
        self._run_evaluation()

    def log_result(self, result: ExperimentResult) -> None:
        """実験結果をログに記録する.

        Args:
            result: 記録する実験結果
        """
        self.logger.log_result(result)

    def _run_evaluation(self) -> None:
        """Run evaluation mode."""
        assert self.simulator is not None
        assert self.ad_component is not None

        # Check if running in CI environment
        is_ci = bool(os.getenv("CI"))

        if is_ci:
            # Use a dummy context manager in CI
            from contextlib import nullcontext

            mlflow_context = nullcontext()
        else:
            mlflow_context = mlflow.start_run()

        with mlflow_context:
            result_artifacts = self._collect_input_artifacts()

            # Initialize metadata
            from datetime import datetime

            result_params = {
                "ad_component": self.config.components.ad_component.type,
                **self.config.components.ad_component.params,
                "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            mcap_path = Path(self.config.logging.mcap.output_dir) / "simulation.mcap"

            print("Starting simulation...")
            start_time = time.time()

            # Run simulation
            max_steps = (
                self.config.execution.max_steps_per_episode if self.config.execution else 2000
            )

            # Setup Executor
            sim_rate = self.config.simulator.rate_hz

            # Create Context & Reset simulator
            initial_state = self.simulator.reset()
            context = SimulationContext(
                current_time=0.0,
                sim_state=initial_state,
                vehicle_state=initial_state,  # Initialize perceived state
            )

            # Collect Nodes
            nodes = []
            # 1. Physics
            goal_radius = (
                self.config.execution.goal_radius
                if self.config.execution and hasattr(self.config.execution, "goal_radius")
                else 5.0
            )
            nodes.append(PhysicsNode(self.simulator, rate_hz=sim_rate, goal_radius=goal_radius))

            # 2. ADComponent Nodes
            nodes.extend(self.ad_component.get_schedulable_nodes())

            executor = SingleProcessExecutor(nodes, context)

            # Run
            duration = max_steps * (1.0 / sim_rate)  # Approximate duration based on max_steps
            sim_result = executor.run(duration=duration, dt=1.0 / sim_rate)

            end_time = time.time()
            print(f"Simulation finished in {end_time - start_time:.2f}s")
            print(f"Result: {sim_result.reason} (success={sim_result.success})")

            # Add metadata to log
            sim_result.log.metadata = result_params
            # Save MCAP
            self._save_mcap(sim_result.log, mcap_path)
            if self.config.logging.mcap.enabled and mcap_path.exists():
                result_artifacts.append(Artifact(local_path=mcap_path))

            # Calculate metrics
            _, metrics_obj = self._calculate_metrics(sim_result.log, sim_result.success)
            result_metrics = metrics_obj

            # Create ExperimentResult
            from core.data.experiment import ExperimentResult

            experiment_result = ExperimentResult(
                experiment_name=self.config.experiment.name,
                experiment_type=self.config.experiment.type.value,
                execution_time=datetime.now(),
                simulation_results=[sim_result],
                config=self.config,
                params=result_params,
                metrics=result_metrics,
                artifacts=result_artifacts,
            )

            # Generate dashboard
            dashboard_artifact = self._generate_dashboard(experiment_result, is_ci)
            if dashboard_artifact:
                experiment_result.artifacts.append(dashboard_artifact)

            # Log Consolidated Result
            self.log_result(experiment_result)

            # Clean up
            if mcap_path.exists():
                mcap_path.unlink()

    def _collect_input_artifacts(self) -> list[Artifact]:
        """Collect input artifacts from configuration."""
        artifacts: list[Artifact] = []
        for input_path in self.config.logging.inputs:
            full_path = get_project_root() / input_path
            if full_path.exists():
                artifacts.append(Artifact(local_path=full_path, remote_path="input_data"))
            else:
                print(f"Warning: Input file not found: {full_path}")
        return artifacts

    def _save_mcap(self, log: SimulationLog, output_path: Path) -> None:
        """Save simulation log to MCAP."""
        with MCAPLogger(output_path) as mcap_logger:
            for step in log.steps:
                mcap_logger.log_step(step)

    def _calculate_metrics(
        self, log: SimulationLog, success: bool
    ) -> tuple[dict[str, float], EvaluationMetrics]:
        """Calculate simulation metrics."""
        print("Calculating metrics...")
        calculator = MetricsCalculator()
        metrics = calculator.calculate(log)

        # Override success metric with SimulationResult.success
        metrics.success = 1 if success else 0

        result_metrics = metrics.to_dict()

        print("\nMetrics:")
        for key, value in result_metrics.items():
            print(f"  {key}: {value}")

        return result_metrics, metrics

    def _generate_dashboard(self, result: ExperimentResult, is_ci: bool) -> Artifact | None:
        """Generate interactive dashboard.

        Args:
            result: Experiment result containing simulation results
            is_ci: Whether running in CI environment

        Returns:
            Dashboard artifact if generated, None otherwise
        """
        if not self.config.logging.dashboard.enabled:
            return None

        print("Generating interactive dashboard...")
        dashboard_path = Path("/tmp/dashboard.html")

        # Use dashboard package
        from dashboard import HTMLDashboardGenerator

        # Find OSM file from simulator config
        osm_path = None
        sim_params = self.config.simulator.params
        if sim_params and "map_path" in sim_params:
            map_path_str = sim_params["map_path"]
            if map_path_str:
                workspace_root = get_project_root()
                potential_path = workspace_root / map_path_str
                if potential_path.exists():
                    osm_path = potential_path
                else:
                    print(f"Warning: Configured map path not found: {potential_path}")

        generator = HTMLDashboardGenerator()
        generator.generate(result, dashboard_path, osm_path)

        artifact = None
        if dashboard_path.exists():
            artifact = Artifact(local_path=dashboard_path)

        if is_ci:
            # In CI, save simulation log as JSON for dashboard injection
            ci_log_path = Path("simulation_log.json")
            result.simulation_results[0].log.save(ci_log_path)
            print(f"Simulation log saved to {ci_log_path} for CI dashboard injection")

            # Also save dashboard to persistent location for artifact upload
            ci_dashboard_path = Path("dashboard.html")
            if dashboard_path.exists():
                import shutil

                shutil.copy(dashboard_path, ci_dashboard_path)
                print(f"Dashboard saved to {ci_dashboard_path} for CI artifact upload")

        # Cleanup is handled by caller or assumed temporary
        # Note: Caller cleans up dashboard_path if not CI.
        # But here we return artifact pointing to it.
        # The artifact logging happens BEFORE cleanup in caller.
        return artifact
