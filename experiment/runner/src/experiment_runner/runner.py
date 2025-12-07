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
from experiment_runner.config import ExperimentConfig, ExperimentType
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
        exp_type = self.config.experiment.type

        if exp_type == ExperimentType.DATA_COLLECTION:
            self._setup_components()
            self._run_data_collection()
        elif exp_type == ExperimentType.TRAINING:
            self._run_training()
        elif exp_type == ExperimentType.EVALUATION:
            self._setup_components()
            self._run_evaluation()
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

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

    def _run_data_collection(self) -> None:
        """Run data collection mode."""
        assert self.simulator is not None
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
            if self.config.data_collection.output_dir:
                output_dir = Path(self.config.data_collection.output_dir)
            else:
                # Use default temporary directory for S3 upload
                output_dir = Path("data/temp/collection")

            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Starting data collection for {self.config.execution.num_episodes} episodes...")
            print(f"Output directory: {output_dir}")

            collected_files = []

            for episode in range(self.config.execution.num_episodes):
                print(f"\nEpisode {episode + 1}/{self.config.execution.num_episodes}")

                # Prepare stack
                ad_stack = (
                    self.ad_component.to_stack()
                    if hasattr(self.ad_component, "to_stack")
                    else self.ad_component
                )

                # Run episode using simulator.run()
                result = self.simulator.run(
                    ad_component=ad_stack,
                    max_steps=self.config.execution.max_steps_per_episode,
                )

                print(f"  Episode completed: {result.reason}")

                # Save episode data
                if (episode + 1) % self.config.data_collection.save_frequency == 0:
                    log = result.log

                    if self.config.data_collection.format == "json":
                        log_path = output_dir / f"episode_{episode:04d}.json"
                        log.save(log_path)
                        collected_files.append(log_path)
                        print(f"  Saved: {log_path}")
                    elif self.config.data_collection.format == "mcap":
                        # TODO: Implement MCAP saving
                        print("  MCAP format not yet implemented")

            print(
                f"\nData collection completed. {self.config.execution.num_episodes} episodes saved"
            )

            # Upload to S3 or keep local based on storage backend
            if self.config.data_collection.storage_backend == "s3" and not is_ci:
                from experiment_runner.storage import DatasetStorage

                storage = DatasetStorage()

                # Ensure datasets bucket exists
                storage.ensure_bucket_exists("datasets")

                # Build S3 path
                s3_base_path = storage.build_dataset_path(
                    project=self.config.data_collection.project,  # type: ignore
                    scenario=self.config.data_collection.scenario,  # type: ignore
                    version=self.config.data_collection.version,  # type: ignore
                    stage=self.config.data_collection.stage,
                )

                print(f"\nUploading data to S3: {s3_base_path}")

                # Upload all collected files to S3
                for log_file in collected_files:
                    s3_path = f"{s3_base_path}{log_file.name}"
                    storage.upload_file(log_file, s3_path)

                print(f"Uploaded {len(collected_files)} files to S3")

                # Log dataset metadata to MLflow
                mlflow.log_param("dataset_project", self.config.data_collection.project)
                mlflow.log_param("dataset_scenario", self.config.data_collection.scenario)
                mlflow.log_param("dataset_version", self.config.data_collection.version)
                mlflow.log_param("dataset_stage", self.config.data_collection.stage)
                mlflow.log_param("dataset_path", s3_base_path)
                mlflow.log_param("num_files", len(collected_files))

                print("Dataset metadata logged to MLflow")
                print(f"Dataset path: {s3_base_path}")

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

            # Determine data directory and files
            # Determine data directory and files
            data_files = []

            # Case 1: S3 Dataset (Recommended)
            if (
                self.config.training.dataset_project
                and self.config.training.dataset_scenario
                and self.config.training.dataset_version
            ) or self.config.training.dataset_path:
                from experiment_runner.storage import DatasetStorage

                storage = DatasetStorage()

                if self.config.training.dataset_path:
                    dataset_path = self.config.training.dataset_path
                else:
                    dataset_path = storage.build_dataset_path(
                        project=self.config.training.dataset_project,  # type: ignore
                        scenario=self.config.training.dataset_scenario,  # type: ignore
                        version=self.config.training.dataset_version,  # type: ignore
                        stage=self.config.training.dataset_stage,
                    )

                print(f"Using S3 dataset: {dataset_path}")

                # List files in S3
                s3_files = storage.list_files(dataset_path, "*.json")

                if not s3_files:
                    print(f"No training data found in {dataset_path}")
                    return

                # Download files to temporary directory for training
                # Note: In a real distributed setting, we might stream data or download on workers
                temp_data_dir = Path("data/cache") / dataset_path.replace("s3://", "")
                temp_data_dir.mkdir(parents=True, exist_ok=True)

                print(f"Downloading {len(s3_files)} files to cache: {temp_data_dir}")

                for s3_file in s3_files:
                    filename = s3_file.split("/")[-1]
                    local_path = temp_data_dir / filename

                    if not local_path.exists():
                        storage.download_file(s3_file, local_path)

                    data_files.append(local_path)

                # Log dataset info to MLflow
                mlflow.log_param("dataset_path", dataset_path)
                if self.config.training.dataset_version:
                    mlflow.log_param("dataset_version", self.config.training.dataset_version)

                if self.config.training.dataset_version:
                    mlflow.log_param("dataset_version", self.config.training.dataset_version)

            else:
                raise ValueError("No data source specified")

            if not data_files:
                print("No training data (.json) found")
                return

            # Choose trainer based on model type
            if model_type == "MLP":
                # Simple function approximation
                from experiment_training.function_trainer import FunctionTrainer

                trainer = FunctionTrainer(config=training_config)

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

                workspace_root = get_project_root()
                ref_traj_path = workspace_root / self.config.training.reference_trajectory_path
                reference_trajectory = load_track_csv(ref_traj_path)

                # Initialize Trainer
                trainer = Trainer(
                    config=training_config,
                    reference_trajectory=reference_trajectory,
                    workspace_root=workspace_root,
                )

                # Run training
                trainer.train(data_files)

            else:
                raise ValueError(f"Unknown model type: {model_type}")
