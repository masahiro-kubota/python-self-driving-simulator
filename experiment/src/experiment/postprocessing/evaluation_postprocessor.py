import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from core.data.experiment import Artifact, ExperimentResult
from core.data.simulator import SimulationLog
from core.utils import get_project_root
from experiment.interfaces import ExperimentPostprocessor
from experiment.postprocessing.mcap_logger import MCAPLogger
from experiment.postprocessing.metrics import EvaluationMetrics, MetricsCalculator
from experiment.postprocessing.mlflow_logger import MLflowExperimentLogger
from experiment.preprocessing.schemas import ResolvedExperimentConfig


class EvaluationPostprocessor(
    ExperimentPostprocessor[Any, ExperimentResult]
):  # TResult is Any because SimulationResult is not easily imported here without circular deps? No, we can import it.
    """Postprocessor for evaluation experiments."""

    def process(self, result: Any, config: ResolvedExperimentConfig) -> ExperimentResult:
        """Process evaluation results.

        Args:
            result: SimulationResult
            config: Experiment configuration

        Returns:
            Processed experiment result
        """
        # Logic from runner.py _run_evaluation (post-execution part)
        sim_result = result

        # Check if running in CI environment
        is_ci = bool(os.getenv("CI"))

        if is_ci:
            from contextlib import nullcontext

            mlflow_context = nullcontext()
        else:
            # We need to setup mlflow logging if not already done?
            # Runner.py did it inside the run method wrapping execution.
            # Here we are in postprocessing. Execution is done.
            # But we want to log the result to mlflow.
            # If we want to capture execution time etc, we might have lost it if we start run here.
            # But typically MLflow run context should wrap the whole experiment or at least be active when logging.
            # If we start it here, it's a new run.
            # Ideally Orchestrator manages the MLflow run?
            # Or Postprocessor starts it just for logging?
            # Since Runner.py wrapped both execution and logging, splitting them is tricky for MLflow context.
            # However, if we just want to log metrics and params, we can start a run, log, and end it.
            mlflow.set_tracking_uri(config.logging.mlflow.tracking_uri)
            mlflow.set_experiment(config.experiment.name)
            mlflow_context = mlflow.start_run()

        # Prepare params to log
        result_params = {
            "ad_component": config.components.ad_component.type,
            **config.components.ad_component.params,
            "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Prepare artifacts container
        result_artifacts = self._collect_input_artifacts(config)

        with mlflow_context:
            # 1. Save MCAP
            mcap_path = Path(config.logging.mcap.output_dir) / "simulation.mcap"
            self._save_mcap(sim_result.log, mcap_path)
            if config.logging.mcap.enabled and mcap_path.exists():
                result_artifacts.append(Artifact(local_path=mcap_path))

            # 2. Add metadata to log
            sim_result.log.metadata = result_params

            # 3. Calculate metrics
            _, metrics_obj = self._calculate_metrics(sim_result.log, sim_result.success)
            result_metrics = metrics_obj

            # 4. Create ExperimentResult
            experiment_result = ExperimentResult(
                experiment_name=config.experiment.name,
                experiment_type=config.experiment.type.value,
                execution_time=datetime.now(),
                simulation_results=[sim_result],
                config=config,
                params=result_params,
                metrics=result_metrics,
                artifacts=result_artifacts,
            )

            # 5. Generate Dashboard
            dashboard_artifact = self._generate_dashboard(experiment_result, config, is_ci)
            if dashboard_artifact:
                experiment_result.artifacts.append(dashboard_artifact)

            # 6. Log to MLflow
            logger = MLflowExperimentLogger(
                tracking_uri=config.logging.mlflow.tracking_uri,
                experiment_name=config.experiment.name,
            )
            logger.log_result(experiment_result)

            # Clean up
            if mcap_path.exists():
                mcap_path.unlink()

        return experiment_result

    def _collect_input_artifacts(self, config: ResolvedExperimentConfig) -> list[Artifact]:
        """Collect input artifacts from configuration."""
        artifacts: list[Artifact] = []
        for input_path in config.logging.inputs:
            full_path = get_project_root() / input_path
            if full_path.exists():
                artifacts.append(Artifact(local_path=full_path, remote_path="input_data"))
            else:
                print(f"Warning: Input file not found: {full_path}")
        return artifacts

    def _save_mcap(self, log: SimulationLog, output_path: Path) -> bool:
        """Save simulation log to MCAP.

        Returns:
            bool: True if save was successful
        """
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with MCAPLogger(output_path) as mcap_logger:
            for step in log.steps:
                mcap_logger.log_step(step)
        return True

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

    def _generate_dashboard(
        self, result: ExperimentResult, config: ResolvedExperimentConfig, is_ci: bool
    ) -> Artifact | None:
        """Generate interactive dashboard."""
        if not config.logging.dashboard.enabled:
            return None

        print("Generating interactive dashboard...")
        dashboard_path = Path("/tmp/dashboard.html")

        # Find OSM file from simulator config
        osm_path = None
        sim_params = config.simulator.params
        if sim_params and "map_path" in sim_params:
            map_path_str = sim_params["map_path"]
            if map_path_str:
                potential_path = Path(map_path_str)
                if not potential_path.is_absolute():
                    potential_path = get_project_root() / potential_path

                if potential_path.exists():
                    osm_path = potential_path
                else:
                    print(f"Warning: Configured map path not found: {potential_path}")

        # Use dashboard package implementation via interface
        # Dynamic loading to avoid static dependency
        import importlib

        from core.interfaces import DashboardGenerator

        try:
            # Dynamically import the dashboard module
            dashboard_module = importlib.import_module("dashboard")
            # Get the generator class
            generator_class = getattr(dashboard_module, "HTMLDashboardGenerator")
            # Instantiate
            generator: DashboardGenerator = generator_class()
            generator.generate(result, dashboard_path, osm_path)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load dashboard generator: {e}")
            # Dashboard generation failed, but we continue experiment execution
            return None

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
                shutil.copy(dashboard_path, ci_dashboard_path)
                print(f"Dashboard saved to {ci_dashboard_path} for CI artifact upload")

        return artifact
