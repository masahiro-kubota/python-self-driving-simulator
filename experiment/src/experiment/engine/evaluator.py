import logging
from datetime import datetime
from pathlib import Path

import hydra
from core.clock import create_clock
from core.data import SimulationResult
from core.data.frame_data import collect_node_output_fields, create_frame_data_type
from core.executor import SingleProcessExecutor
from omegaconf import DictConfig

import mlflow
from experiment.core.structures import Artifact, ExperimentResult, Metrics
from experiment.engine.base import BaseEngine
from logger import LoggerNode

logger = logging.getLogger(__name__)


class SimulatorRunner:
    """シミュレーションを実行するための汎用クラス"""

    def run_simulation(self, experiment_structure) -> SimulationResult:
        config = experiment_structure.config
        nodes = experiment_structure.nodes

        clock_rate = config.execution.clock_rate_hz
        duration = config.execution.duration_sec
        clock_type = config.execution.clock_type

        fields = collect_node_output_fields(nodes)
        dynamic_frame_data_type = create_frame_data_type(fields)
        frame_data = dynamic_frame_data_type()

        for field_name, field_type in fields.items():
            if field_type is bool:
                setattr(frame_data, field_name, False)

        for node in nodes:
            node.set_frame_data(frame_data)

        clock = create_clock(start_time=0.0, rate_hz=clock_rate, clock_type=clock_type)
        executor = SingleProcessExecutor(nodes, clock)
        executor.run(duration=duration)

        log = None
        # Prefer Simulator log as it contains better metadata (obstacles, etc.)
        for node in nodes:
            class_name = node.__class__.__name__
            logger.debug(f"Checking node: {class_name}, has get_log: {hasattr(node, 'get_log')}")
            if class_name == "Simulator":
                if hasattr(node, "get_log"):
                    log = node.get_log()
                    logger.debug(f"Got log from Simulator: {log is not None}")
                break

        # Fallback to LoggerNode
        if log is None:
            for node in nodes:
                if isinstance(node, LoggerNode):
                    if hasattr(node, "get_log"):
                        log = node.get_log()
                        logger.debug(f"Got log from LoggerNode: {log is not None}")
                    break

        if log is None:
            logger.warning("No log found from any node")

        return SimulationResult(
            success=getattr(frame_data, "success", False),
            reason=getattr(frame_data, "done_reason", None),
            final_state=getattr(frame_data, "sim_state", None),
            log=log,
            metrics={
                "goal_count": getattr(frame_data, "goal_count", 0),
                "checkpoint_count": getattr(frame_data, "checkpoint_count", 0),
            },
        )


class EvaluatorEngine(BaseEngine):
    """評価エンジン"""

    def _run_impl(self, cfg: DictConfig) -> ExperimentResult:
        logger.info("Running Evaluation Engine...")

        # MLflow tags
        mlflow.set_tag("evaluation_type", cfg.experiment.get("type", "standard"))

        output_dir_raw = cfg.get("output_dir")
        if output_dir_raw:
            output_dir = Path(output_dir_raw)
        else:
            # Safe HydraConfig access
            try:
                hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
            except (ValueError, AttributeError):
                hydra_dir = Path("outputs/latest")
            output_dir = hydra_dir / "evaluation"

        output_dir.mkdir(parents=True, exist_ok=True)

        from experiment.engine.collector import CollectorEngine

        collector = CollectorEngine()

        results = []
        num_episodes = cfg.execution.num_episodes

        logger.info(f"Evaluating model on {num_episodes} episodes...")

        import numpy as np

        artifacts: list[Artifact] = []

        if "seed" not in cfg:
            raise ValueError("Configuration must include 'seed' parameter.")

        for i in range(num_episodes):
            episode_cfg = cfg.copy()
            # Set seed for reproducibility/randomization in evaluation
            seed = cfg.seed + i
            rng = np.random.default_rng(seed)

            # Apply configuration randomization/resolution (handles obstacle dict->list conversion)
            collector.randomize_simulation_config(episode_cfg, rng)

            experiment_structure = collector.create_experiment_instance(
                episode_cfg, output_dir=output_dir, episode_idx=i
            )
            runner = SimulatorRunner()
            res = runner.run_simulation(experiment_structure)
            results.append(res)

            reason = res.reason or "timeout"
            goal_count = res.metrics.get("goal_count", 0)
            checkpoint_count = res.metrics.get("checkpoint_count", 0)

            # Enhanced Logging for AI Visibility
            if res.success:
                result_str = "SUCCESS"
            elif reason == "timeout":
                result_str = "TIMEOUT"
            else:
                result_str = f"FAILED ({reason})"

            banner = "=" * 40
            logger.info(
                f"\n{banner}\nEPISODE {i + 1}/{num_episodes}: {result_str}\nCheckpoints: {checkpoint_count}\nGoals: {goal_count}\n{banner}"
            )

            # Record artifact if MCAP was generated
            # CollectorEngine enforces a specific path structure: episode_XXXX/simulation.mcap
            episode_dir = output_dir / f"episode_{i:04d}"
            mcap_path = episode_dir / "simulation.mcap"

            if mcap_path.exists():
                artifacts.append(Artifact(local_path=mcap_path))
                # Also generate dashboard for this episode if enabled
                if cfg.postprocess.dashboard.enabled:
                    from dashboard.generator import HTMLDashboardGenerator
                    from dashboard.reader import load_simulation_data

                    generator = HTMLDashboardGenerator()
                    dashboard_path = episode_dir / "dashboard.html"

                    try:
                        # Load data explicitly
                        dashboard_data = load_simulation_data(mcap_path, vehicle_params=cfg.vehicle)

                        # Generate dashboard
                        generator.generate(
                            data=dashboard_data,
                            output_path=dashboard_path,
                            osm_path=Path(cfg.env.map_path),
                        )
                        artifacts.append(Artifact(local_path=dashboard_path))
                        mlflow.log_artifact(str(dashboard_path))
                    except Exception as e:
                        logger.warning(f"Failed to generate dashboard for episode {i}: {e}")

        # Calculate Aggregate Metrics
        success_count = sum(1 for r in results if r.success)
        success_rate = success_count / num_episodes

        metrics = Metrics(
            success_rate=success_rate,
            goal_count=sum(r.metrics.get("goal_count", 0) for r in results),
            checkpoint_count=sum(r.metrics.get("checkpoint_count", 0) for r in results),
            collision_count=sum(1 for r in results if r.reason and "collision" in r.reason.lower()),
            termination_code=0,
        )

        # Log Metrics to MLflow
        mlflow.log_metric("success_rate", success_rate)
        mlflow.log_metric("num_episodes", num_episodes)
        mlflow.log_metric("goal_count", metrics.goal_count)
        mlflow.log_metric("checkpoint_count", metrics.checkpoint_count)

        return ExperimentResult(
            experiment=None,  # No longer needed for result processing
            simulation_results=results,
            metrics=metrics,
            execution_time=datetime.now(),
            artifacts=artifacts,
        )
