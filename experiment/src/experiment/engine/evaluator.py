import logging
import webbrowser
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
from core.clock import create_clock
from core.data import SimulationResult
from core.data.frame_data import collect_node_output_fields, create_frame_data_type
from core.executor import SingleProcessExecutor
from omegaconf import DictConfig

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
        enable_progress_bar = config.execution.enable_progress_bar

        fields = collect_node_output_fields(nodes)
        dynamic_frame_data_type = create_frame_data_type(fields)
        frame_data = dynamic_frame_data_type()

        for field_name, field_type in fields.items():
            if field_type is bool:
                # Update TopicSlot data instead of overwriting the slot itself
                slot = getattr(frame_data, field_name)
                if hasattr(slot, "update"):
                    slot.update(False)

        for node in nodes:
            node.set_frame_data(frame_data)

        clock = create_clock(start_time=0.0, rate_hz=clock_rate, clock_type=clock_type)
        executor = SingleProcessExecutor(nodes, clock)
        executor.run(duration=duration, enable_progress_bar=enable_progress_bar)

        log = None
        # Prefer Simulator log as it contains better metadata (obstacles, etc.)
        for node in nodes:
            class_name = node.__class__.__name__
            logger.debug(f"Checking node: {class_name}, has get_log: {hasattr(node, 'get_log')}")
            if class_name == "SimulatorNode":
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

        # Helper to safely extract data from TopicSlot
        def get_val(name, default):
            slot = getattr(frame_data, name, None)
            if hasattr(slot, "data"):
                val = slot.data
                return val if val is not None else default
            return default

        # Get reason and normalize empty/None based on simulation state
        reason = get_val("done_reason", None)
        if reason == "" or reason is None:
            # Check if simulation reached duration limit (timeout) vs unexpected termination
            final_time = clock.now
            # Allow small tolerance for floating point comparison
            if final_time >= duration - (1.0 / clock_rate):
                reason = "timeout"
            else:
                reason = "unknown"

        return SimulationResult(
            success=get_val("success", False),
            reason=reason,
            final_state=get_val("sim_state", None),
            log=log,
            metrics={
                "goal_count": get_val("goal_count", 0),
                "checkpoint_count": get_val("checkpoint_count", 0),
            },
        )


class EvaluatorEngine(BaseEngine):
    """評価エンジン"""

    def _run_impl(self, cfg: DictConfig) -> ExperimentResult:
        logger.info("Running Evaluation Engine...")

        # MLflow tags
        mlflow.set_tag("evaluation_type", cfg.experiment.get("type", "standard"))

        # Always resolve hydra_dir correctly to find simulation logs
        # In Multirun, each job gets its own hydra_dir (e.g., outputs/DATE/TIME/0/, outputs/DATE/TIME/1/, etc.)
        try:
            hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        except (ValueError, AttributeError):
            hydra_dir = Path("outputs/latest")

        # Use hydra_dir/evaluation as output_dir for consistency
        # This ensures each Multirun job has its own evaluation directory
        output_dir = hydra_dir / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        from experiment.engine.collector import CollectorEngine

        collector = CollectorEngine()

        results = []
        num_episodes = cfg.execution.num_episodes

        logger.info(f"Evaluating model on {num_episodes} episodes...")

        import numpy as np

        artifacts: list[Artifact] = []

        last_foxglove_url = None

        for i in range(num_episodes):
            episode_cfg = cfg.copy()
            # Set seed for reproducibility/randomization in evaluation
            episode_seed = None
            if cfg.env.obstacles.generation:
                episode_seed = cfg.env.obstacles.generation.seed + i

            rng = np.random.default_rng(episode_seed)

            # Apply configuration randomization/resolution (handles obstacle dict->list conversion)
            collector.randomize_simulation_config(episode_cfg, rng, i)

            episode_dir = output_dir / f"episode_{i:04d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            experiment_structure = collector.create_experiment_instance(
                episode_cfg, episode_dir=episode_dir
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

            # Save result to JSON for metrics aggregation
            result_path = episode_dir / "result.json"
            mcap_path = episode_dir / "simulation.mcap"
            foxglove_url = self._get_foxglove_url(mcap_path)

            try:
                import json

                with open(result_path, "w") as f:
                    json.dump(
                        {
                            "episode_idx": i,
                            "seed": episode_seed,
                            "success": res.success,
                            "reason": res.reason,
                            "foxglove_url": foxglove_url,
                            "metrics": res.metrics,
                            "final_state": {
                                "x": res.final_state.x if res.final_state else None,
                                "y": res.final_state.y if res.final_state else None,
                            }
                            if res.final_state
                            else None,
                        },
                        f,
                        indent=4,
                    )
                logger.debug(f"Saved result to {result_path}")
            except Exception as e:
                logger.warning(f"Failed to save result.json for episode {i}: {e}")

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

                # Log Foxglove URL
                if foxglove_url:
                    last_foxglove_url = foxglove_url

                    # Use print with flush to ensure it shows up in terminal
                    print(f"\n🦊 View in Foxglove: {foxglove_url}", flush=True)

                    # Auto-open if configured
                    if cfg.postprocess.foxglove.auto_open:
                        logger.info(f"Auto-opening Foxglove URL: {foxglove_url}")
                        webbrowser.open(foxglove_url)

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

        # Log Foxglove link to MLflow Notes (clickable from UI)
        if last_foxglove_url:
            mlflow.set_tag(
                "mlflow.note.content",
                f"### 🦊 Foxglove Visualization\n[View in Foxglove]({last_foxglove_url})",
            )

        # Print Simulation Logs to Console after execution
        # Use hydra_dir (root of output) instead of output_dir (evaluation specific)
        sim_log_path = hydra_dir / "simulation.log"
        if sim_log_path.exists():
            print("\n" + "=" * 80)
            print(f" LOGS FROM {sim_log_path} ")
            print("=" * 80)
            try:
                with open(sim_log_path) as f:
                    print(f.read())
            except Exception as e:
                print(f"Failed to read simulation log: {e}")
            print("=" * 80 + "\n")
        else:
            logger.warning(f"Simulation log not found at {sim_log_path}")

        return ExperimentResult(
            experiment=None,  # No longer needed for result processing
            simulation_results=results,
            metrics=metrics,
            execution_time=datetime.now(),
            artifacts=artifacts,
        )
