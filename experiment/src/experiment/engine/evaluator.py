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
        executor.run(duration=duration)

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

        return SimulationResult(
            success=get_val("success", False),
            reason=get_val("done_reason", None),
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
        try:
            hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
        except (ValueError, AttributeError):
            hydra_dir = Path("outputs/latest")

        output_dir_raw = cfg.get("output_dir")
        if output_dir_raw:
            output_dir = Path(output_dir_raw)
        else:
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

                # Log Foxglove URL
                try:
                    import urllib.parse

                    # Find project root by looking for uv.lock or .git
                    current_dir = Path(__file__).resolve().parent
                    project_root = None
                    for parent in [current_dir, *list(current_dir.parents)]:
                        if (parent / "uv.lock").exists() or (parent / ".git").exists():
                            project_root = parent
                            break

                    if project_root:
                        rel_mcap_path = mcap_path.resolve().relative_to(project_root.resolve())
                        mcap_url = f"http://127.0.0.1:8080/{rel_mcap_path}"
                        encoded_url = urllib.parse.quote(mcap_url, safe="")
                        foxglove_url = (
                            f"https://app.foxglove.dev/view?ds=remote-file&ds.url={encoded_url}"
                        )
                        last_foxglove_url = foxglove_url

                        # Use print with flush to ensure it shows up in terminal
                        print(f"\n🦊 View in Foxglove: {foxglove_url}", flush=True)

                        # Auto-open if configured
                        if cfg.postprocess.foxglove.auto_open:
                            logger.info(f"Auto-opening Foxglove URL: {foxglove_url}")
                            webbrowser.open(foxglove_url)
                except Exception:
                    pass

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
