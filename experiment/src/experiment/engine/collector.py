import logging
import uuid
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from core.data.experiment.config import (
    ExecutionConfig,
    ExperimentMetadata,
    NodeConfig,
    ResolvedExperimentConfig,
)
from core.interfaces.node import Node
from core.utils.node_factory import NodeFactory
from omegaconf import DictConfig, ListConfig, OmegaConf

from experiment.core.structures import Experiment as ExperimentStructure
from experiment.engine.base import BaseEngine
from experiment.engine.evaluator import SimulatorRunner

logger = logging.getLogger(__name__)


class CollectorEngine(BaseEngine):
    """データ収集エンジン"""

    def _run_impl(self, cfg: DictConfig) -> Any:
        num_episodes = cfg.execution.num_episodes
        split = cfg.split

        # Safe HydraConfig access for both Run and Multirun
        try:
            hydra_config = hydra.core.hydra_config.HydraConfig.get()
            # runtime.output_dir is available in both single run and multirun
            hydra_dir = Path(hydra_config.runtime.output_dir)
        except (ValueError, AttributeError, Exception) as e:
            logger.warning(f"Failed to get hydra output dir: {e}. Fallback to outputs/latest")
            hydra_dir = Path("outputs/latest")

        output_dir = hydra_dir / split / "raw_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting data collection: {num_episodes} episodes, split={split}")

        for i in range(num_episodes):
            # Resolve seed for scenario reproducibility (initial state + obstacles)
            # Use obstacles.generation.seed if available
            episode_seed = None
            if cfg.env.obstacles.generation:
                # User config ensures seed is present if generation is configured
                episode_seed = cfg.env.obstacles.generation.seed + i

            # Generate consistent episode directory
            episode_dir_name = f"episode_seed{episode_seed}"
            episode_dir = output_dir / episode_dir_name
            episode_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Episode {i + 1}/{num_episodes} (Seed: {episode_seed}) ---")

            rng = np.random.default_rng(episode_seed)

            episode_cfg = cfg.copy()
            self.randomize_simulation_config(episode_cfg, rng, i)
            experiment = self.create_experiment_instance(episode_cfg, episode_dir)

            runner = SimulatorRunner()
            result = runner.run_simulation(experiment)

            # Save result to JSON for metrics aggregation and filtering
            result_path = episode_dir / "result.json"
            mcap_path = episode_dir / "simulation.mcap"
            foxglove_url = self._get_foxglove_url(mcap_path)

            try:
                import json

                with open(result_path, "w") as f:
                    # Collect seed if available
                    seed_val = episode_seed

                    json.dump(
                        {
                            "episode_idx": i,
                            "seed": seed_val,
                            "success": result.success,
                            "reason": result.reason,
                            "foxglove_url": foxglove_url,
                            "metrics": result.metrics,
                            "final_state": {
                                "x": result.final_state.x if result.final_state else None,
                                "y": result.final_state.y if result.final_state else None,
                            }
                            if result.final_state
                            else None,
                        },
                        f,
                        indent=4,
                    )
            except Exception as e:
                logger.warning(f"Failed to save result.json for episode {i}: {e}")

            if result.success:
                logger.info("Episode successful.")
            else:
                logger.warning(f"Episode failed: {result.reason}")

            # Generate dashboard if enabled
            if cfg.postprocess.dashboard.enabled:
                try:
                    # episode_dir is already defined

                    from dashboard.generator import HTMLDashboardGenerator
                    from dashboard.reader import load_simulation_data

                    mcap_path = episode_dir / "simulation.mcap"
                    dashboard_path = episode_dir / "dashboard.html"

                    if mcap_path.exists():
                        # Load data explicitly
                        dashboard_data = load_simulation_data(mcap_path, vehicle_params=cfg.vehicle)

                        # Generate dashboard
                        generator = HTMLDashboardGenerator()
                        generator.generate(
                            data=dashboard_data,
                            output_path=dashboard_path,
                            osm_path=Path(cfg.env.map_path),
                        )
                        logger.info(f"Generated dashboard: {dashboard_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate dashboard for episode {i}: {e}")

        # Create convenience symlink to this run
        try:
            project_root = Path.cwd()
            symlink_path = project_root / "latest_output"
            if symlink_path.is_symlink() or symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(output_dir)
            logger.info(f"Created/Updated symlink: {symlink_path} -> {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create latest_output symlink: {e}")

        logger.info(f"Data collection completed. Output: {output_dir}")
        return output_dir

    def _convert_nodes_to_list(self, nodes_data: Any) -> list[dict[str, Any]]:
        """Convert dictionary-based nodes configuration to list format.

        Args:
            nodes_data: Either a dict (new format) or list (legacy format) of node configs

        Returns:
            List of node dictionaries with 'name' field populated
        """
        if isinstance(nodes_data, dict):
            # Dictionary format: convert to list preserving order
            nodes_list = []
            for node_key, node_config in nodes_data.items():
                node_dict = dict(node_config)
                # Use the dictionary key as the node name if not explicitly set
                if "name" not in node_dict:
                    # Capitalize first letter of each word for display name
                    node_dict["name"] = node_key.replace("_", " ").title()
                nodes_list.append(node_dict)
            return nodes_list
        else:
            # Legacy list format: return as-is
            return list(nodes_data)

    def create_experiment_instance(self, cfg: DictConfig, episode_dir: Path) -> ExperimentStructure:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        experiment_data = cfg_dict["experiment"]
        execution_data = cfg_dict["execution"]
        postprocess_data = cfg_dict["postprocess"]
        system_data = cfg_dict["system"]

        # episode_dir is already created by caller

        # Convert dictionary-based nodes to list format
        nodes_data = self._convert_nodes_to_list(system_data["nodes"])

        # Handle AD component nodes
        ad_nodes_data = []
        if "ad_components" in cfg_dict and "nodes" in cfg_dict["ad_components"]:
            ad_nodes_data = self._convert_nodes_to_list(cfg_dict["ad_components"]["nodes"])
        elif "nodes" in cfg_dict:
            ad_nodes_data = self._convert_nodes_to_list(cfg_dict["nodes"])

        if ad_nodes_data:
            # Find insertion point (before Supervisor if it exists)
            insert_idx = len(nodes_data)
            for i, node in enumerate(nodes_data):
                if node["name"] == "Supervisor":
                    insert_idx = i
                    break
            # Insert AD component nodes in reverse order to maintain correct sequence
            for node in reversed(ad_nodes_data):
                nodes_data.insert(insert_idx, node)

        # Filter out LoggerNode if MCAP is disabled
        mcap_enabled = postprocess_data.get("mcap", {}).get("enabled", True)
        if not mcap_enabled:
            nodes_data = [n for n in nodes_data if n.get("type") != "LoggerNode"]

        for node in nodes_data:
            if node["name"] == "Logger":
                node["params"]["output_mcap_path"] = str(episode_dir / "simulation.mcap")

        resolved_nodes = [NodeConfig(**node) for node in nodes_data]
        resolved_config = ResolvedExperimentConfig(
            experiment=ExperimentMetadata(**experiment_data),
            nodes=resolved_nodes,
            execution=ExecutionConfig(**execution_data),
            postprocess=postprocess_data,
        )

        factory = NodeFactory()
        nodes: list[Node] = []
        for node_config in resolved_nodes:
            node = factory.create(
                node_type=node_config.type,
                rate_hz=node_config.rate_hz,
                params=node_config.params,
                priority=node_config.priority,
            )
            nodes.append(node)

        return ExperimentStructure(
            id=str(uuid.uuid4()),
            type=resolved_config.experiment.type,
            config=resolved_config,
            nodes=nodes,
        )

    def randomize_simulation_config(
        self, cfg: DictConfig, rng: np.random.Generator, episode_idx: int = 0
    ) -> None:
        from collections.abc import Mapping

        nodes = cfg.system.nodes
        # Handle both dictionary and list formats
        # For DictConfig/dict, we need to iterate over values, not keys
        if isinstance(nodes, Mapping):
            # Dictionary format: get the simulator node by key or by searching values
            sim_node = nodes.get("simulator")
            if not sim_node:
                # Fallback: search by type
                sim_node = next(
                    (n for n in nodes.values() if n.get("type") == "SimulatorNode"), None
                )
        else:
            # List format: search by name
            sim_node = next((n for n in nodes if n.get("name") == "Simulator"), None)

        if not sim_node:
            return

        initial_state = sim_node.params.initial_state

        if cfg.env.get("initial_state_sampling", {}).get("enabled", False):
            self._sample_and_update_initial_state(cfg, sim_node, rng)

        obstacles = sim_node.params.obstacles

        # Check for generator config (Dict input instead of List)
        if isinstance(obstacles, dict | DictConfig) and "generation" in obstacles:
            from experiment.engine.obstacle_generator import ObstacleGenerator

            map_path = Path(sim_node.params.map_path)
            track_path_str = cfg.env.get("track_path")
            track_path = Path(track_path_str) if track_path_str else None

            # Use configured seed from generation config
            # We already know it exists because we are inside 'if generation in obstacles'
            # and the schema enforces 'seed' presence.
            gen_seed = obstacles.generation.seed + episode_idx

            generator = ObstacleGenerator(map_path, track_path=track_path, seed=gen_seed)

            # Pass initial state to generator for exclusion zone validation
            initial_state_dict = {
                "x": initial_state.x,
                "y": initial_state.y,
                "yaw": initial_state.yaw,
                "velocity": initial_state.velocity,
            }

            generated_obstacles = generator.generate(
                obstacles.generation, initial_state=initial_state_dict
            )

            # Support mixing explicit list with generated obstacles
            if "list" in obstacles and obstacles.list:
                generated_obstacles.extend(OmegaConf.to_container(obstacles.list, resolve=True))

            # Replace with flat list for Simulator
            sim_node.params.obstacles = generated_obstacles

        # Legacy behavior: perturb existing list
        elif isinstance(obstacles, list | ListConfig):
            for obs in obstacles:
                obs["position"]["x"] += rng.uniform(-1.0, 1.0)
                obs["position"]["y"] += rng.uniform(-1.0, 1.0)
                obs["position"]["yaw"] = rng.uniform(0, 6.28)

    def _sample_and_update_initial_state(
        self, cfg: DictConfig, sim_node: Any, rng: np.random.Generator
    ) -> None:
        """Sample and update the initial state of the simulation."""
        from simulator.map import LaneletMap

        from experiment.engine.initial_state_sampler import InitialStateSampler

        # Load map and track
        map_path = Path(sim_node.params.map_path)
        track_path_str = cfg.env.get("track_path")
        if not track_path_str:
            logger.warning("track_path not found in config, skipping initial state sampling")
            return

        track_path = Path(track_path_str)

        # Create sampler
        lanelet_map = LaneletMap(map_path)
        sampler = InitialStateSampler(track_path, lanelet_map)

        # Sample new initial state
        sampling_config = cfg.env.initial_state_sampling
        try:
            sampled_state = sampler.sample_initial_state(
                rng=rng,
                lateral_offset_range=tuple(sampling_config.lateral_offset_range),
                yaw_offset_range=tuple(sampling_config.yaw_offset_range),
                velocity_range=tuple(sampling_config.velocity_range),
                max_retries=sampling_config.get("max_retries", 10),
            )

            # Update initial state
            initial_state = sim_node.params.initial_state
            initial_state.x = sampled_state["x"]
            initial_state.y = sampled_state["y"]
            initial_state.yaw = sampled_state["yaw"]
            initial_state.velocity = sampled_state["velocity"]

            logger.info(
                f"Updated initial state: x={initial_state.x:.2f}, y={initial_state.y:.2f}, "
                f"yaw={initial_state.yaw:.3f}, velocity={initial_state.velocity:.2f}"
            )
        except RuntimeError as e:
            logger.error(f"Failed to sample initial state: {e}, using default")
