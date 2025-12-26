import logging
import random
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
from experiment.core.structures import Experiment as ExperimentStructure
from experiment.engine.base import BaseEngine
from experiment.engine.evaluator import SimulatorRunner
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


class CollectorEngine(BaseEngine):
    """データ収集エンジン"""

    def _run_impl(self, cfg: DictConfig) -> Any:
        if "seed" not in cfg:
            raise ValueError("Configuration must include 'seed' parameter.")
        seed = cfg.seed
        num_episodes = cfg.execution.num_episodes
        split = cfg.split
        # Safe HydraConfig access
        try:
            hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
        except (ValueError, AttributeError):
            hydra_dir = Path("outputs/latest")

        output_dir = hydra_dir / split / "raw_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting data collection: {num_episodes} episodes, seed={seed}, split={split}"
        )

        for i in range(num_episodes):
            episode_seed = seed + i
            logger.info(f"--- Episode {i + 1}/{num_episodes} (Seed: {episode_seed}) ---")

            random.seed(episode_seed)
            np.random.seed(episode_seed)
            rng = np.random.default_rng(episode_seed)

            episode_cfg = cfg.copy()
            self.randomize_simulation_config(episode_cfg, rng)
            experiment = self.create_experiment_instance(episode_cfg, output_dir, i)

            runner = SimulatorRunner()
            result = runner.run_simulation(experiment)

            if result.success:
                logger.info("Episode successful.")
            else:
                logger.warning(f"Episode failed: {result.reason}")

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

    def create_experiment_instance(
        self, cfg: DictConfig, output_dir: Path, episode_idx: int
    ) -> ExperimentStructure:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        experiment_data = cfg_dict["experiment"]
        execution_data = cfg_dict["execution"]
        postprocess_data = cfg_dict["postprocess"]
        system_data = cfg_dict["system"]

        episode_dir = output_dir / f"episode_{episode_idx:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Convert dictionary-based nodes to list format
        nodes_data = self._convert_nodes_to_list(system_data["nodes"])

        # Handle agent nodes
        agent_nodes_data = []
        if "agent" in cfg_dict and "nodes" in cfg_dict["agent"]:
            agent_nodes_data = self._convert_nodes_to_list(cfg_dict["agent"]["nodes"])
        elif "nodes" in cfg_dict:
            agent_nodes_data = self._convert_nodes_to_list(cfg_dict["nodes"])

        if agent_nodes_data:
            # Find insertion point (before Supervisor if it exists)
            insert_idx = len(nodes_data)
            for i, node in enumerate(nodes_data):
                if node["name"] == "Supervisor":
                    insert_idx = i
                    break
            # Insert agent nodes in reverse order to maintain correct sequence
            for node in reversed(agent_nodes_data):
                nodes_data.insert(insert_idx, node)

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
            )
            nodes.append(node)

        return ExperimentStructure(
            id=str(uuid.uuid4()),
            type=resolved_config.experiment.type,
            config=resolved_config,
            nodes=nodes,
        )

    def randomize_simulation_config(self, cfg: DictConfig, rng: np.random.Generator) -> None:
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
                    (n for n in nodes.values() if n.get("type") == "KinematicSimulator"), None
                )
        else:
            # List format: search by name
            sim_node = next((n for n in nodes if n.get("name") == "Simulator"), None)

        if not sim_node:
            return

        initial_state = sim_node.params.initial_state
        initial_state.x += rng.uniform(-2.0, 2.0)
        initial_state.y += rng.uniform(-2.0, 2.0)
        initial_state.yaw += rng.uniform(-0.5, 0.5)

        obstacles = sim_node.params.obstacles

        # Check for generator config (Dict input instead of List)
        if isinstance(obstacles, dict | DictConfig) and "generation" in obstacles:
            from experiment.engine.obstacle_generator import ObstacleGenerator

            map_path = Path(sim_node.params.map_path)
            gen_seed = int(rng.integers(0, 2**32 - 1))

            generator = ObstacleGenerator(map_path, seed=gen_seed)
            generated_obstacles = generator.generate(obstacles.generation)

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
