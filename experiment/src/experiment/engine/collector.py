import logging
import random
import uuid
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

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

logger = logging.getLogger(__name__)


class CollectorEngine(BaseEngine):
    """データ収集エンジン"""

    def _run_impl(self, cfg: DictConfig) -> Any:
        seed = cfg.get("seed", 42)
        num_episodes = cfg.execution.num_episodes
        split = cfg.get("split", "train")

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
            logger.info(f"--- Episode {i+1}/{num_episodes} (Seed: {episode_seed}) ---")

            random.seed(episode_seed)
            np.random.seed(episode_seed)
            rng = np.random.default_rng(episode_seed)

            episode_cfg = cfg.copy()
            self._randomize_simulation_config(episode_cfg, rng)
            experiment = self.create_experiment_instance(episode_cfg, output_dir, i)

            runner = SimulatorRunner()
            result = runner.run_simulation(experiment)

            if result.success:
                logger.info("Episode successful.")
            else:
                logger.warning(f"Episode failed: {result.reason}")

        logger.info(f"Data collection completed. Output: {output_dir}")
        return output_dir

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

        nodes_data = system_data["nodes"]
        agent_nodes_data = []
        if "agent" in cfg_dict and "nodes" in cfg_dict["agent"]:
            agent_nodes_data = cfg_dict["agent"]["nodes"]
        elif "nodes" in cfg_dict:
            agent_nodes_data = cfg_dict["nodes"]

        if agent_nodes_data:
            insert_idx = len(nodes_data)
            for i, node in enumerate(nodes_data):
                if node["name"] == "Supervisor":
                    insert_idx = i
                    break
            for node in reversed(agent_nodes_data):
                nodes_data.insert(insert_idx, node)

        for node in nodes_data:
            if node["name"] == "Logger":
                node["params"]["output_mcap_path"] = str(episode_dir)

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

    def _randomize_simulation_config(self, cfg: DictConfig, rng: np.random.Generator) -> None:
        nodes = cfg.system.nodes
        sim_node = next((n for n in nodes if n.name == "Simulator"), None)
        if not sim_node:
            return

        initial_state = sim_node.params.initial_state
        initial_state.x += rng.uniform(-2.0, 2.0)
        initial_state.y += rng.uniform(-2.0, 2.0)
        initial_state.yaw += rng.uniform(-0.5, 0.5)

        obstacles = sim_node.params.obstacles
        for obs in obstacles:
            obs["position"]["x"] += rng.uniform(-1.0, 1.0)
            obs["position"]["y"] += rng.uniform(-1.0, 1.0)
            obs["position"]["yaw"] = rng.uniform(0, 6.28)
