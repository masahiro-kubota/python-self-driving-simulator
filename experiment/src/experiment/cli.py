#!/usr/bin/env python3
"""Run experiment from Hydra configuration."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .orchestrator import ExperimentOrchestrator


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Run experiment with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print("Running experiment with Hydra configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    orchestrator = ExperimentOrchestrator()
    orchestrator.run_from_hydra(cfg)

    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
