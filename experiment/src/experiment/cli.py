#!/usr/bin/env python3
"""Run experiment from Hydra configuration."""

import math # Added

from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from experiment.core.orchestrator import ExperimentOrchestrator


def div_int_resolver(a: int, b: int) -> int:
    """Integer division resolver."""
    return a // b

def div_ceil_resolver(a: int, b: int) -> int:
    """Ceiling division resolver."""
    return math.ceil(a / b)

def add_int_resolver(a: int, b: int) -> int:
    """Addition resolver."""
    return a + b

def seed_range_resolver(base: int, offset: int, total_episodes: int, step: int) -> str:
    """Resolve seed range dynamically.
    
    Args:
        base: Base seed value (from execution.base_seed)
        offset: Offset for this specific component (e.g. 100 for obstacles)
        total_episodes: Total number of episodes across all jobs
        step: Step size (episodes per job)
    
    Returns:
        String format for Hydra range: 'range(start, end, step)'
    """
    start = base + offset
    end = start + total_episodes
    return f"range({start},{end},{step})"

# Register custom resolvers
OmegaConf.register_new_resolver("div_int", div_int_resolver, replace=True)
OmegaConf.register_new_resolver("div_ceil", div_ceil_resolver, replace=True)
OmegaConf.register_new_resolver("add_int", add_int_resolver, replace=True)
OmegaConf.register_new_resolver("seed_range", seed_range_resolver, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)



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
    load_dotenv()  # Load environment variables from .env file

    # Configure logging level from config
    import logging

    try:
        log_level_str = cfg.execution.get("log_level", "INFO")
        log_level = getattr(logging, log_level_str.upper())
        logging.getLogger().setLevel(log_level)
        # Also set for specific loggers if needed, but root should cover it
        # Hydra's own handlers might need adjustment, but setLevel on root affects propagation
    except Exception as e:
        print(f"Warning: Failed to set log level: {e}")

    # Update outputs/latest symlink
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.run.dir).resolve()

        # Assuming run_dir is inside an 'outputs' directory or equivalent root
        # We try to find the 'outputs' directory
        # Standard structure: outputs/YYYY-MM-DD/HH-MM-SS
        output_base = run_dir.parent.parent

        if output_base.exists():
            latest_link = output_base / "latest"
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()

            # Create relative symlink
            relative_target = run_dir.relative_to(output_base)
            latest_link.symlink_to(relative_target)
            print(f"Updated symlink: {latest_link} -> {relative_target}")
    except Exception as e:
        print(f"Warning: Could not update 'latest' symlink: {e}")

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
