#!/usr/bin/env python3
"""Run experiment from YAML configuration."""

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .runner import ExperimentRunner


def main() -> None:
    """Run experiment."""
    parser = argparse.ArgumentParser(description="Run experiment from YAML configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="experiment/configs/current_experiment.yaml",
        help="Path to the experiment configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    # If the config file doesn't exist, list available templates
    if not config_path.exists():
        print(f"Error: Configuration file '{args.config}' not found.")
        print("\nAvailable templates:")
        templates_dir = Path("experiment/configs/experiments")
        if templates_dir.exists():
            for template in templates_dir.glob("*.yaml"):
                print(f"  - {template.name}")
        print("\nTo start a new experiment:")
        print(
            f"  1. Copy a template: cp experiment/configs/experiments/pure_pursuit.yaml {args.config}"
        )
        print("  2. Edit the configuration file")
        print(f"  3. Run: uv run experiment-runner --config {args.config}")
        return

    print(f"Loading configuration from {config_path}...")
    config = ExperimentConfig.from_yaml(config_path)

    print(f"Running experiment: {config.experiment.name}")
    if config.experiment.description:
        print(f"Description: {config.experiment.description}")

    runner = ExperimentRunner(config, config_path=config_path)
    runner.run()


if __name__ == "__main__":
    main()
