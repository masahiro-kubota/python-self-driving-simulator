#!/usr/bin/env python3
"""Run experiment from YAML configuration."""

import argparse
from pathlib import Path

from .orchestrator import ExperimentOrchestrator


def main() -> None:
    """Run experiment."""
    parser = argparse.ArgumentParser(description="Run experiment from YAML configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="experiment/configs/experiments/pure_pursuit_lookahead_sweep.yaml",
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
            f"  1. Copy a template: cp experiment/configs/experiments/pure_pursuit_lookahead_sweep.yaml {args.config}"
        )
        print("  2. Edit the configuration file")
        print(f"  3. Run: uv run experiment-runner --config {args.config}")
        return

    if args.dry_run:
        print(f"Dry run: Validation of {config_path} is successful (mocked).")
        return

    print(f"Running experiment with config: {config_path}")

    orchestrator = ExperimentOrchestrator()
    orchestrator.run(config_path)

    # We might want to print a summary of the result?
    # result is ExperimentResult object.
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
