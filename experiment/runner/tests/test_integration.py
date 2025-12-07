"""Integration tests for experiment runner."""

import os
from pathlib import Path

import pytest
from experiment_runner import ExperimentRunner, load_experiment_config


@pytest.fixture(autouse=True)
def _setup_mlflow_env() -> None:
    """Set up MLflow environment for tests."""
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"


@pytest.mark.integration
def test_pure_pursuit_experiment() -> None:
    """Test Pure Pursuit experiment execution end-to-end."""
    # Load config
    # __file__ is in experiment_runner/tests/test_integration.py
    # Go up 2 levels to get to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = load_experiment_config(config_path)

    # Verify configuration
    assert config.experiment.name == "pure_pursuit"
    assert (
        config.components.ad_component.type == "experiment_runner.ad_components.StandardADComponent"
    )
    # The StandardADComponent params are: { planning: { type: ..., params: ... }, control: ... }
    planner_config = config.components.ad_component.params["planning"]
    assert "PurePursuitPlanner" in planner_config["type"]
    assert planner_config["params"]["lookahead_distance"] == 5.0

    assert config.execution.max_steps_per_episode == 2000

    # Run experiment
    runner = ExperimentRunner(config)
    runner.run()


@pytest.mark.integration
def test_config_loading() -> None:
    """Test loading configuration."""
    # __file__ is in experiment_runner/tests/test_integration.py
    # Go up 2 levels to get to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = load_experiment_config(config_path)

    # Verify structure
    assert config.experiment.name == "pure_pursuit"
    assert (
        config.components.ad_component.type == "experiment_runner.ad_components.StandardADComponent"
    )

    planner_config = config.components.ad_component.params["planning"]
    assert "PurePursuitPlanner" in planner_config["type"]
    assert planner_config["params"]["lookahead_distance"] == 5.0

    controller_config = config.components.ad_component.params["control"]
    assert "PIDController" in controller_config["type"]
    assert controller_config["params"]["kp"] == 1.0

    assert config.simulator.type == "KinematicSimulator"
    assert config.logging.mlflow.enabled is True


@pytest.mark.integration
def test_custom_track_loading(_setup_mlflow_env: None) -> None:
    """Test loading a custom track from the data directory."""
    # Load base config
    # __file__ is in experiment_runner/tests/test_integration.py
    # Go up 2 levels to get to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = load_experiment_config(config_path)

    # Create a dummy custom track file
    # __file__ is in experiment_runner/tests/test_integration.py
    # Go up 2 levels to get to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    custom_track_path = "data/planning/pure_pursuit/test_custom_track.csv"
    full_path = workspace_root / custom_track_path

    # Ensure directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy default track to custom path
    default_track = (
        workspace_root
        / "ad_components/planning/pure_pursuit/src/pure_pursuit/data/tracks/raceline_awsim_15km.csv"
    )
    import shutil

    shutil.copy(default_track, full_path)

    try:
        # Modify to use custom track
        # Since we use StandardADComponent, we need to inject into planning params
        config.components.ad_component.params["planning"]["params"]["track_path"] = (
            custom_track_path
        )

        # Run experiment setup
        runner = ExperimentRunner(config)
        runner._setup_components()

        # Verify track was loaded (via planner state)
        assert runner.ad_component.planner is not None
        # Check if reference trajectory is set (implies track loaded successfully)
        assert hasattr(runner.ad_component.planner, "reference_trajectory")
        assert len(runner.ad_component.planner.reference_trajectory) > 0  # type: ignore

    finally:
        # Cleanup
        if full_path.exists():
            full_path.unlink()
