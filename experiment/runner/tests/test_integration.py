"""Integration tests for experiment runner."""

import os
from pathlib import Path

import pytest
from experiment_runner import ExperimentConfig, ExperimentRunner

import mlflow


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
    config = ExperimentConfig.from_yaml(config_path)

    # Verify configuration
    assert config.experiment.name == "pure_pursuit_tracking"
    assert config.components.planning.type == "pure_pursuit.PurePursuitPlanner"
    assert config.components.control.type == "pid_controller.PIDController"
    assert config.execution.max_steps_per_episode == 2000

    # Run experiment
    runner = ExperimentRunner(config)
    runner.run()


@pytest.mark.integration
def test_pure_pursuit_experiment_new_config() -> None:
    """Test Pure Pursuit experiment execution with new config format (vehicle/scene)."""
    # Load config
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit_new_config.yaml"
    config = ExperimentConfig.from_yaml(config_path)

    # Verify configuration
    assert config.experiment.name == "pure_pursuit_tracking_new_config"
    assert (
        config.simulator.params["vehicle_config"]
        == "experiment/configs/vehicles/default_vehicle.yaml"
    )
    assert config.simulator.params["scene_config"] == "experiment/configs/scenes/default_scene.yaml"

    # Run experiment
    runner = ExperimentRunner(config)
    runner.run()

    # Verify MLflow run was created
    if not os.getenv("CI"):
        mlflow.set_tracking_uri(config.logging.mlflow.tracking_uri)
        experiment = mlflow.get_experiment_by_name(config.experiment.name)
        assert experiment is not None, "MLflow experiment should be created"

        # Get the latest run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        assert len(runs) > 0, "At least one run should exist"

        latest_run = runs.iloc[0]

        # Verify metrics were logged
        assert "metrics.success" in latest_run, "Success metric should be logged"
        assert (
            latest_run["metrics.success"] == 1.0
        ), "Simulation should complete full lap successfully"
        assert "metrics.lap_time_sec" in latest_run, "Lap time should be logged"
        assert latest_run["metrics.lap_time_sec"] > 0, "Lap time should be positive"

        # Verify parameters were logged
        assert "params.planner" in latest_run, "Planner parameter should be logged"
        assert latest_run["params.planner"] == "pure_pursuit.PurePursuitPlanner"
        assert "params.controller" in latest_run, "Controller parameter should be logged"
        assert latest_run["params.controller"] == "pid_controller.PIDController"

        # Verify artifacts were uploaded
        run_id = latest_run["run_id"]
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [artifact.path for artifact in artifacts]

        assert "simulation.mcap" in artifact_names, "MCAP file should be uploaded"
        assert "dashboard.html" in artifact_names, "Dashboard should be uploaded"


@pytest.mark.integration
def test_config_loading() -> None:
    """Test loading configuration."""
    # __file__ is in experiment_runner/tests/test_integration.py
    # Go up 2 levels to get to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = ExperimentConfig.from_yaml(config_path)

    # Verify structure
    assert config.experiment.name == "pure_pursuit_tracking"
    assert config.components.planning.params["lookahead_distance"] == 5.0
    assert config.components.control.params["kp"] == 1.0
    assert config.simulator.type == "simulator_kinematic.KinematicSimulator"

    assert config.logging.mlflow.enabled is True


@pytest.mark.integration
def test_custom_track_loading(_setup_mlflow_env: None) -> None:
    """Test loading a custom track from the data directory."""
    # Load base config
    # __file__ is in experiment_runner/tests/test_integration.py
    # Go up 2 levels to get to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = ExperimentConfig.from_yaml(config_path)

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
        config.components.planning.params["track_path"] = custom_track_path

        # Run experiment setup
        runner = ExperimentRunner(config)
        runner._setup_components()

        # Verify track was loaded from custom path
        assert runner.track_path is not None
        assert str(runner.track_path).endswith(custom_track_path)
        assert runner.planner is not None
        # Check if reference trajectory is set (implies track loaded successfully)
        assert hasattr(runner.planner, "reference_trajectory")
        assert len(runner.planner.reference_trajectory) > 0  # type: ignore

    finally:
        # Cleanup
        if full_path.exists():
            full_path.unlink()
