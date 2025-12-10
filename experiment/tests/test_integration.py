import os
from pathlib import Path

import pytest

from experiment.orchestrator import ExperimentOrchestrator
from experiment.preprocessing.loader import DefaultPreprocessor, load_experiment_config


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
    workspace_root = Path(__file__).parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = load_experiment_config(config_path)

    # Verify configuration
    assert config.experiment.name == "pure_pursuit"
    assert (
        config.components.ad_component.type
        == "ad_component_core.flexible_ad_component.FlexibleADComponent"
    )

    # FlexibleADComponent uses "nodes" list in params
    nodes_config = config.components.ad_component.params["nodes"]

    # Find Planning node
    planning_node = next(n for n in nodes_config if n["name"] == "Planning")
    # Adapter removal: processor params are now direct
    processor_config = planning_node["processor"]
    assert "PurePursuitPlanner" in processor_config["type"]
    assert processor_config["params"]["lookahead_distance"] == 5.0

    assert config.execution.max_steps_per_episode == 2000

    # Run experiment via Orchestrator
    orchestrator = ExperimentOrchestrator()
    # Note: orchestrator.run returns the result
    result = orchestrator.run(config_path)
    assert result is not None
    # We can inspect the result if needed
    assert len(result.simulation_results) > 0


@pytest.mark.integration
def test_config_loading() -> None:
    """Test loading configuration."""
    workspace_root = Path(__file__).parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = load_experiment_config(config_path)

    # Verify structure
    assert config.experiment.name == "pure_pursuit"
    assert (
        config.components.ad_component.type
        == "ad_component_core.flexible_ad_component.FlexibleADComponent"
    )

    nodes_config = config.components.ad_component.params["nodes"]

    # Planning
    planning_node = next(n for n in nodes_config if n["name"] == "Planning")
    # Adapter removal: processor params are now direct
    planner_config = planning_node["processor"]
    assert "PurePursuitPlanner" in planner_config["type"]
    assert planner_config["params"]["lookahead_distance"] == 5.0

    # Control
    control_node = next(n for n in nodes_config if n["name"] == "Control")
    # Adapter removal: processor params are now direct
    controller_config = control_node["processor"]
    assert "PIDController" in controller_config["type"]
    assert controller_config["params"]["kp"] == 1.0

    assert config.simulator.type == "KinematicSimulator"
    assert config.logging.mlflow.enabled is True


@pytest.mark.integration
def test_custom_track_loading(_setup_mlflow_env: None) -> None:
    """Test loading a custom track from the data directory."""
    # Load base config
    workspace_root = Path(__file__).parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = load_experiment_config(config_path)

    # Create a dummy custom track file
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
        # Need to find Planning node and update its config
        nodes_config = config.components.ad_component.params["nodes"]
        planning_node = next(n for n in nodes_config if n["name"] == "Planning")
        # Adapter removal: set track_path directly in params
        planning_node["processor"]["params"]["track_path"] = custom_track_path

        # Run experiment setup using Preprocessor directly
        preprocessor = DefaultPreprocessor()
        components = preprocessor.setup_components(config)

        ad_component = components["ad_component"]

        # Additional verification: Check if nodes are created
        assert len(ad_component.get_schedulable_nodes()) > 0

        # Check if Planning node exists
        nodes = ad_component.get_schedulable_nodes()
        planning_node_instance = next(n for n in nodes if n.name == "Planning")
        assert planning_node_instance is not None

    finally:
        # Cleanup
        if full_path.exists():
            full_path.unlink()
