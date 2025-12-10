from pathlib import Path

import pytest
from pid_controller.pid_controller_node import PIDConfig, PIDControllerNode
from pure_pursuit.pure_pursuit_node import PurePursuitConfig, PurePursuitNode

from core.data import VehicleParameters
from experiment.orchestrator import ExperimentOrchestrator
from experiment.preprocessing.loader import load_experiment_config


@pytest.mark.integration
def test_pure_pursuit_experiment_nodes() -> None:
    """Test Pure Pursuit experiment execution with Native Nodes."""
    # Load config
    workspace_root = Path(__file__).parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/pure_pursuit.yaml"
    config = load_experiment_config(config_path)

    # Verify configuration loaded new types
    ad_component = config.components.ad_component
    assert ad_component.type == "ad_component_core.flexible_ad_component.FlexibleADComponent"

    # We can't easily check internal nodes list of config since it's just params
    # But we can check values
    nodes_config = ad_component.params["nodes"]
    planning_config = next(n for n in nodes_config if n["name"] == "Planning")
    assert planning_config["type"] == "pure_pursuit.PurePursuitNode"
    assert planning_config["params"]["min_lookahead_distance"] == 3.0

    # Run experiment
    orchestrator = ExperimentOrchestrator()
    result = orchestrator.run(config_path)

    assert result is not None
    assert len(result.simulation_results) > 0
    # Check if we have non-zero action eventually
    # (Checking logical correctness of simulation)
    last_step = result.simulation_results[-1].log.steps[-1]
    # We expect some movement
    assert last_step.vehicle_state.x != 89630.067  # Initial X


def test_node_instantiation(tmp_path) -> None:
    """Test direct instantiation of Nodes."""
    vp = VehicleParameters()
    # Create dummy track
    dummy_track = tmp_path / "test_track.csv"
    dummy_track.write_text(
        "x,y,z,x_quat,y_quat,z_quat,w_quat,speed\n0,0,0,0,0,0,1,10\n10,0,0,0,0,0,1,10"
    )

    # Pure Pursuit
    pp_config = {
        "track_path": str(dummy_track),
        "min_lookahead_distance": 2.0,
        "max_lookahead_distance": 10.0,
        "lookahead_speed_ratio": 1.0,
    }
    pp_node = PurePursuitNode(config=pp_config, rate_hz=10.0, vehicle_params=vp)
    assert isinstance(pp_node.config, PurePursuitConfig)
    assert pp_node.config.min_lookahead_distance == 2.0

    # PID
    pid_config = {"kp": 1.0, "ki": 0.0, "kd": 0.0, "u_min": -5.0, "u_max": 5.0}
    pid_node = PIDControllerNode(config=pid_config, rate_hz=30.0, vehicle_params=vp)
    assert isinstance(pid_node.config, PIDConfig)
    assert pid_node.config.u_max == 5.0
