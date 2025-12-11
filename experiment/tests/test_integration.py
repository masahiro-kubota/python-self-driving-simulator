from pathlib import Path

import pytest
from pid_controller.pid_controller_node import PIDConfig, PIDControllerNode
from pure_pursuit.pure_pursuit_node import PurePursuitConfig, PurePursuitNode

from core.data import VehicleParameters
from experiment.orchestrator import ExperimentOrchestrator
from experiment.preprocessing.loader import load_experiment_config
from experiment.processors.sensor import IdealSensorNode


@pytest.mark.integration
def test_pure_pursuit_experiment_nodes() -> None:
    """Test Pure Pursuit experiment execution with Native Nodes."""
    # Load config
    workspace_root = Path(__file__).parent.parent.parent
    config_path = workspace_root / "experiment/configs/experiments/default_experiment.yaml"
    config = load_experiment_config(config_path)

    # Verify configuration structure
    ad_component = config.components.ad_component
    assert ad_component is not None

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

    # Check for success and metrics
    sim_result = result.simulation_results[0]
    metrics = result.metrics

    assert sim_result.success, f"Simulation failed with reason: {sim_result.reason}"

    # Detailed metric assertions
    assert metrics.success == 1, f"Metric success should be 1, got {metrics.success}"
    assert metrics.lap_time_sec < 90.0, f"Lap time {metrics.lap_time_sec:.2f}s is too slow (>= 90s)"
    assert metrics.collision_count == 0, f"Collision count {metrics.collision_count} should be 0"
    assert (
        metrics.termination_code == 1
    ), f"Termination code {metrics.termination_code} != 1 (Goal Reached)"
    assert metrics.goal_count == 1, f"Goal count {metrics.goal_count} != 1"


def test_node_instantiation(tmp_path) -> None:
    """Test direct instantiation of Nodes."""
    vp = VehicleParameters(
        wheelbase=2.5,
        width=1.8,
        front_overhang=1.0,
        rear_overhang=1.0,
        max_steering_angle=0.6,
        max_velocity=20.0,
        max_acceleration=3.0,
        mass=1500.0,
        inertia=2500.0,
        lf=1.2,
        lr=1.3,
        cf=80000.0,
        cr=80000.0,
        c_drag=0.3,
        c_roll=0.015,
        max_drive_force=5000.0,
        max_brake_force=8000.0,
    )
    # Create dummy track
    dummy_track = tmp_path / "test_track.csv"
    dummy_track.write_text(
        "x,y,z,x_quat,y_quat,z_quat,w_quat,speed\n0,0,0,0,0,0,1,10\n10,0,0,0,0,0,1,10"
    )

    # Pure Pursuit

    pp_config_dict = {
        "track_path": str(dummy_track),
        "min_lookahead_distance": 2.0,
        "max_lookahead_distance": 10.0,
        "lookahead_speed_ratio": 1.0,
    }
    pp_config = PurePursuitConfig(**pp_config_dict)
    pp_node = PurePursuitNode(config=pp_config, rate_hz=10.0, vehicle_params=vp)
    assert pp_node.name == "PurePursuit"
    assert pp_node.config.track_path == str(dummy_track)

    # PID Controller

    pid_config_dict = {"kp": 1.0, "ki": 0.1, "kd": 0.01, "u_min": -10.0, "u_max": 10.0}
    pid_config = PIDConfig(**pid_config_dict)
    pid_node = PIDControllerNode(config=pid_config, rate_hz=10.0, vehicle_params=vp)
    assert pid_node.name == "PIDController"
    assert pid_node.config.kp == 1.0

    # Sensor
    from experiment.processors.sensor import IdealSensorConfig

    sensor_config = IdealSensorConfig()
    sensor_node = IdealSensorNode(config=sensor_config, rate_hz=50.0)
    assert sensor_node.name == "Sensor"
