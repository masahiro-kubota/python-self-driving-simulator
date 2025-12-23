import shutil
from pathlib import Path

import pytest
from ideal_sensor.sensor_node import IdealSensorConfig, IdealSensorNode
from pid_controller.pid_controller_node import PIDConfig, PIDControllerNode
from pure_pursuit.pure_pursuit_node import PurePursuitConfig, PurePursuitNode

from core.data import VehicleParameters
from experiment.core.orchestrator import ExperimentOrchestrator


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch, tmp_path) -> None:
    """Set a temporary MLflow tracking URI for tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlflow")


@pytest.mark.integration
def test_pure_pursuit_experiment_nodes() -> None:
    """Test Pure Pursuit experiment execution with Hydra configuration."""
    from omegaconf import OmegaConf

    # Load Hydra config files manually
    workspace_root = Path(__file__).parent.parent.parent
    config_dir = workspace_root / "experiment/conf"

    # Create tmp directory for MCAP output
    tmp_path = workspace_root / "tmp"
    tmp_path.mkdir(exist_ok=True)

    # Manually compose config by loading each component
    env_cfg = OmegaConf.load(config_dir / "env/default.yaml")
    vehicle_cfg = OmegaConf.load(config_dir / "vehicle/default.yaml")
    agent_cfg = OmegaConf.load(config_dir / "agent/pure_pursuit.yaml")
    experiment_cfg = OmegaConf.load(config_dir / "experiment/evaluation.yaml")

    # Merge configs
    cfg = OmegaConf.merge(
        experiment_cfg, {"env": env_cfg, "vehicle": vehicle_cfg, "agent": agent_cfg}
    )

    # Override for testing
    cfg.execution.duration_sec = 200.0
    cfg.execution.num_episodes = 1
    cfg.output_dir = str(tmp_path)
    cfg.postprocess.mcap.output_dir = str(tmp_path)

    # Manually replace Hydra interpolations before resolution
    OmegaConf.set_struct(cfg, False)
    cfg.hydra = {"runtime": {"output_dir": str(tmp_path)}}
    # Replace ${hydra:runtime.output_dir} in Logger params
    for node in cfg.system.nodes:
        if node.name == "Logger" and "output_mcap_path" in node.params:
            node.params.output_mcap_path = str(tmp_path)
    OmegaConf.set_struct(cfg, True)

    # Resolve all interpolations
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # Run experiment with Hydra config
    orchestrator = ExperimentOrchestrator()
    result = orchestrator.run_from_hydra(cfg)

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

    if not sim_result.success:
        print(f"Simulation failed with reason: {sim_result.reason}")

    # We aim for success now
    assert sim_result.success, f"Simulation failed: {sim_result.reason}"

    assert metrics.collision_count == 0, f"Collision count {metrics.collision_count} should be 0"
    assert (
        metrics.termination_code != 5
    ), f"Termination code {metrics.termination_code} should not be 5 (Collision)"
    # Goal reached
    assert metrics.goal_count == 1, f"Goal count {metrics.goal_count} != 1"

    # Move MCAP file from episode subdirectory to tmp root for user visibility
    mcap_source = tmp_path / "episode_0000" / "simulation.mcap"
    mcap_path = tmp_path / "simulation.mcap"

    if mcap_source.exists():
        if mcap_path.exists():
            mcap_path.unlink()
        shutil.move(mcap_source, mcap_path)

    assert mcap_path.exists(), f"MCAP file not found at {mcap_path}"

    # Simple verification using mcap library
    from mcap.reader import make_reader

    print("\nMCAP Verification:")
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        assert summary is not None

        # Check statistics
        print(
            f"  Duration: {summary.statistics.message_start_time} - {summary.statistics.message_end_time}"
        )
        print(f"  Message Count: {summary.statistics.message_count}")
        print(f"  Channel Count: {summary.statistics.channel_count}")

        # Verify specific topics exist
        topics = [c.topic for c in summary.channels.values()]
        print(f"  Topics: {topics}")

        # NOTE: Lidar verification skipped as per user request
        expected_topics = [
            "/tf",
            "/localization/kinematic_state",
            # "/perception/lidar/scan", # Disabled
            "/simulation/info",
            "/control/command/control_cmd",
            "/map/vector",
        ]
        for topic in expected_topics:
            assert topic in topics, f"Topic {topic} missing in MCAP"

        print("  MCAP verification passed.")

    # Verify Dashboard HTML exists in artifacts and save to tmp for inspection
    dashboard_artifact = next((a for a in result.artifacts if a.local_path.suffix == ".html"), None)
    assert dashboard_artifact is not None, "Dashboard HTML artifact not found"

    # Copy dashboard to tmp for user visibility
    target_dashboard = tmp_path / "dashboard.html"
    shutil.copy(dashboard_artifact.local_path, target_dashboard)
    print(f"  Dashboard copied to {target_dashboard}")


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
        "vehicle_params": vp,
    }
    pp_config = PurePursuitConfig(**pp_config_dict)
    pp_node = PurePursuitNode(config=pp_config, rate_hz=10.0)
    assert pp_node.name == "PurePursuit"
    assert pp_node.config.track_path == dummy_track

    # PID Controller

    pid_config_dict = {
        "kp": 1.0,
        "ki": 0.1,
        "kd": 0.01,
        "u_min": -10.0,
        "u_max": 10.0,
        "vehicle_params": vp,
    }
    pid_config = PIDConfig(**pid_config_dict)
    pid_node = PIDControllerNode(config=pid_config, rate_hz=10.0)
    assert pid_node.name == "PIDController"
    assert pid_node.config.kp == 1.0

    # Sensor
    sensor_config = IdealSensorConfig()
    sensor_node = IdealSensorNode(config=sensor_config, rate_hz=50.0)
    assert sensor_node.name == "Sensor"
