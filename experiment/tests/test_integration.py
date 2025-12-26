import contextlib
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from core.data import VehicleParameters
from experiment.core.orchestrator import ExperimentOrchestrator
from ideal_sensor.sensor_node import IdealSensorConfig, IdealSensorNode
from pid_controller.pid_controller_node import PIDConfig, PIDControllerNode
from pure_pursuit.pure_pursuit_node import PurePursuitConfig, PurePursuitNode


@pytest.mark.integration
@patch("experiment.engine.base.mlflow")
@patch("experiment.engine.evaluator.mlflow")
def test_pure_pursuit_experiment_nodes(_mock_mlflow_eval, _mock_mlflow_base) -> None:
    """Test Pure Pursuit experiment execution with Hydra configuration."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Load Hydra config files manually
    workspace_root = Path(__file__).parent.parent.parent
    config_dir = str(workspace_root / "experiment/conf")

    # Create tmp directory for MCAP output
    tmp_path = workspace_root / "tmp"
    tmp_path.mkdir(exist_ok=True)

    # Use Hydra's compose to properly handle defaults
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=evaluation",
                "agent=pure_pursuit",
                "execution.duration_sec=200.0",
                "execution.num_episodes=1",
            ],
        )

        # Manually resolve hydra interpolations for testing
        OmegaConf.set_struct(cfg, False)
        cfg["hydra"] = {"runtime": {"output_dir": str(tmp_path)}}
        cfg["output_dir"] = str(tmp_path)

        # Replace ${hydra:runtime.output_dir} in postprocess config
        if "postprocess" in cfg and "mcap" in cfg.postprocess:
            cfg.postprocess.mcap.output_dir = str(tmp_path)

        # Replace ${hydra:runtime.output_dir} in Logger params
        nodes_iter = (
            cfg.system.nodes.values() if isinstance(cfg.system.nodes, dict) else cfg.system.nodes
        )
        for node in nodes_iter:
            node_name = node.get("name") if isinstance(node, dict) else getattr(node, "name", None)
            if node_name == "Logger" and "output_mcap_path" in (
                node.get("params", {}) if isinstance(node, dict) else node.params
            ):
                if isinstance(node, dict):
                    node["params"]["output_mcap_path"] = str(tmp_path)
                else:
                    node.params.output_mcap_path = str(tmp_path)

        OmegaConf.set_struct(cfg, True)

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
        print(f"Metrics: {metrics}")
        print(f"Sim Result Metrics: {sim_result.metrics}")

    # We aim for success now
    assert sim_result.success, f"Simulation failed: {sim_result.reason}"

    assert metrics.collision_count == 0, f"Collision count {metrics.collision_count} should be 0"
    assert metrics.termination_code != 5, (
        f"Termination code {metrics.termination_code} should not be 5 (Collision)"
    )
    # Goal reached
    assert metrics.goal_count == 1, f"Goal count {metrics.goal_count} != 1 (Goal)"
    assert metrics.checkpoint_count == 3, (
        f"Checkpoint count {metrics.checkpoint_count} != 3 (Checkpoints)"
    )
    assert sim_result.metrics.get("goal_count") == 1, "Per-episode goal count mismatch"
    assert sim_result.metrics.get("checkpoint_count") == 3, "Per-episode checkpoint count mismatch"

    # Move MCAP file from episode subdirectory to tmp root for user visibility
    mcap_source = tmp_path / "episode_0000" / "simulation.mcap"
    mcap_path = tmp_path / "simulation.mcap"

    if mcap_source.exists():
        if mcap_path.exists():
            mcap_path.unlink()

        with contextlib.suppress(shutil.SameFileError):
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
    if dashboard_artifact.local_path.absolute() != target_dashboard.absolute():
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
        tire_params={},
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
