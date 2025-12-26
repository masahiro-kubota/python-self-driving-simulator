import numpy as np
import pytest
from mppi_planner.mppi import MPPIController
from mppi_planner.mppi_planner_node import MPPIPlannerConfig

from core.data import (
    ObstacleShape,
    SimulatorObstacle,
    StaticObstaclePosition,
    Trajectory,
    TrajectoryPoint,
    VehicleParameters,
    VehicleState,
)


@pytest.fixture
def vehicle_params():
    return VehicleParameters(
        wheelbase=1.0,  # L=1.0 for simple math
        width=1.0,
        front_overhang=0.0,
        rear_overhang=0.0,
        max_steering_angle=1.0,
        max_velocity=10.0,
        max_acceleration=1.0,
        mass=1000.0,
        inertia=1000.0,
        lf=0.5,
        lr=0.5,
        cf=1.0,
        cr=1.0,
        c_drag=0.0,
        c_roll=0.0,
        max_drive_force=100.0,
        max_brake_force=100.0,
        tire_params={},
    )


@pytest.fixture
def mppi_config(vehicle_params):
    return MPPIPlannerConfig(
        track_path="dummy.csv",
        map_path="dummy.osm",
        vehicle_params=vehicle_params,
        horizon=10,
        dt=0.1,
        num_samples=5,
        temperature=1.0,
        noise_sigma_steering=0.1,
        seed=42,  # Fixed seed for reproducible tests
        obstacle_cost_weight=1000.0,
        collision_threshold=0.5,
        off_track_cost_weight=10000.0,
    )


@pytest.fixture
def mppi_controller(vehicle_params, mppi_config):
    # Instantiate with explicit args as passed in Node
    return MPPIController(
        vehicle_params=vehicle_params,
        horizon=mppi_config.horizon,
        dt=mppi_config.dt,
        num_samples=mppi_config.num_samples,
        temperature=mppi_config.temperature,
        noise_sigma_steering=mppi_config.noise_sigma_steering,
        u_min_steering=-vehicle_params.max_steering_angle,
        u_max_steering=vehicle_params.max_steering_angle,
        seed=mppi_config.seed,
        obstacle_cost_weight=mppi_config.obstacle_cost_weight,
        collision_threshold=mppi_config.collision_threshold,
        lanelet_map=None,  # No map for unit tests
        off_track_cost_weight=mppi_config.off_track_cost_weight,
    )


def test_initialization(mppi_controller):
    assert mppi_controller.T == 10
    assert mppi_controller.dt == 0.1
    assert mppi_controller.K == 5
    # U should be 1D (T, 1)
    assert mppi_controller.U.shape == (10, 1)


def test_rollout_dynamics(mppi_controller):
    # Test straight line dynamics
    # Initial state: x=0, y=0, yaw=0, v=1.0
    init_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=1.0)

    # Control: 0 steering.
    # u_samples shape (K, T, 1)
    u_samples = np.zeros((1, 10, 1))

    # Target velocity matches current velocity (constant speed)
    target_v = 1.0

    states = mppi_controller._rollout(init_state, u_samples, target_v)

    # Validating first simulated step
    # accel should be 0 because v=target_v
    # x_next = x + v * cos(yaw) * dt = 0 + 1 * 1 * 0.1 = 0.1
    # y_next = 0
    # yaw_next = 0
    # v_next = 1

    assert np.isclose(states[0, 1, 0], 0.1)
    assert np.isclose(states[0, 1, 1], 0.0)
    assert np.isclose(states[0, 1, 2], 0.0)
    assert np.isclose(states[0, 1, 3], 1.0)

    # Validating 10th step (t=1.0s) -> x should be 1.0
    assert np.isclose(states[0, 10, 0], 1.0)


def test_cost_function_obstacle(mppi_controller):
    # Setup standard straight trajectory reference
    ref_points = [TrajectoryPoint(x=i * 0.1, y=0.0, yaw=0.0, velocity=1.0) for i in range(20)]
    reference = Trajectory(points=ref_points)

    # Two samples:
    # 0: straight (collision candidate)
    # 1: avoid (offset y)

    num_samples = 2
    horizon = 10
    trajectories = np.zeros((num_samples, horizon + 1, 4))

    # Trajectory 0: Straight on x-axis (y=0)
    for t in range(horizon + 1):
        trajectories[0, t, 0] = t * 0.1
        trajectories[0, t, 1] = 0.0

    # Trajectory 1: Offset (y=2.0)
    for t in range(horizon + 1):
        trajectories[1, t, 0] = t * 0.1
        trajectories[1, t, 1] = 2.0

    # 1D controls
    controls = np.zeros((num_samples, horizon, 1))
    epsilon = np.zeros((num_samples, horizon, 1))

    # Place obstacle at (0.5, 0.0) with radius 0.2
    obs1 = SimulatorObstacle(
        type="static",
        shape=ObstacleShape(type="circle", radius=0.2),
        position=StaticObstaclePosition(x=0.5, y=0.0, yaw=0.0),
    )
    obstacles = [obs1]

    costs = mppi_controller._compute_costs(trajectories, controls, epsilon, reference, obstacles)

    assert costs[0] > costs[1]
    assert costs[0] > 1000.0


def test_solve_runs(mppi_controller):
    # Just verify solve runs without error and returns trajectory
    init_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=1.0)
    ref_points = [TrajectoryPoint(x=i * 1.0, y=0.0, yaw=0.0, velocity=1.0) for i in range(20)]
    reference = Trajectory(points=ref_points)
    obstacles = []

    traj, controls = mppi_controller.solve(init_state, reference, obstacles)

    assert isinstance(traj, Trajectory)
    assert len(traj.points) == 10
    # Solve returns rounded 2D controls now
    assert controls.shape == (10, 2)


def test_straight_line_tracking(mppi_controller):
    """Verify that for a perfectly straight reference, the planner drives straight."""
    init_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0)

    ref_points = [TrajectoryPoint(x=i * 1.0, y=0.0, yaw=0.0, velocity=5.0) for i in range(20)]
    reference = Trajectory(points=ref_points)
    obstacles = []

    traj, controls = mppi_controller.solve(init_state, reference, obstacles)

    # Check Controls: Steering should be very close to 0
    steering_angles = controls[:, 0]
    avg_steering = np.mean(np.abs(steering_angles))

    print(f"Average absolute steering: {avg_steering}")
    assert avg_steering < 0.1, f"Steering is too high for straight line: {avg_steering}"

    y_positions = [p.y for p in traj.points]
    max_y_deviation = np.max(np.abs(y_positions))

    print(f"Max Y deviation: {max_y_deviation}")
    assert max_y_deviation < 0.5, f"Trajectory deviates too much in Y: {max_y_deviation}"


def test_start_from_stop(mppi_controller):
    """Verify that from stop, with reference ahead, the vehicle accelerates forward (via P-control)."""
    init_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0)

    # Reference path: Straight ahead, with 5.0 m/s
    ref_points = [TrajectoryPoint(x=i * 1.0, y=0.0, yaw=0.0, velocity=5.0) for i in range(20)]
    reference = Trajectory(points=ref_points)
    obstacles = []

    traj, controls = mppi_controller.solve(init_state, reference, obstacles)

    # Check Acceleration (should be positive due to P-control)
    accels = controls[:, 1]
    avg_accel = np.mean(accels)

    print(f"Average accel: {avg_accel}")
    # accel = 1.0 * (5.0 - 0) = 5.0 (clipped to 3.0)
    # Should be close to 3.0 ideally, or positive.
    assert avg_accel > 0.5, f"Vehicle failing to accelerate forward: {avg_accel}"

    velocities = [p.velocity for p in traj.points]
    assert velocities[-1] > 0.1, "Final velocity should be positive"
