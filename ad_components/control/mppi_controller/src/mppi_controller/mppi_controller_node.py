from pathlib import Path

from core.data import Action, ComponentConfig, VehicleParameters, VehicleState
from core.data.node_io import NodeIO
from core.data.ros import ColorRGBA, Header, Marker, MarkerArray, Point, Time, Vector3
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import Field

from mppi_controller.mppi import MPPIController


class MPPIControllerConfig(ComponentConfig):
    """Configuration for MPPIControllerNode."""

    track_path: Path = Field(..., description="Path to reference trajectory CSV")
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    map_path: Path = Field(..., description="Path to Lanelet2 map file")

    # MPPI Parameters
    horizon: int = Field(..., description="Prediction horizon steps")
    dt: float = Field(..., description="Time step [s]")
    num_samples: int = Field(..., description="Number of samples")
    temperature: float = Field(..., description="Temperature (lambda)")
    noise_sigma_steering: float = Field(..., description="Steering noise sigma")
    seed: int | None = Field(None, description="Random seed for reproducibility")

    # Obstacle avoidance parameters
    obstacle_cost_weight: float = Field(..., description="Obstacle avoidance cost weight")
    collision_threshold: float = Field(..., description="Collision threshold distance [m]")

    # Map boundary parameters
    off_track_cost_weight: float = Field(..., description="Off-track penalty weight")

    # Cost weights
    position_weight: float = Field(20.0, description="Tracking cost weight")

    # Velocity control
    target_velocity: float = Field(5.0, description="Target velocity [m/s]")

    # Visualization colors
    candidate_color: str = Field("#00FFFF1A", description="Candidate trajectories color")
    optimal_color: str = Field("#FF0000CC", description="Optimal trajectory color")


class MPPIControllerNode(Node[MPPIControllerConfig]):
    """MPPI Controller Node.

    This node uses Model Predictive Path Integral (MPPI) control to compute
    optimal control commands (steering angle and acceleration) that track a
    reference trajectory while avoiding obstacles and staying within map boundaries.
    """

    def __init__(self, config: MPPIControllerConfig, rate_hz: float):
        super().__init__("MPPIController", rate_hz, config)

        from core.utils import get_project_root

        # Load Map
        from simulator.map import LaneletMap

        map_path = self.config.map_path
        if not map_path.is_absolute():
            map_path = get_project_root() / map_path

        self.lanelet_map = LaneletMap(map_path)

        # Load Reference Trajectory
        from planning_utils import load_track_csv

        track_path = self.config.track_path
        if not track_path.is_absolute():
            track_path = get_project_root() / track_path

        self.reference_trajectory = load_track_csv(track_path)

        # Initialize Controller
        self.controller = MPPIController(
            vehicle_params=config.vehicle_params,
            horizon=config.horizon,
            dt=config.dt,
            num_samples=config.num_samples,
            temperature=config.temperature,
            noise_sigma_steering=config.noise_sigma_steering,
            u_min_steering=-config.vehicle_params.max_steering_angle,
            u_max_steering=config.vehicle_params.max_steering_angle,
            seed=config.seed,
            obstacle_cost_weight=config.obstacle_cost_weight,
            collision_threshold=config.collision_threshold,
            lanelet_map=self.lanelet_map,
            off_track_cost_weight=config.off_track_cost_weight,
            target_velocity=config.target_velocity,
            position_weight=config.position_weight,
        )

    def get_node_io(self) -> NodeIO:
        return NodeIO(
            inputs={
                "vehicle_state": VehicleState,
                "obstacles": list,  # List[SimulatorObstacle]
            },
            outputs={
                "action": Action,
                "mppi_candidates": MarkerArray,
                "mppi_optimal": MarkerArray,
            },
        )

    def on_run(self, current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        # Get Inputs
        vehicle_state = getattr(self.frame_data, "vehicle_state", None)
        obstacles = getattr(self.frame_data, "obstacles", [])

        if vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        # Run MPPI
        trajectory, controls, states = self.controller.solve(
            initial_state=vehicle_state,
            reference_trajectory=self.reference_trajectory,
            obstacles=obstacles,
        )

        # Extract first control command (MPC receding horizon)
        # controls: (T, 2) where each row is [steering_angle, acceleration]
        steering_angle = float(controls[0, 0])
        acceleration = float(controls[0, 1])

        # Output Action
        self.frame_data.action = Action(
            steering=steering_angle,
            acceleration=acceleration,
            timestamp=current_time,
        )

        # Generate Debug Markers (Candidate Paths + Optimal Trajectory)
        # states: (K, T+1, 4)
        num_samples, horizon, _ = states.shape
        markers = []

        # Determine timestamp
        sec = int(current_time)
        nanosec = int((current_time - sec) * 1e9)
        stamp = Time(sec=sec, nanosec=nanosec)

        # Add candidate trajectories
        for k in range(num_samples):
            points = []
            for t in range(horizon):
                # Use x, y from state
                points.append(Point(x=float(states[k, t, 0]), y=float(states[k, t, 1]), z=0.0))

            marker = Marker(
                header=Header(stamp=stamp, frame_id="map"),
                ns="mppi_candidates",
                id=k,
                type=4,  # LINE_STRIP
                action=0,  # ADD
                scale=Vector3(x=0.05, y=0.0, z=0.0),
                color=ColorRGBA.from_hex(self.config.candidate_color),
                points=points,
                lifetime=Time(sec=0, nanosec=0),
                frame_locked=False,
            )
            markers.append(marker)

        self.frame_data.mppi_candidates = MarkerArray(markers=markers)

        # Add optimal trajectory (for debugging)
        optimal_points = []
        for point in trajectory.points:
            optimal_points.append(Point(x=point.x, y=point.y, z=0.0))

        optimal_marker = Marker(
            header=Header(stamp=stamp, frame_id="map"),
            ns="mppi_optimal",
            id=0,
            type=4,  # LINE_STRIP
            action=0,  # ADD
            scale=Vector3(x=0.1, y=0.0, z=0.0),
            color=ColorRGBA.from_hex(self.config.optimal_color),
            points=optimal_points,
            lifetime=Time(sec=0, nanosec=0),
            frame_locked=False,
        )
        self.frame_data.mppi_optimal = MarkerArray(markers=[optimal_marker])

        return NodeExecutionResult.SUCCESS
