from pathlib import Path

from core.data import ComponentConfig, Trajectory, VehicleParameters, VehicleState
from core.data.node_io import NodeIO
from core.data.ros import ColorRGBA, Header, Marker, Point, Time, Vector3
from core.interfaces.node import Node, NodeExecutionResult
from pydantic import Field

from mppi_planner.mppi import MPPIController


class MPPIPlannerConfig(ComponentConfig):
    """Configuration for MPPIPlannerNode."""

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
    position_weight: float = Field(..., description="Tracking cost weight")


class MPPIPlannerNode(Node[MPPIPlannerConfig]):
    """MPPI Path Planner Node."""

    def __init__(self, config: MPPIPlannerConfig, rate_hz: float, priority: int):
        super().__init__("MPPIPlanner", rate_hz, config, priority)

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
            position_weight=config.position_weight,
        )

    def get_node_io(self) -> NodeIO:
        return NodeIO(
            inputs={
                "vehicle_state": VehicleState,
                "obstacles": list,  # List[SimulatorObstacle]
            },
            outputs={
                "trajectory": Trajectory,
            },
        )

    def on_run(self, current_time: float) -> NodeExecutionResult:
        if self.frame_data is None:
            return NodeExecutionResult.FAILED

        # Get Inputs
        vehicle_state = self.subscribe("vehicle_state")
        obstacles = self.subscribe("obstacles") or []

        if vehicle_state is None:
            return NodeExecutionResult.SKIPPED

        # Run MPPI
        trajectory, _, states = self.controller.solve(
            initial_state=vehicle_state,
            reference_trajectory=self.reference_trajectory,
            obstacles=obstacles,
        )

        # Set Trajectory Output
        self.publish("trajectory", trajectory)

        # Generate Debug Markers (Candidate Paths)
        # states: (K, T+1, 4)
        num_samples, horizon, _ = states.shape
        markers = []

        # Determine timestamp
        sec = int(current_time)
        nanosec = int((current_time - sec) * 1e9)
        stamp = Time(sec=sec, nanosec=nanosec)

        # Subsample for performance if needed, but "all paths" requested.
        # Ensure imports are available

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
                color=ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.1),
                points=points,
                lifetime=Time(sec=0, nanosec=0),
                frame_locked=False,
            )
            markers.append(marker)

        # Note: Candidate paths visualization is now available via MCAP recording

        return NodeExecutionResult.SUCCESS
