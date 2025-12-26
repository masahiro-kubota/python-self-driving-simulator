"""Vehicle dynamics functions."""

import math
from typing import TYPE_CHECKING

from core.utils.geometry import normalize_angle

from simulator.state import SimulationVehicleState

if TYPE_CHECKING:
    from core.data import VehicleParameters, VehicleState
    from shapely.geometry import Polygon


def update_bicycle_model(
    state: SimulationVehicleState,
    steering: float,
    acceleration: float,
    dt: float,
    wheelbase: float,
) -> SimulationVehicleState:
    """Update state using kinematic bicycle model.

    Args:
        state: Current simulation state
        steering: Steering angle [rad]
        acceleration: Acceleration [m/s^2]
        dt: Time step [s]
        wheelbase: Vehicle wheelbase [m]

    Returns:
        Updated state
    """
    # Kinematic bicycle model equations
    # x_dot = vx * cos(yaw) - vy * sin(yaw)
    # y_dot = vx * sin(yaw) + vy * cos(yaw)
    # yaw_dot = vx / L * tan(delta)
    # vx_dot = ax

    # Kinamatics model assumes vy = 0
    vx_next = state.vx + acceleration * dt

    # Use average velocity for position update
    vx_avg = (state.vx + vx_next) / 2.0

    # Calculate yaw rate (using average velocity)
    if abs(vx_avg) < 0.01:
        yaw_rate_next = 0.0
    else:
        yaw_rate_next = vx_avg / wheelbase * math.tan(steering)

    # Update position and orientation (using average velocity)
    x_next = state.x + vx_avg * math.cos(state.yaw) * dt
    y_next = state.y + vx_avg * math.sin(state.yaw) * dt
    yaw_next = normalize_angle(state.yaw + yaw_rate_next * dt)

    # Update timestamp if present
    timestamp_next: float | None = None
    if state.timestamp is not None:
        timestamp_next = state.timestamp + dt

    return SimulationVehicleState(
        # Position (2D update, z=0 fixed)
        x=x_next,
        y=y_next,
        z=0.0,
        # Orientation (yaw only, roll=pitch=0 fixed)
        roll=0.0,
        pitch=0.0,
        yaw=yaw_next,
        # Velocity (vx only, vy=vz=0 fixed)
        vx=vx_next,
        vy=0.0,
        vz=0.0,
        # Angular velocity (yaw_rate only)
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=yaw_rate_next,
        # Acceleration
        ax=acceleration,
        ay=0.0,
        az=0.0,
        # Input
        steering=steering,
        throttle=0.0,
        # Timestamp
        timestamp=timestamp_next,
    )


def create_vehicle_polygon(
    x: float,
    y: float,
    yaw: float,
    front_edge_dist: float,
    rear_edge_dist: float,
    half_width: float,
) -> "Polygon":
    """Create vehicle polygon from parameters.

    Args:
        x: Reference point X
        y: Reference point Y
        yaw: Yaw angle
        front_edge_dist: Distance to front edge from reference
        rear_edge_dist: Distance to rear edge from reference
        half_width: Half width of vehicle

    Returns:
        Shapely Polygon
    """
    from shapely.geometry import Polygon

    # Vehicle frame coordinates (x forward, y left)
    p1 = (front_edge_dist, half_width)
    p2 = (front_edge_dist, -half_width)
    p3 = (rear_edge_dist, -half_width)
    p4 = (rear_edge_dist, half_width)

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    points = []
    for px, py in [p1, p2, p3, p4]:
        # Rotate
        rx = px * cos_yaw - py * sin_yaw
        ry = px * sin_yaw + py * cos_yaw
        # Translate
        tx = rx + x
        ty = ry + y
        points.append((tx, ty))

    return Polygon(points)


def get_bicycle_model_polygon(
    state: "VehicleState",
    params: "VehicleParameters",
) -> "Polygon":
    """Get vehicle polygon for kinematic model (rear-axle centered)."""
    # Kinematic model is rear-axle centered
    front_edge = params.wheelbase + params.front_overhang
    rear_edge = -params.rear_overhang

    return create_vehicle_polygon(
        x=state.x,
        y=state.y,
        yaw=state.yaw,
        front_edge_dist=front_edge,
        rear_edge_dist=rear_edge,
        half_width=params.width / 2.0,
    )
