"""Vehicle visualizer for creating vehicle markers."""

from typing import Any

from core.data import VehicleState
from core.data.ros import ColorRGBA, Header, Marker, Point, Pose, Vector3
from logger.ros_message_builder import quaternion_from_yaw, to_ros_time


class VehicleVisualizer:
    """Creates visualization markers for the vehicle."""

    def __init__(self, vehicle_params: Any) -> None:
        """Initialize vehicle visualizer.

        Args:
            vehicle_params: Vehicle parameters (VehicleParameters object).
        """
        self.vehicle_params = vehicle_params
        self._calculate_dimensions()

    def _calculate_dimensions(self) -> None:
        """Calculate vehicle dimensions from parameters."""
        # Assume object/DictConfig access
        self.length = (
            self.vehicle_params.wheelbase
            + self.vehicle_params.front_overhang
            + self.vehicle_params.rear_overhang
        )
        self.width = self.vehicle_params.width

        self.height = 1.5

    def create_marker(self, vehicle_state: VehicleState, timestamp: float) -> Marker:
        """Create a marker for the vehicle.

        Args:
            vehicle_state: Current vehicle state.
            timestamp: Current timestamp.

        Returns:
            Marker representing the vehicle.
        """
        import math

        ros_time = to_ros_time(timestamp)
        q = quaternion_from_yaw(vehicle_state.yaw)

        # vehicle_state (x, y) is at rear axle center (base_link)
        # CUBE marker is centered at its pose, so we need to offset it
        # to the geometric center of the vehicle
        # Offset from rear axle to geometric center in vehicle frame (x-forward)
        offset_x = (
            self.vehicle_params.wheelbase
            + self.vehicle_params.front_overhang
            - self.vehicle_params.rear_overhang
        ) / 2.0

        # Transform offset to global frame
        marker_x = vehicle_state.x + offset_x * math.cos(vehicle_state.yaw)
        marker_y = vehicle_state.y + offset_x * math.sin(vehicle_state.yaw)

        return Marker(
            header=Header(stamp=ros_time, frame_id="map"),
            ns="vehicle",
            id=0,
            type=1,  # CUBE
            action=0,
            pose=Pose(
                position=Point(x=marker_x, y=marker_y, z=self.height / 2),
                orientation=q,
            ),
            scale=Vector3(x=self.length, y=self.width, z=self.height),
            color=ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8),
            frame_locked=True,
        )
