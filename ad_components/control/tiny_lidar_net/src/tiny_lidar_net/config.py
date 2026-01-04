"""Configuration for Tiny LiDAR Net node."""

from pathlib import Path

from core.data import ComponentConfig, VehicleParameters
from pydantic import Field


class TinyLidarNetConfig(ComponentConfig):
    """Configuration for TinyLidarNetNode."""

    model_path: Path = Field(..., description="Path to .npy weights file")
    input_dim: int = Field(..., description="LiDAR input dimension (number of beams)")
    output_dim: int = Field(..., description="Output dimension (acceleration + steering)")
    architecture: str = Field(..., description="Model architecture ('large' or 'small')")
    max_range: float = Field(..., description="Maximum LiDAR range for normalization [m]")
    target_velocity: float = Field(..., description="Target velocity [m/s]")
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
    control_cmd_topic: str = Field(
        "control_cmd", description="Output topic name for control command"
    )
