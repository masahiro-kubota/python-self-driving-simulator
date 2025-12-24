"""Configuration for Tiny LiDAR Net node."""

from pathlib import Path

from pydantic import Field

from core.data import ComponentConfig, VehicleParameters


class TinyLidarNetConfig(ComponentConfig):
    """Configuration for TinyLidarNetNode."""

    model_path: Path = Field(..., description="Path to .npy weights file")
    input_dim: int = Field(..., description="LiDAR input dimension (number of beams)")
    output_dim: int = Field(..., description="Output dimension (acceleration, steering)")
    architecture: str = Field(..., description="Model architecture ('large' or 'small')")
    max_range: float = Field(..., description="Maximum LiDAR range for normalization [m]")
    control_mode: str = Field(..., description="Control mode ('ai' or 'fixed')")
    fixed_acceleration: float = Field(
        ..., description="Fixed acceleration when control_mode='fixed'"
    )
    vehicle_params: VehicleParameters = Field(..., description="Vehicle parameters")
