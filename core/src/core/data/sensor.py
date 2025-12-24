from pydantic import BaseModel, Field

from core.data.node import ComponentConfig


class LidarConfig(ComponentConfig):
    """Configuration for LiDAR sensor."""

    num_beams: int = Field(..., gt=0)
    fov: float = Field(..., gt=0)
    range_min: float = Field(...)
    range_max: float = Field(...)
    angle_increment: float = Field(...)
    # Mounting position relative to vehicle center
    x: float = Field(...)
    y: float = Field(...)
    z: float = Field(...)
    yaw: float = Field(...)


class LidarScan(BaseModel):
    """LiDAR scan data."""

    timestamp: float
    config: LidarConfig
    ranges: list[float]  # inf for no return
    intensities: list[float] | None = None
