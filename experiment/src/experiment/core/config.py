"""Global configuration schema for the experiment system."""

from pathlib import Path
from typing import Any, Literal

from core.data import VehicleParameters
from pydantic import BaseModel, ConfigDict, Field


class ExecutionConfig(BaseModel):
    """Configuration for experiment execution."""

    model_config = ConfigDict(extra="forbid")

    num_episodes: int = Field(..., gt=0, description="Number of episodes to run")
    clock_rate_hz: float = Field(..., gt=0, description="Simulation clock rate in Hz")
    duration_sec: float = Field(
        ..., gt=0, description="Maximum duration of each episode in seconds"
    )
    clock_type: Literal["stepped", "realtime"] = Field(..., description="Clock type")
    enable_progress_bar: bool = Field(..., description="Enable progress bar")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level"
    )
    num_jobs: int | None = Field(None, gt=0, description="Total number of jobs (sweeps)")
    total_episodes: int | None = Field(None, gt=0, description="Total number of episodes across all jobs")
    base_seed: int = Field(0, ge=0, description="Base random seed for experiment")


class ObstaclePlacement(BaseModel):
    """Configuration for obstacle placement strategy."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["random_track", "random_map", "track_forward"]
    lateral_offset_range: tuple[float, float] | list[float] = Field(..., description="Lateral offset range [min, max] from centerline")
    yaw_mode: Literal["aligned", "random"] = Field(..., description="Yaw mode (aligned or random)")
    bounds: dict[str, float] | None = None
    min_distance: float = Field(0.0, ge=0.0)
    forward_distance: float | None = Field(None, ge=0.0, description="Forward distance for track_forward strategy (fixed value)")
    forward_distance_range: tuple[float, float] | list[float] | None = Field(None, description="Forward distance range [min, max] for track_forward strategy (random sampling)")
    require_within_bounds: bool = Field(False, description="Whether to require obstacle to be strictly within map bounds")
    yaw_offset_range: dict[str, float] | tuple[float, float] | list[float] = Field(..., description="Yaw offset range (min, max) for aligned mode")


class ObstacleGroup(BaseModel):
    """Configuration for a group of generated obstacles."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    count: int = Field(..., gt=0)
    placement: ObstaclePlacement
    shape: dict[str, Any] | None = None


class ExclusionZoneConfig(BaseModel):
    """Configuration for obstacle exclusion zone."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    distance: float = Field(..., gt=0)


class ObstacleGenerationConfig(BaseModel):
    """Configuration for dynamic obstacle generation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    groups: list[ObstacleGroup] = Field(default_factory=list)

    exclusion_zone: ExclusionZoneConfig = Field(...)
    seed: int = Field(..., description="Random seed for obstacle generation")


class Checkpoint(BaseModel):
    """Configuration for a checkpoint."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., description="Checkpoint x position")
    y: float = Field(..., description="Checkpoint y position")
    tolerance: float | None = Field(None, gt=0, description="Checkpoint tolerance radius")


class ObstaclesConfig(BaseModel):
    """Configuration for all obstacles."""

    model_config = ConfigDict(extra="forbid")

    generation: ObstacleGenerationConfig | None = None
    # Use 'items' to avoid shadowing builtin 'list', but alias it to 'list' for YAML compatibility
    items: list[dict[str, Any]] = Field(
        default_factory=list, description="Static list of obstacles", alias="list"
    )


class EnvironmentConfig(BaseModel):
    """Configuration for the simulation environment."""

    model_config = ConfigDict(extra="forbid")

    map_path: Path = Field(..., description="Path to the map file")
    track_path: Path = Field(..., description="Path to the reference track file")
    initial_state: dict[str, float] = Field(
        ..., description="Initial vehicle state (x, y, yaw, velocity)"
    )
    goal_position: dict[str, float] = Field(..., description="Goal position (x, y)")
    goal_tolerance: float = Field(..., gt=0, description="Goal tolerance radius")
    checkpoints: list[Checkpoint] = Field(
        default_factory=list, description="List of intermediate checkpoints"
    )
    obstacles: ObstaclesConfig = Field(..., description="Obstacle configuration")
    initial_state_sampling: dict[str, Any] | None = Field(
        None, description="Initial state sampling configuration (optional)"
    )


class PostProcessConfig(BaseModel):
    """Configuration for post-processing and logging."""

    model_config = ConfigDict(extra="ignore")

    inputs: list[str] = Field(default_factory=list)
    mlflow: dict[str, Any] = Field(default_factory=dict)
    dashboard: dict[str, Any] = Field(default_factory=dict)
    mcap: dict[str, Any] = Field(default_factory=dict)
    foxglove: dict[str, Any] = Field(default_factory=dict)


class ExperimentMetaConfig(BaseModel):
    """Metadata for the experiment."""

    model_config = ConfigDict(extra="allow")

    name: str
    type: str
    description: str


class DataLoaderConfig(BaseModel):
    """Configuration for data loading."""

    model_config = ConfigDict(extra="forbid")

    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True)
    persistent_workers: bool = Field(default=True)


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0)
    num_epochs: int = Field(..., gt=0)
    pretrained_model_path: str | None = None
    dataloader: DataLoaderConfig | None = None


class ModelConfig(BaseModel):
    """Configuration for the neural network model."""

    model_config = ConfigDict(extra="forbid")

    input_width: int = Field(..., gt=0)
    output_dim: int = Field(..., gt=0)


class NormalizationConfig(BaseModel):
    """Configuration for data normalization."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    method: Literal["standard", "minmax"]


class DVCConfig(BaseModel):
    """Configuration for DVC automation."""

    model_config = ConfigDict(extra="forbid")

    auto_add: bool
    auto_push: bool


class ExperimentConfig(BaseModel):
    """Root configuration for the experiment."""

    model_config = ConfigDict(extra="ignore")

    experiment: ExperimentMetaConfig = Field(..., description="Experiment metadata")
    execution: ExecutionConfig = Field(..., description="Execution parameters")
    system: dict[str, Any] = Field(..., description="System/Node configuration")
    env: EnvironmentConfig = Field(..., description="Environment configuration")
    vehicle: VehicleParameters = Field(..., description="Vehicle parameters")
    postprocess: PostProcessConfig = Field(..., description="Post-processing configuration")

    split: str = Field(..., description="Data split (train, val, test)")

    # Phase-specific configurations (Optional but strict if present)
    training: TrainingConfig | None = None
    model: ModelConfig | None = None
    normalization: NormalizationConfig | None = None
    dvc: DVCConfig | None = None

    # Path arguments for phases
    train_data: str | None = None
    val_data: str | None = None
    input_dir: str | None = None
    output_dir: str | None = None
    exclude_failure_reasons: list[str] | None = Field(
        None, description="Failure reasons to exclude (None=exclude all failures, []=include all)"
    )

    ad_components: dict[str, Any] | None = None
