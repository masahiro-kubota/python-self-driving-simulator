"""Configuration models for experiment runner."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.interfaces.node import StrictConfig


class ExperimentType(str, Enum):
    """Experiment type enumeration."""

    EVALUATION = "evaluation"


class ComponentConfig(BaseModel):
    """Configuration for a component (AD component nodes container)."""

    params: dict[str, Any] = Field(..., description="Component parameters (contains nodes list)")


class SimulatorConfig(BaseModel):
    """Configuration for simulator."""

    type: str = Field(..., description="Simulator class name")
    rate_hz: float = Field(..., description="Physics frequency in Hz")
    params: dict[str, Any] = Field(..., description="Simulator parameters")


class ExecutionConfig(BaseModel):
    """Configuration for execution."""

    num_episodes: int = Field(..., description="Number of episodes to run")
    clock_rate_hz: float = Field(
        ..., description="Clock rate in Hz (should match simulator.rate_hz for efficiency)"
    )
    duration_sec: float = Field(..., description="Simulation duration in seconds")
    parallel: bool = Field(..., description="Run episodes in parallel")
    num_workers: int = Field(..., description="Number of parallel workers")

    # Future support for executor/clock switching
    clock_type: Literal["stepped", "realtime", "external"] = Field(
        ..., description="Clock type for simulation timing"
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    metrics: list[str] = Field(
        ...,
        description="Metrics to calculate",
    )
    generate_dashboard: bool = Field(..., description="Generate dashboard")


class MLflowConfig(StrictConfig):
    """Configuration for MLflow logging."""

    enabled: bool = Field(..., description="Enable MLflow logging")
    tracking_uri: str = Field(..., description="MLflow tracking URI")


class MCAPConfig(StrictConfig):
    """Configuration for MCAP logging."""

    enabled: bool = Field(..., description="Enable MCAP logging")
    output_dir: str = Field(..., description="Output directory for MCAP files")


class DashboardConfig(StrictConfig):
    """Configuration for dashboard generation."""

    enabled: bool = Field(..., description="Enable dashboard generation")
    map_path: str = Field(..., description="Path to Lanelet2 OSM map file for visualization")
    vehicle_config_path: str = Field(..., description="Path to vehicle configuration YAML file")


class PostprocessConfig(StrictConfig):
    """Configuration for postprocessing (logging, metrics, dashboard)."""

    inputs: list[str] = Field(..., description="List of input files to log as artifacts")
    mlflow: MLflowConfig = Field(...)
    mcap: MCAPConfig = Field(...)
    dashboard: DashboardConfig = Field(...)


class ExperimentMetadata(BaseModel):
    """Experiment metadata."""

    name: str = Field(..., description="Experiment name")
    type: ExperimentType = Field(..., description="Experiment type")
    description: str = Field(..., description="Experiment description")


class ComponentsConfig(BaseModel):
    """Configuration for all components."""

    ad_component: ComponentConfig = Field(..., description="AD component config")


class ModelCheckpointConfig(BaseModel):
    """Configuration for model checkpointing."""

    save_best: bool = Field(..., description="Save best model")
    save_every_n_epochs: int = Field(..., description="Save checkpoint every N epochs")
    output_dir: str = Field(..., description="Output directory for models")


class ModuleConfig(BaseModel):
    """Module configuration (Pipeline definition)."""

    name: str = Field(..., description="Module name")
    components: dict[str, Any] = Field(..., description="Component definitions")


class SystemConfig(BaseModel):
    """System configuration (Environment settings)."""

    name: str = Field(..., description="System name")
    module: str = Field(..., description="Path to module configuration")
    vehicle: dict[str, Any] = Field(..., description="Vehicle configuration")
    map_path: str | None = Field(None, description="Path to map file (e.g. Lanelet2 OSM)")
    simulator: dict[str, Any] | None = Field(None, description="Simulator configuration")
    simulator_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Simulator parameter overrides (Deprecated in favor of 'simulator')",
    )
    runtime: dict[str, Any] | None = Field(None, description="Runtime configuration")


class ExperimentLayerConfig(BaseModel):
    """Experiment layer configuration."""

    # Experiment metadata (can be at root or nested under 'experiment' key)
    name: str = Field(..., description="Experiment name")
    type: ExperimentType = Field(..., description="Experiment type")
    description: str = Field("", description="Experiment description")

    # Configuration fields
    system: str = Field(..., description="Path to system configuration")
    execution: ExecutionConfig | None = Field(None, description="Execution configuration")
    postprocess: PostprocessConfig | None = Field(None, description="Postprocessing configuration")
    supervisor: dict[str, Any] | None = Field(None, description="Supervisor parameter overrides")


class SupervisorConfig(BaseModel):
    """Configuration for supervisor."""

    params: dict[str, Any] = Field(..., description="Supervisor parameters")


class ResolvedExperimentConfig(BaseModel):
    """Complete, resolved experiment configuration for evaluation experiments."""

    experiment: ExperimentMetadata = Field(..., description="Experiment metadata")
    components: ComponentsConfig = Field(..., description="Components configuration")
    simulator: SimulatorConfig = Field(..., description="Simulator configuration")
    supervisor: SupervisorConfig | None = Field(None, description="Supervisor configuration")
    execution: ExecutionConfig | None = Field(None, description="Execution configuration")
    evaluation: EvaluationConfig | None = Field(None, description="Evaluation configuration")
    postprocess: PostprocessConfig = Field(..., description="Postprocessing configuration")
    runtime: dict[str, Any] | None = Field(None, description="Runtime configuration")


# Alias for backward compatibility
