"""Configuration models for experiment runner."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from core.interfaces.node import StrictConfig


class ExperimentType(str, Enum):
    """Experiment type enumeration."""

    DATA_COLLECTION = "data_collection"
    TRAINING = "training"
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


class TrainingConfig(BaseModel):
    """Configuration for training."""

    # S3 dataset configuration
    dataset_project: str | None = Field(None, description="Dataset project name")
    dataset_scenario: str | None = Field(None, description="Dataset scenario/task name")
    dataset_version: str | None = Field(None, description="Dataset version")
    dataset_stage: Literal["raw", "processed", "features"] = Field(
        "raw", description="Data processing stage"
    )
    dataset_path: str | None = Field(None, description="Direct S3 dataset path")

    reference_trajectory_path: str | None = Field(None, description="Path to reference trajectory")
    epochs: int = Field(..., description="Number of training epochs")
    batch_size: int = Field(..., description="Batch size")
    learning_rate: float = Field(..., description="Learning rate")
    validation_split: float = Field(..., description="Validation split ratio")
    optimizer: str = Field(..., description="Optimizer type")
    loss_function: str = Field(..., description="Loss function type")

    @model_validator(mode="after")
    def validate_data_source(self) -> "TrainingConfig":
        """Validate data source configuration."""
        has_s3_components = bool(
            self.dataset_project and self.dataset_scenario and self.dataset_version
        )
        has_s3_path = bool(self.dataset_path)

        if not (has_s3_components or has_s3_path):
            raise ValueError(
                "Must specify either (dataset_project + dataset_scenario + dataset_version) "
                "or dataset_path"
            )

        return self


class DataCollectionConfig(BaseModel):
    """Configuration for data collection."""

    storage_backend: Literal["s3"] = Field(..., description="Storage backend")

    # S3 dataset configuration
    project: str | None = Field(None, description="Project name")
    scenario: str | None = Field(None, description="Scenario/task name")
    version: str | None = Field(None, description="Dataset version")
    stage: Literal["raw", "processed", "features"] = Field(..., description="Data stage")

    format: Literal["json", "mcap"] = Field(..., description="Data format")
    save_frequency: int = Field(..., description="Save every N episodes")

    @model_validator(mode="after")
    def validate_storage_config(self) -> "DataCollectionConfig":
        """Validate storage configuration."""
        if self.storage_backend == "s3" and not (self.project and self.scenario and self.version):
            raise ValueError("project, scenario, and version are required for S3 storage")
        return self


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    metrics: list[str] = Field(
        ...,
        description="Metrics to calculate",
    )
    generate_dashboard: bool = Field(..., description="Generate dashboard")


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    type: str = Field(..., description="Model type")
    architecture: dict[str, Any] = Field(
        default_factory=dict, description="Model architecture params"
    )


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
    runtime: dict[str, Any] = Field(..., description="Runtime configuration")


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
    """Complete, resolved experiment configuration (formerly ExperimentConfig)."""

    experiment: ExperimentMetadata = Field(..., description="Experiment metadata")
    components: ComponentsConfig | None = Field(None, description="Components configuration")
    simulator: SimulatorConfig | None = Field(None, description="Simulator configuration")
    supervisor: SupervisorConfig | None = Field(
        None, description="Supervisor configuration"
    )  # Added
    execution: ExecutionConfig | None = Field(None, description="Execution configuration")
    model: ModelConfig | None = Field(None, description="Model configuration")
    training: TrainingConfig | None = Field(None, description="Training configuration")
    data_collection: DataCollectionConfig | None = Field(
        None, description="Data collection configuration"
    )
    evaluation: EvaluationConfig | None = Field(None, description="Evaluation configuration")
    postprocess: PostprocessConfig = Field(...)
    runtime: dict[str, Any] = Field(..., description="Runtime configuration")

    @model_validator(mode="after")
    def validate_experiment_type(self) -> "ResolvedExperimentConfig":
        """Validate required fields based on experiment type."""
        exp_type = self.experiment.type

        if exp_type == ExperimentType.DATA_COLLECTION:
            if not self.components:
                raise ValueError("components is required for data_collection experiments")
            if not self.simulator:
                raise ValueError("simulator is required for data_collection experiments")
            if not self.data_collection:
                raise ValueError("data_collection is required for data_collection experiments")

        elif exp_type == ExperimentType.TRAINING:
            if not self.model:
                raise ValueError("model is required for training experiments")
            if not self.training:
                raise ValueError("training is required for training experiments")

        elif exp_type == ExperimentType.EVALUATION:
            if not self.components:
                raise ValueError("components is required for evaluation experiments")
            if not self.simulator:
                raise ValueError("simulator is required for evaluation experiments")

        return self


# Alias for backward compatibility
