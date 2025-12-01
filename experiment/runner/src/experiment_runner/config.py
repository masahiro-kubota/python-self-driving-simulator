"""Configuration models for experiment runner."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ExperimentType(str, Enum):
    """Experiment type enumeration."""

    DATA_COLLECTION = "data_collection"
    TRAINING = "training"
    EVALUATION = "evaluation"


class ComponentConfig(BaseModel):
    """Configuration for a component."""

    type: str = Field(..., description="Component class name")
    params: dict[str, Any] = Field(default_factory=dict, description="Component parameters")


class SimulatorConfig(BaseModel):
    """Configuration for simulator."""

    type: str = Field(..., description="Simulator class name")
    params: dict[str, Any] = Field(default_factory=dict, description="Simulator parameters")


class ExecutionConfig(BaseModel):
    """Configuration for execution."""

    num_episodes: int = Field(1, description="Number of episodes to run")
    max_steps_per_episode: int = Field(2000, description="Maximum steps per episode")
    parallel: bool = Field(False, description="Run in parallel")
    num_workers: int = Field(1, description="Number of parallel workers")


class TrainingConfig(BaseModel):
    """Configuration for training."""

    data_dir: str = Field(..., description="Directory containing training data")
    reference_trajectory_path: str | None = Field(None, description="Path to reference trajectory")
    epochs: int = Field(100, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size")
    learning_rate: float = Field(0.001, description="Learning rate")
    validation_split: float = Field(0.2, description="Validation split ratio")
    optimizer: str = Field("adam", description="Optimizer type")
    loss_function: str = Field("mse", description="Loss function type")


class DataCollectionConfig(BaseModel):
    """Configuration for data collection."""

    output_dir: str = Field(..., description="Output directory for collected data")
    format: Literal["json", "mcap"] = Field("json", description="Data format")
    save_frequency: int = Field(1, description="Save every N episodes")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    metrics: list[str] = Field(
        default_factory=lambda: ["lap_time", "lateral_error", "comfort"],
        description="Metrics to calculate",
    )
    generate_dashboard: bool = Field(True, description="Generate dashboard")


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    type: str = Field(..., description="Model type")
    architecture: dict[str, Any] = Field(
        default_factory=dict, description="Model architecture params"
    )


class MLflowConfig(BaseModel):
    """Configuration for MLflow logging."""

    enabled: bool = Field(True, description="Enable MLflow logging")
    tracking_uri: str = Field("http://localhost:5000", description="MLflow tracking URI")


class MCAPConfig(BaseModel):
    """Configuration for MCAP logging."""

    enabled: bool = Field(True, description="Enable MCAP logging")
    output_dir: str = Field("/tmp", description="Output directory for MCAP files")


class DashboardConfig(BaseModel):
    """Configuration for dashboard generation."""

    enabled: bool = Field(True, description="Enable dashboard generation")


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    mcap: MCAPConfig = Field(default_factory=MCAPConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


class ExperimentMetadata(BaseModel):
    """Experiment metadata."""

    name: str = Field(..., description="Experiment name")
    type: ExperimentType = Field(..., description="Experiment type")
    description: str = Field("", description="Experiment description")


class ComponentsConfig(BaseModel):
    """Configuration for all components."""

    perception: ComponentConfig | None = Field(None, description="Perception component config")
    planning: ComponentConfig = Field(..., description="Planning component config")
    control: ComponentConfig = Field(..., description="Control component config")


class ModelCheckpointConfig(BaseModel):
    """Configuration for model checkpointing."""

    save_best: bool = Field(True, description="Save best model")
    save_every_n_epochs: int = Field(10, description="Save checkpoint every N epochs")
    output_dir: str = Field("data/models", description="Output directory for models")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    experiment: ExperimentMetadata = Field(..., description="Experiment metadata")
    components: ComponentsConfig | None = Field(None, description="Components configuration")
    simulator: SimulatorConfig | None = Field(None, description="Simulator configuration")
    execution: ExecutionConfig | None = Field(None, description="Execution configuration")
    model: ModelConfig | None = Field(None, description="Model configuration")
    training: TrainingConfig | None = Field(None, description="Training configuration")
    data_collection: DataCollectionConfig | None = Field(
        None, description="Data collection configuration"
    )
    evaluation: EvaluationConfig | None = Field(None, description="Evaluation configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_experiment_type(self) -> "ExperimentConfig":
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            ExperimentConfig instance
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
