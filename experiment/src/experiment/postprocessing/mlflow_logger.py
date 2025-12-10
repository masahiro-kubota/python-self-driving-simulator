"""MLflow integration implementation."""

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import mlflow
from core.data.experiment import Artifact, ExperimentResult


class MLflowExperimentLogger:
    """MLflow implementation for experiment logging."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize MLflow logger.

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._is_ci = bool(os.getenv("CI"))

        if not self._is_ci:
            # Set up MinIO credentials (hardcoded for now as in original runner)
            os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

    def start_run(self) -> AbstractContextManager:
        """MLflowランを開始する（コンテキストマネージャ）.

        Returns:
            MLflow run context manager
        """
        if self._is_ci:
            return nullcontext()
        return mlflow.start_run()  # type: ignore

    def log_params(self, params: dict[str, Any]) -> bool:
        """Log parameters to MLflow.

        Returns:
            bool: True if logging was successful
        """
        if not self._is_ci:
            mlflow.log_params(params)
        return True

    def log_metrics(self, metrics: dict[str, float]) -> bool:
        """Log metrics to MLflow.

        Returns:
            bool: True if logging was successful
        """
        if not self._is_ci:
            mlflow.log_metrics(metrics)
        return True

    def log_artifact(self, artifact: Artifact) -> bool:
        """Log artifact to MLflow.

        Returns:
            bool: True if logging was successful, False if artifact not found
        """
        if not self._is_ci:
            if not os.path.exists(artifact.local_path):
                print(f"Warning: Artifact not found: {artifact.local_path}")
                return False

            mlflow.log_artifact(str(artifact.local_path), artifact_path=artifact.remote_path)
        return True

    def log_result(self, result: ExperimentResult) -> bool:
        """Log entire experiment result to MLflow.

        Returns:
            bool: ログ記録が成功した場合True
        """
        if self._is_ci:
            return True

        # Start run if not active, or assume active context?
        # Typically log_result is called at the END of an experiment.
        # But mlflow.start_run context might have closed if we are not careful.
        # The runner handles the context scope, so we assume we are inside a run OR we start one.
        # However, MLflow is stateful.
        # For safety, we check if there is an active run. If not, we rely on the caller to manage context
        # OR we just log (which will fail if no run is active, or start a new one depending on config).
        # Given the previous design used a context manager, we should likely check `mlflow.active_run()`.

        if mlflow.active_run() is None:
            print("Warning: No active MLflow run. Starting a new one for logging result.")
            with mlflow.start_run():
                self._log_content(result)
        else:
            self._log_content(result)

        return True

    def _log_content(self, result: ExperimentResult) -> None:
        """Internal helper to log content."""
        # Log config if available
        if result.config:
            config_dict = self._config_to_dict(result.config)
            # Flatten config for MLflow params (MLflow doesn't support nested dicts well)
            flattened_config = self._flatten_dict(config_dict, prefix="config")
            mlflow.log_params(flattened_config)

        if result.params:
            mlflow.log_params(result.params)

        if result.metrics:
            mlflow.log_metrics(result.metrics.to_dict())

        for artifact in result.artifacts:
            self.log_artifact(artifact)

        # Log extra metadata if needed
        if result.mlflow_run_id:
            mlflow.set_tag("original_run_id", result.mlflow_run_id)

    def _flatten_dict(
        self, d: dict, prefix: str = "", sep: str = "."
    ) -> dict[str, str | int | float | bool]:
        """Flatten nested dictionary for MLflow params."""
        items: list[tuple[str, str | int | float | bool]] = []
        for k, v in d.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list | tuple):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            elif v is None:
                items.append((new_key, "None"))
            else:
                items.append((new_key, v))
        return dict(items)

    def _config_to_dict(self, config: Any) -> dict:
        """Convert ExperimentConfig to dict, handling Pydantic models and Enum types."""
        try:
            # Try Pydantic model first
            if hasattr(config, "model_dump"):
                return config.model_dump(mode="python")
            elif hasattr(config, "dict"):
                return config.dict()
        except Exception:
            pass

        # Fallback to dataclass handling
        from dataclasses import fields, is_dataclass
        from enum import Enum

        def convert_value(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            elif is_dataclass(obj):
                return {f.name: convert_value(getattr(obj, f.name)) for f in fields(obj)}
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, list | tuple):
                return [convert_value(item) for item in obj]
            else:
                return obj

        return convert_value(config)
