"""Integration tests for training workflow."""

import tempfile
from pathlib import Path

import pytest
from experiment_runner.preprocessing.schemas import ExperimentType, ResolvedExperimentConfig

from core.data import Action, SimulationLog, SimulationStep, VehicleState


class TestTrainingIntegration:
    """Integration tests for training."""

    @pytest.fixture
    def dummy_log_data(self) -> SimulationLog:
        """Create dummy simulation log data."""
        steps = []
        for i in range(10):
            step = SimulationStep(
                timestamp=float(i) * 0.1,
                vehicle_state=VehicleState(x=float(i), y=0.0, yaw=0.0, velocity=5.0),
                action=Action(steering=0.1, acceleration=0.5),
            )
            steps.append(step)

        return SimulationLog(steps=steps, metadata={"track": "test_track"})

    def test_training_workflow(self, dummy_log_data: SimulationLog) -> None:
        """Test training configuration validation and workflow setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Setup dummy data (simulating S3 content)
            # data_dir = S3 bucket content
            data_dir = tmp_path / "s3_data"
            data_dir.mkdir()
            log_path = data_dir / "log_0.json"

            from simulator.io import JsonSimulationLogRepository

            repository = JsonSimulationLogRepository()
            repository.save(dummy_log_data, log_path)

            # 2. Setup config use S3 dataset params
            config_dict = {
                "experiment": {
                    "name": "test_training",
                    "type": "training",
                    "description": "Test training integration",
                },
                "model": {
                    "type": "MLP",
                    "architecture": {"hidden_sizes": [64, 64]},
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "dataset_project": "test_project",
                    "dataset_scenario": "test_scenario",
                    "dataset_version": "v1.0.0",
                },
                "logging": {
                    "mlflow": {"enabled": False},
                },
            }

            # Verify config validation passes
            config = ResolvedExperimentConfig(**config_dict)

            # Verify config structure
            assert config.experiment.type == ExperimentType.TRAINING
            assert config.model is not None
            assert config.model.type == "MLP"
            assert config.training is not None
            assert config.training.dataset_project == "test_project"
