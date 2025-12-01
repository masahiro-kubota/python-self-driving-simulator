"""Integration tests for training workflow."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from core.data import Action, SimulationLog, SimulationStep, VehicleState

from experiment_runner.config import ExperimentConfig
from experiment_runner.runner import ExperimentRunner


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
        """Test complete training workflow via ExperimentRunner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Setup dummy data
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            log_path = data_dir / "log_0.json"
            dummy_log_data.save(log_path)

            # 2. Setup config
            config_dict = {
                "experiment": {"name": "test_training", "description": "Test training integration"},
                "components": {
                    "planning": {
                        "type": "PurePursuitPlanner",
                        "params": {"lookahead_distance": 5.0},
                    },
                    "control": {
                        "type": "NeuralController",
                        "params": {
                            "model_path": str(tmp_path / "model.pth"),
                            "scaler_path": str(tmp_path / "scaler.json"),
                        },
                    },
                },
                "simulator": {"type": "KinematicSimulator", "params": {}},
                "execution": {"mode": "training", "max_steps": 10},
                "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001},
                "logging": {
                    "mcap": {
                        "enabled": True,
                        "output_dir": str(data_dir),  # Use data dir as input source for training
                    },
                    "mlflow": {"enabled": False},
                    "dashboard": {"enabled": False},
                },
            }

            config = ExperimentConfig(**config_dict)

            # 3. Initialize Runner
            runner = ExperimentRunner(config)

            # Mock planner's reference trajectory since we don't load a real track
            # The runner needs a planner with reference trajectory to calculate errors
            mock_planner = MagicMock()
            # Create a dummy reference trajectory (straight line)
            from core.data import Trajectory, TrajectoryPoint

            ref_traj = Trajectory(
                [TrajectoryPoint(x=float(i), y=0.0, yaw=0.0, velocity=5.0) for i in range(20)]
            )
            mock_planner.reference_trajectory = ref_traj
            runner.planner = mock_planner

            # 4. Run training
            # We need to mock MLflow to avoid connection errors in test environment if not local
            with patch("mlflow.start_run") as mock_run:
                mock_run.return_value.__enter__.return_value = None

                # Also patch Trainer's save method to avoid actual file writing if needed,
                # but here we want to verify file creation, so we let it run.
                # However, Trainer saves to "outputs/" by default in current dir.
                # We should probably make output dir configurable in Trainer.
                # For this test, we'll check if it runs without error.

                # We need to patch the Trainer import or the class itself to check calls,
                # but integration test should run real code.
                # The issue is Trainer saves to "outputs/final_model.pth" relative to CWD.
                # Let's change CWD to tmpdir for the test duration.
                import os

                cwd = Path.cwd()
                os.chdir(tmpdir)
                try:
                    runner._run_training()
                finally:
                    os.chdir(cwd)

            # 5. Verify artifacts
            # Check if model file was created in tmpdir/outputs
            output_model = tmp_path / "outputs" / "final_model.pth"
            assert output_model.exists(), "Model file was not created"
