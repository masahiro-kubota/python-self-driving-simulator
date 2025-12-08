"""Tests for JsonSimulationLogRepository."""

import tempfile
from pathlib import Path

import pytest

from core.data import Action, SimulationLog, SimulationStep, VehicleState
from simulators.core import JsonSimulationLogRepository


class TestJsonSimulationLogRepository:
    """Tests for JsonSimulationLogRepository."""

    @pytest.fixture
    def sample_log(self) -> SimulationLog:
        """Create a sample simulation log."""
        steps = [
            SimulationStep(
                timestamp=0.0,
                vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0),
                action=Action(steering=0.0, acceleration=0.0),
            ),
            SimulationStep(
                timestamp=0.1,
                vehicle_state=VehicleState(x=1.0, y=0.0, yaw=0.0, velocity=1.0),
                action=Action(steering=0.1, acceleration=0.5),
            ),
        ]
        return SimulationLog(steps=steps, metadata={"track": "test_track", "version": "1.0"})

    def test_save(self, sample_log: SimulationLog) -> None:
        """Test saving simulation log to JSON file."""
        repository = JsonSimulationLogRepository()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_log.json"
            repository.save(sample_log, file_path)

            # Verify file exists
            assert file_path.exists()

            # Verify file is valid JSON
            import json

            with open(file_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "steps" in data
            assert len(data["steps"]) == 2

    def test_load(self, sample_log: SimulationLog) -> None:
        """Test loading simulation log from JSON file."""
        repository = JsonSimulationLogRepository()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_log.json"

            # Save first
            repository.save(sample_log, file_path)

            # Load
            loaded_log = repository.load(file_path)

            # Verify
            assert len(loaded_log.steps) == len(sample_log.steps)
            assert loaded_log.metadata == sample_log.metadata

            # Verify first step
            assert loaded_log.steps[0].timestamp == sample_log.steps[0].timestamp
            assert loaded_log.steps[0].vehicle_state.x == sample_log.steps[0].vehicle_state.x
            assert loaded_log.steps[0].action.steering == sample_log.steps[0].action.steering

    def test_round_trip(self, sample_log: SimulationLog) -> None:
        """Test save and load round trip."""
        repository = JsonSimulationLogRepository()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_log.json"

            # Save and load
            repository.save(sample_log, file_path)
            loaded_log = repository.load(file_path)

            # Verify all steps
            assert len(loaded_log.steps) == len(sample_log.steps)
            for original, loaded in zip(sample_log.steps, loaded_log.steps):
                assert original.timestamp == loaded.timestamp
                assert original.vehicle_state.x == loaded.vehicle_state.x
                assert original.vehicle_state.y == loaded.vehicle_state.y
                assert original.action.steering == loaded.action.steering
                assert original.action.acceleration == loaded.action.acceleration

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file raises error."""
        repository = JsonSimulationLogRepository()

        with pytest.raises(FileNotFoundError):
            repository.load(Path("/nonexistent/file.json"))
