"""Tests for JsonSimulationLogRepository."""

import tempfile
from pathlib import Path

import pytest
from core.data import SimulationLog, SimulationStep, VehicleState
from core.data.autoware import AckermannControlCommand, AckermannLateralCommand, LongitudinalCommand

from simulator import JsonSimulationLogRepository


class TestJsonSimulationLogRepository:
    """Tests for JsonSimulationLogRepository."""

    @pytest.fixture
    def sample_log(self) -> SimulationLog:
        """Create a sample simulation log."""
        steps = [
            SimulationStep(
                timestamp=0.0,
                vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0),
                action=AckermannControlCommand(
                    lateral=AckermannLateralCommand(steering_tire_angle=0.0),
                    longitudinal=LongitudinalCommand(acceleration=0.0),
                ),
            ),
            SimulationStep(
                timestamp=0.1,
                vehicle_state=VehicleState(x=1.0, y=0.0, yaw=0.0, velocity=1.0),
                action=AckermannControlCommand(
                    lateral=AckermannLateralCommand(steering_tire_angle=0.1),
                    longitudinal=LongitudinalCommand(acceleration=0.5),
                ),
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
            assert (
                loaded_log.steps[0].action.lateral.steering_tire_angle
                == sample_log.steps[0].action.lateral.steering_tire_angle
            )

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
                assert (
                    original.action.lateral.steering_tire_angle
                    == loaded.action.lateral.steering_tire_angle
                )
                assert (
                    original.action.longitudinal.acceleration
                    == loaded.action.longitudinal.acceleration
                )

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file raises error."""
        repository = JsonSimulationLogRepository()

        with pytest.raises(FileNotFoundError):
            repository.load(Path("/nonexistent/file.json"))


class TestJsonSimulationLogRepositoryErrorHandling:
    """Tests for error handling in JsonSimulationLogRepository."""

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Test loading invalid JSON file."""
        repository = JsonSimulationLogRepository()

        # Create invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("This is not valid JSON")

        with pytest.raises(Exception):  # Will raise JSON decode error
            repository.load(invalid_file)

    def test_save_to_readonly_directory(self) -> None:
        """Test saving to read-only directory."""

        repository = JsonSimulationLogRepository()
        sample_log = SimulationLog(steps=[], metadata={})

        # Try to save to a path that doesn't exist and can't be created
        # This will raise a permission or file not found error
        with pytest.raises(Exception):
            repository.save(sample_log, Path("/nonexistent_root/test.json"))

    def test_empty_log(self) -> None:
        """Test saving and loading empty log."""
        repository = JsonSimulationLogRepository()
        empty_log = SimulationLog(steps=[], metadata={})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "empty.json"

            # Save and load empty log
            repository.save(empty_log, file_path)
            loaded_log = repository.load(file_path)

            assert len(loaded_log.steps) == 0
            assert loaded_log.metadata == {}


class TestJsonSimulationLogRepositoryDataIntegrity:
    """Tests for data integrity in JsonSimulationLogRepository."""

    def test_large_log(self) -> None:
        """Test saving and loading large log."""
        repository = JsonSimulationLogRepository()

        # Create log with many steps
        steps = [
            SimulationStep(
                timestamp=float(i) * 0.1,
                vehicle_state=VehicleState(x=float(i), y=float(i), yaw=0.0, velocity=5.0),
                action=AckermannControlCommand(
                    lateral=AckermannLateralCommand(steering_tire_angle=0.0),
                    longitudinal=LongitudinalCommand(acceleration=0.0),
                ),
            )
            for i in range(1000)
        ]
        large_log = SimulationLog(steps=steps, metadata={"test": "large"})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "large.json"

            repository.save(large_log, file_path)
            loaded_log = repository.load(file_path)

            assert len(loaded_log.steps) == 1000
            assert loaded_log.metadata["test"] == "large"

    def test_special_characters_in_metadata(self) -> None:
        """Test metadata with special characters."""
        repository = JsonSimulationLogRepository()

        # Metadata with special characters
        metadata = {
            "track": "test_track",
            "special": "日本語テスト",
            "symbols": "!@#$%^&*()",
            "quotes": 'test "quoted" value',
        }
        log = SimulationLog(steps=[], metadata=metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "special.json"

            repository.save(log, file_path)
            loaded_log = repository.load(file_path)

            assert loaded_log.metadata == metadata

    def test_none_values_in_state(self) -> None:
        """Test handling of None values in vehicle state."""
        repository = JsonSimulationLogRepository()

        # Create state with None optional fields
        steps = [
            SimulationStep(
                timestamp=0.0,
                vehicle_state=VehicleState(
                    x=0.0, y=0.0, yaw=0.0, velocity=0.0, acceleration=None, steering=None
                ),
                action=AckermannControlCommand(
                    lateral=AckermannLateralCommand(steering_tire_angle=0.0),
                    longitudinal=LongitudinalCommand(acceleration=0.0),
                ),
            )
        ]
        log = SimulationLog(steps=steps, metadata={})

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "none_values.json"

            repository.save(log, file_path)
            loaded_log = repository.load(file_path)

            assert len(loaded_log.steps) == 1
            # None values should be preserved or handled correctly
            assert loaded_log.steps[0].vehicle_state.x == 0.0
