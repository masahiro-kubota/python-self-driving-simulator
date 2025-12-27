import numpy as np
import pytest
from core.data import VehicleParameters
from core.data.frame_data import FrameData
from core.data.ros import LaserScan
from core.interfaces.node import NodeExecutionResult
from tiny_lidar_net.config import TinyLidarNetConfig
from tiny_lidar_net.core import TinyLidarNetCore
from tiny_lidar_net.node import TinyLidarNetNode


class TestTinyLidarNetCore:
    """Tests for TinyLidarNetCore."""

    @pytest.fixture
    def core_without_weights(self) -> TinyLidarNetCore:
        """Create a TinyLidarNetCore instance without loading weights."""
        return TinyLidarNetCore(
            input_dim=1080,
            output_dim=2,
            architecture="large",
            ckpt_path="",
            max_range=30.0,
        )

    def test_initialization(self, core_without_weights: TinyLidarNetCore) -> None:
        """Test core initialization."""
        assert core_without_weights.input_dim == 1080
        assert core_without_weights.output_dim == 2
        assert core_without_weights.max_range == 30.0
        assert core_without_weights.model is not None

    def test_preprocess_ranges_nan_inf(self, core_without_weights: TinyLidarNetCore) -> None:
        """Test preprocessing handles NaN and Inf values."""
        # Create test data with NaN and Inf
        ranges = np.array([1.0, np.nan, 5.0, np.inf, 10.0, -np.inf], dtype=np.float32)

        processed = core_without_weights._preprocess_ranges(ranges)

        # Check that NaN and Inf are handled
        assert not np.isnan(processed).any()
        assert not np.isinf(processed).any()
        assert len(processed) == core_without_weights.input_dim

    def test_preprocess_ranges_clipping(self, core_without_weights: TinyLidarNetCore) -> None:
        """Test preprocessing clips values to max_range."""
        # Create test data with values exceeding max_range
        ranges = np.array([5.0, 40.0, 15.0, 50.0], dtype=np.float32)

        processed = core_without_weights._preprocess_ranges(ranges)

        # After normalization, max value should be 1.0 (30.0 / 30.0)
        assert np.all(processed <= 1.0)
        assert np.all(processed >= 0.0)

    def test_preprocess_ranges_resize_downsample(
        self, core_without_weights: TinyLidarNetCore
    ) -> None:
        """Test preprocessing resizes larger input."""
        # Create test data larger than input_dim
        ranges = np.ones(2000, dtype=np.float32) * 10.0

        processed = core_without_weights._preprocess_ranges(ranges)

        assert len(processed) == core_without_weights.input_dim

    def test_preprocess_ranges_resize_upsample(
        self, core_without_weights: TinyLidarNetCore
    ) -> None:
        """Test preprocessing resizes smaller input."""
        # Create test data smaller than input_dim
        ranges = np.ones(500, dtype=np.float32) * 10.0

        processed = core_without_weights._preprocess_ranges(ranges)

        assert len(processed) == core_without_weights.input_dim

    def test_preprocess_ranges_normalization(self, core_without_weights: TinyLidarNetCore) -> None:
        """Test preprocessing normalizes values."""
        # Create test data
        ranges = np.array([0.0, 15.0, 30.0], dtype=np.float32)

        processed = core_without_weights._preprocess_ranges(ranges)

        # Check normalization (values should be divided by max_range)
        # Note: resizing may affect exact values, but range should be [0, 1]
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)

    def test_process_output_shape(self, core_without_weights: TinyLidarNetCore) -> None:
        """Test process returns correct output shape."""
        # Create dummy LiDAR data
        ranges = np.ones(720, dtype=np.float32) * 10.0

        accel, steer = core_without_weights.process(ranges)

        # Check output types
        assert isinstance(accel, float)
        assert isinstance(steer, float)

        # Check output range (should be clipped to [-1, 1])
        assert -1.0 <= accel <= 1.0
        assert -1.0 <= steer <= 1.0

    def test_control_mode_fixed(self) -> None:
        """Test fixed control mode."""
        core = TinyLidarNetCore(
            input_dim=1080,
            output_dim=2,
            architecture="large",
            ckpt_path="",
            control_mode="fixed",
            acceleration=0.5,
        )

        ranges = np.ones(720, dtype=np.float32) * 10.0
        accel, steer = core.process(ranges)

        # In fixed mode, acceleration should be the fixed value
        assert accel == 0.5
        assert isinstance(steer, float)


class TestTinyLidarNetNode:
    """Tests for TinyLidarNetNode."""

    @pytest.fixture
    def config(self, tmp_path) -> TinyLidarNetConfig:
        """Create a test configuration."""
        # Create a dummy weight file
        weights_path = tmp_path / "test_weights.npy"
        dummy_weights = {}
        for layer in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
            dummy_weights[f"{layer}_weight"] = np.random.randn(1, 1, 1).astype(np.float32)
            dummy_weights[f"{layer}_bias"] = np.random.randn(1).astype(np.float32)
        for layer in ["fc1", "fc2", "fc3", "fc4"]:
            dummy_weights[f"{layer}_weight"] = np.random.randn(1, 1).astype(np.float32)
            dummy_weights[f"{layer}_bias"] = np.random.randn(1).astype(np.float32)
        np.save(weights_path, dummy_weights)

        # Load default vehicle parameters from config file
        from pathlib import Path

        import yaml

        vehicle_config_path = Path("experiment/configs/vehicles/default_vehicle.yaml")
        with open(vehicle_config_path) as f:
            vehicle_config = yaml.safe_load(f)

        return TinyLidarNetConfig(
            model_path=weights_path,
            input_dim=1080,
            output_dim=2,
            architecture="large",
            max_range=30.0,
            vehicle_params=VehicleParameters(**vehicle_config),
        )

    @pytest.fixture
    def node(self, config: TinyLidarNetConfig) -> TinyLidarNetNode:
        """Create a TinyLidarNetNode instance."""
        return TinyLidarNetNode(config=config, rate_hz=30.0)

    def test_node_initialization(self, node: TinyLidarNetNode) -> None:
        """Test node initialization."""
        assert node.name == "TinyLidarNet"
        assert node.rate_hz == 30.0
        assert node.core is not None

    def test_node_io(self, node: TinyLidarNetNode) -> None:
        """Test node IO specification."""
        node_io = node.get_node_io()

        assert "control_cmd" in node_io.outputs
        from core.data.ros import AckermannDriveStamped

        assert node_io.outputs["control_cmd"] == AckermannDriveStamped

    def test_on_run_success(self, node: TinyLidarNetNode) -> None:
        """Test successful execution."""
        # Create frame data with LaserScan
        lidar_scan = LaserScan(
            range_max=30.0,
            ranges=[10.0] * 720,
        )

        frame_data = FrameData()
        frame_data.lidar_scan = lidar_scan
        node.frame_data = frame_data

        result = node.on_run(0.0)

        assert result == NodeExecutionResult.SUCCESS
        assert hasattr(frame_data, "control_cmd")
        from core.data.ros import AckermannDriveStamped

        assert isinstance(frame_data.control_cmd, AckermannDriveStamped)
        assert -1.0 <= frame_data.control_cmd.drive.acceleration <= 1.0
        assert -1.0 <= frame_data.control_cmd.drive.steering_angle <= 1.0

    def test_on_run_no_lidar_scan(self, node: TinyLidarNetNode) -> None:
        """Test execution when LiDAR scan is missing."""
        frame_data = FrameData()
        node.frame_data = frame_data

        result = node.on_run(0.0)

        assert result == NodeExecutionResult.SKIPPED

    def test_on_run_no_frame_data(self, node: TinyLidarNetNode) -> None:
        """Test execution when frame_data is None."""
        node.frame_data = None

        result = node.on_run(0.0)

        assert result == NodeExecutionResult.FAILED
