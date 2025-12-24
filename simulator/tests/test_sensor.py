import math
from unittest.mock import MagicMock

import pytest
from shapely.geometry import LinearRing, Polygon

from core.data import (
    LidarConfig,
    ObstacleShape,
    SimulatorObstacle,
    StaticObstaclePosition,
    VehicleState,
)
from simulator.sensor import LidarSensor


class TestLidarSensor:
    """Tests for LidarSensor."""

    @pytest.fixture
    def config(self) -> LidarConfig:
        return LidarConfig(
            num_beams=4,
            fov=360.0,
            range_min=0.0,
            range_max=10.0,
            angle_increment=math.pi / 2,
            x=0.0,
            y=0.0,
            z=0.0,
            yaw=0.0,
        )

    @pytest.fixture
    def vehicle_state(self) -> VehicleState:
        return VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, timestamp=0.0)

    def test_sensor_pose(self, config: LidarConfig, vehicle_state: VehicleState) -> None:
        """Test sensor pose calculation."""
        config.x = 1.0
        config.y = 0.0
        sensor = LidarSensor(config)

        # Vehicle at 0,0, yaw=0 -> Sensor at 1,0
        gx, gy = sensor._get_sensor_pose(vehicle_state)
        assert gx == pytest.approx(1.0)
        assert gy == pytest.approx(0.0)

        # Vehicle at 0,0, yaw=pi/2 -> Sensor at 0,1
        vehicle_state.yaw = math.pi / 2
        gx, gy = sensor._get_sensor_pose(vehicle_state)
        assert gx == pytest.approx(0.0)
        assert gy == pytest.approx(1.0)

    def test_no_obstacles_no_map(self, config: LidarConfig, vehicle_state: VehicleState) -> None:
        """Test scan with no obstacles and no map."""
        sensor = LidarSensor(config)
        scan = sensor.scan(vehicle_state)

        assert len(scan.ranges) == 4
        assert all(r == float("inf") for r in scan.ranges)

    def test_map_boundary_hit(self, config: LidarConfig, vehicle_state: VehicleState) -> None:
        """Test scan hitting map boundary."""
        # Mock map
        mock_map = MagicMock()
        # Create a square room from -5 to 5
        boundary = LinearRing([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        mock_map.drivable_area = Polygon(boundary)

        sensor = LidarSensor(config, map_instance=mock_map)
        scan = sensor.scan(vehicle_state)

        # Beams are at -180, -90, 0, 90 (approx, depending on implementation details)
        # Assuming star_angle = -pi, fov=2pi, num=4
        # Angles: -pi (-180), -pi/2 (-90), 0, pi/2 (90)
        # Ray 0 (Back): Hits (-5, 0) -> dist 5
        # Ray 1 (Right): Hits (0, -5) -> dist 5
        # Ray 2 (Front): Hits (5, 0) -> dist 5
        # Ray 3 (Left): Hits (0, 5) -> dist 5

        # Note: Implementation logic: start_angle + i * increment
        # start = -pi, inc = pi/2
        # i=0: -pi
        # i=1: -pi/2
        # i=2: 0
        # i=3: pi/2

        for r in scan.ranges:
            assert r == pytest.approx(5.0, abs=0.1)

    def test_obstacle_hit(self, config: LidarConfig, vehicle_state: VehicleState) -> None:
        """Test scan hitting an obstacle."""
        # Mock obstacle manager
        mock_manager = MagicMock()

        # Obstacle at x=5, y=0 (Front)
        obs = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="circle", radius=1.0),
            position=StaticObstaclePosition(x=5.0, y=0.0, yaw=0.0),
        )
        mock_manager.obstacles = [obs]

        # Need to patch the import inside LidarSensor or ensure simulator.obstacle imports work
        # Since we run this in environment where simulator is installed/available via pythonpath

        sensor = LidarSensor(config, obstacle_manager=mock_manager)
        scan = sensor.scan(vehicle_state)

        # Ray 2 (angle 0) points to positive X.
        # Sensor at 0,0. Obstacle center at 5,0. Radius 1.0.
        # Hit should be at 4.0.

        # Find the range corresponding to angle 0
        found_hit = False
        for r in scan.ranges:
            if r != float("inf"):
                assert r == pytest.approx(4.0, abs=0.1)
                found_hit = True

        assert found_hit

    def test_occlusion(self, config: LidarConfig, vehicle_state: VehicleState) -> None:
        """Test occlusion (closer object hides farther)."""
        mock_map = MagicMock()
        # Wall at x=8
        boundary = LinearRing([(8, -5), (8, 5), (10, 5), (10, -5)])
        mock_map.drivable_area = Polygon(boundary)

        mock_manager = MagicMock()
        # Obstacle at x=4
        obs = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="circle", radius=1.0),
            position=StaticObstaclePosition(x=4.0, y=0.0, yaw=0.0),
        )
        mock_manager.obstacles = [obs]

        sensor = LidarSensor(config, map_instance=mock_map, obstacle_manager=mock_manager)
        scan = sensor.scan(vehicle_state)

        # Ray towards x-axis should hit Obstacle (3.0m) not Wall (8.0m)
        found_hit = False
        for r in scan.ranges:
            if math.isinf(r):
                continue
            # If hit, it must be the obstacle (dist ~ 3.0)
            assert r == pytest.approx(3.0, abs=0.1)
            found_hit = True
        assert found_hit
