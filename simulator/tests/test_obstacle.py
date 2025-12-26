"""Tests for obstacle management and collision detection."""

import pytest
from core.data import (
    ObstacleShape,
    ObstacleState,
    ObstacleTrajectory,
    SimulatorObstacle,
    StaticObstaclePosition,
    TrajectoryWaypoint,
)
from shapely.geometry import Point, Polygon
from simulator.obstacle import (
    ObstacleManager,
    check_collision,
    get_obstacle_polygon,
    get_obstacle_state,
)


class TestStaticObstacle:
    """Tests for static obstacle."""

    def test_static_obstacle_state(self) -> None:
        """Test static obstacle state calculation."""
        obstacle = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="rectangle", width=2.0, length=4.0),
            position=StaticObstaclePosition(x=10.0, y=5.0, yaw=0.5),
        )

        state = get_obstacle_state(obstacle, time=0.0)

        assert state.x == 10.0
        assert state.y == 5.0
        assert state.yaw == 0.5
        assert state.timestamp == 0.0

    def test_static_obstacle_state_at_different_time(self) -> None:
        """Test that static obstacle state doesn't change with time."""
        obstacle = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="rectangle", width=2.0, length=4.0),
            position=StaticObstaclePosition(x=10.0, y=5.0, yaw=0.5),
        )

        state1 = get_obstacle_state(obstacle, time=0.0)
        state2 = get_obstacle_state(obstacle, time=10.0)

        assert state1.x == state2.x
        assert state1.y == state2.y
        assert state1.yaw == state2.yaw


class TestDynamicObstacle:
    """Tests for dynamic obstacle."""

    def test_dynamic_obstacle_linear_interpolation(self) -> None:
        """Test dynamic obstacle with linear interpolation."""
        obstacle = SimulatorObstacle(
            type="dynamic",
            shape=ObstacleShape(type="circle", radius=1.0),
            trajectory=ObstacleTrajectory(
                type="waypoint",
                interpolation="linear",
                waypoints=[
                    TrajectoryWaypoint(time=0.0, x=0.0, y=0.0, yaw=0.0),
                    TrajectoryWaypoint(time=10.0, x=10.0, y=5.0, yaw=1.0),
                ],
            ),
        )

        # At start
        state0 = get_obstacle_state(obstacle, time=0.0)
        assert state0.x == pytest.approx(0.0)
        assert state0.y == pytest.approx(0.0)
        assert state0.yaw == pytest.approx(0.0)

        # At middle
        state5 = get_obstacle_state(obstacle, time=5.0)
        assert state5.x == pytest.approx(5.0)
        assert state5.y == pytest.approx(2.5)
        assert state5.yaw == pytest.approx(0.5)

        # At end
        state10 = get_obstacle_state(obstacle, time=10.0)
        assert state10.x == pytest.approx(10.0)
        assert state10.y == pytest.approx(5.0)
        assert state10.yaw == pytest.approx(1.0)

    def test_dynamic_obstacle_before_start(self) -> None:
        """Test dynamic obstacle before first waypoint."""
        obstacle = SimulatorObstacle(
            type="dynamic",
            shape=ObstacleShape(type="circle", radius=1.0),
            trajectory=ObstacleTrajectory(
                type="waypoint",
                interpolation="linear",
                waypoints=[
                    TrajectoryWaypoint(time=5.0, x=10.0, y=5.0, yaw=0.0),
                    TrajectoryWaypoint(time=10.0, x=20.0, y=10.0, yaw=1.0),
                ],
            ),
        )

        state = get_obstacle_state(obstacle, time=0.0)
        # Should stay at first waypoint
        assert state.x == pytest.approx(10.0)
        assert state.y == pytest.approx(5.0)

    def test_dynamic_obstacle_after_end(self) -> None:
        """Test dynamic obstacle after last waypoint."""
        obstacle = SimulatorObstacle(
            type="dynamic",
            shape=ObstacleShape(type="circle", radius=1.0),
            trajectory=ObstacleTrajectory(
                type="waypoint",
                interpolation="linear",
                waypoints=[
                    TrajectoryWaypoint(time=0.0, x=0.0, y=0.0, yaw=0.0),
                    TrajectoryWaypoint(time=10.0, x=10.0, y=5.0, yaw=1.0),
                ],
            ),
        )

        state = get_obstacle_state(obstacle, time=20.0)
        # Should stay at last waypoint
        assert state.x == pytest.approx(10.0)
        assert state.y == pytest.approx(5.0)

    def test_dynamic_obstacle_loop(self) -> None:
        """Test dynamic obstacle with looping trajectory."""
        obstacle = SimulatorObstacle(
            type="dynamic",
            shape=ObstacleShape(type="circle", radius=1.0),
            trajectory=ObstacleTrajectory(
                type="waypoint",
                interpolation="linear",
                waypoints=[
                    TrajectoryWaypoint(time=0.0, x=0.0, y=0.0, yaw=0.0),
                    TrajectoryWaypoint(time=10.0, x=10.0, y=5.0, yaw=1.0),
                ],
                loop=True,
            ),
        )

        # First loop
        state5 = get_obstacle_state(obstacle, time=5.0)
        assert state5.x == pytest.approx(5.0)

        # Second loop (time=15 should be same as time=5)
        state15 = get_obstacle_state(obstacle, time=15.0)
        assert state15.x == pytest.approx(5.0)


class TestObstaclePolygon:
    """Tests for obstacle polygon generation."""

    def test_rectangle_polygon(self) -> None:
        """Test rectangle obstacle polygon."""
        obstacle = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="rectangle", width=2.0, length=4.0),
            position=StaticObstaclePosition(x=0.0, y=0.0, yaw=0.0),
        )

        state = ObstacleState(x=0.0, y=0.0, yaw=0.0, timestamp=0.0)
        polygon = get_obstacle_polygon(obstacle, state)

        assert isinstance(polygon, Polygon)
        # Check bounds
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        assert bounds[0] == pytest.approx(-2.0)  # minx
        assert bounds[1] == pytest.approx(-1.0)  # miny
        assert bounds[2] == pytest.approx(2.0)  # maxx
        assert bounds[3] == pytest.approx(1.0)  # maxy

    def test_circle_polygon(self) -> None:
        """Test circle obstacle polygon."""
        obstacle = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="circle", radius=1.0),
            position=StaticObstaclePosition(x=0.0, y=0.0, yaw=0.0),
        )

        state = ObstacleState(x=0.0, y=0.0, yaw=0.0, timestamp=0.0)
        polygon = get_obstacle_polygon(obstacle, state)

        assert isinstance(polygon, Polygon)
        # Check that it's approximately a circle
        center = Point(0.0, 0.0)
        assert polygon.contains(center)
        # Check radius (approximately)
        bounds = polygon.bounds
        assert abs(bounds[2] - bounds[0]) == pytest.approx(2.0, abs=0.1)  # diameter


class TestCollisionDetection:
    """Tests for collision detection."""

    def test_no_collision(self) -> None:
        """Test no collision case."""
        vehicle_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        obstacle_poly = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])

        assert check_collision(vehicle_poly, obstacle_poly) is False

    def test_collision_overlap(self) -> None:
        """Test collision with overlap."""
        vehicle_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        obstacle_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

        assert check_collision(vehicle_poly, obstacle_poly) is True

    def test_collision_touch(self) -> None:
        """Test collision when polygons touch."""
        vehicle_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        obstacle_poly = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

        # Touching edges should be considered collision
        assert check_collision(vehicle_poly, obstacle_poly) is True


class TestObstacleManager:
    """Tests for obstacle manager."""

    def test_no_obstacles(self) -> None:
        """Test with no obstacles."""
        manager = ObstacleManager([])
        vehicle_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        assert manager.check_vehicle_collision(vehicle_poly, 0.0) is False

    def test_single_obstacle_no_collision(self) -> None:
        """Test with single obstacle, no collision."""
        obstacle = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="rectangle", width=2.0, length=4.0),
            position=StaticObstaclePosition(x=10.0, y=10.0, yaw=0.0),
        )
        manager = ObstacleManager([obstacle])
        vehicle_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        assert manager.check_vehicle_collision(vehicle_poly, 0.0) is False

    def test_single_obstacle_collision(self) -> None:
        """Test with single obstacle, collision detected."""
        obstacle = SimulatorObstacle(
            type="static",
            shape=ObstacleShape(type="rectangle", width=2.0, length=2.0),
            position=StaticObstaclePosition(x=0.5, y=0.5, yaw=0.0),
        )
        manager = ObstacleManager([obstacle])
        vehicle_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        assert manager.check_vehicle_collision(vehicle_poly, 0.0) is True

    def test_multiple_obstacles(self) -> None:
        """Test with multiple obstacles."""
        obstacles = [
            SimulatorObstacle(
                type="static",
                shape=ObstacleShape(type="rectangle", width=1.0, length=1.0),
                position=StaticObstaclePosition(x=10.0, y=10.0, yaw=0.0),
            ),
            SimulatorObstacle(
                type="static",
                shape=ObstacleShape(type="circle", radius=0.5),
                position=StaticObstaclePosition(x=0.5, y=0.5, yaw=0.0),
            ),
        ]
        manager = ObstacleManager(obstacles)
        vehicle_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        # Should collide with second obstacle
        assert manager.check_vehicle_collision(vehicle_poly, 0.0) is True
