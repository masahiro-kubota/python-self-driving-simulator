import pytest
from core.data import VehicleState
from core.data.environment.obstacle import Obstacle, ObstacleType
from planning_utils.types import ReferencePath, ReferencePathPoint
from static_avoidance_planner.avoidance_planner import AvoidancePlanner, AvoidancePlannerConfig


def create_straight_trijectory(length=100.0):
    points = []
    for x in range(int(length) + 1):
        points.append(ReferencePathPoint(x=float(x), y=0.0, yaw=0.0, velocity=10.0))
    return ReferencePath(points=points)


@pytest.fixture
def planner():
    ref = create_straight_trijectory()
    config = AvoidancePlannerConfig(
        lookahead_distance=20.0, trajectory_resolution=1.0, vehicle_width=2.0
    )
    return AvoidancePlanner(ref, config)


def test_plan_no_obstacles(planner):
    ego = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=10.0)
    traj = planner.plan(ego, [])

    # Should follow centerline (y=0)
    assert len(traj.points) > 0
    for p in traj.points:
        assert p.y == pytest.approx(0.0, abs=1e-3)
        assert p.x >= 0.0


def test_plan_avoidance(planner):
    ego = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=10.0)

    # Obstacle at x=15, y=0 (Center). Width=1.
    # Ego width=2.
    # Should avoid. Obstacle at l=0 -> Shift Left (sign=1)
    # Required l = 0 + 1 * (0.5+1+0.5) = 2.0 (approx, margin=0.5)

    obs = Obstacle(id="1", type=ObstacleType.STATIC, x=15.0, y=0.0, width=1.0, height=2.0)

    traj = planner.plan(ego, [obs])

    # Check at s=15 (Obstacle location)
    # trajectory points are roughly s samples.
    # Find point closest to x=15

    ys = [p.y for p in traj.points]
    max_y = max(ys)

    assert max_y > 1.0  # Should have shifted left significantly

    # Check if any point is collision?
    # Not strictly collision checking here, just checking generation.


def test_trajectory_properties(planner):
    ego = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=10.0)
    traj = planner.plan(ego, [])

    # Check yaw calculation (Straight line -> yaw=0)
    for p in traj.points:
        assert p.yaw == pytest.approx(0.0, abs=1e-3)
