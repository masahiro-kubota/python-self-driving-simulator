import numpy as np
import pytest
from lateral_shift_planner.obstacle_manager import TargetObstacle
from lateral_shift_planner.shift_profile import ShiftProfile, merge_profiles


@pytest.fixture
def left_obs():
    # Obstacle at s=10, l=1.0 (Left). Width=1, Length=2.
    # To force Right avoidance, space_to_left < space_to_right
    return TargetObstacle(
        id="1",
        s=10.0,
        lat=1.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=1.0,
        right_boundary_dist=3.0,
    )


@pytest.fixture
def right_obs():
    # Obstacle at s=10, l=-1.0 (Right). Width=1, Length=2.
    # To force Left avoidance, space_to_left >= space_to_right
    return TargetObstacle(
        id="2",
        s=10.0,
        lat=-1.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=3.0,
        right_boundary_dist=1.0,
    )


def test_direction(left_obs, right_obs):
    # Left obs -> Avoid Right (negative l)
    p_left = ShiftProfile(left_obs, vehicle_width=2.0)
    assert p_left.sign < 0
    assert p_left.target_lat < left_obs.lat  # Should be smaller (more negative)

    # Right obs -> Avoid Left (positive l)
    p_right = ShiftProfile(right_obs, vehicle_width=2.0)
    assert p_right.sign > 0
    assert p_right.target_lat > right_obs.lat


def test_profile_shape(left_obs):
    # Obstacle s=10, length=2 -> half_length=1
    # d_front=2, d_rear=2, avoid_dist=10
    # s_start: 10 - 1 - 2 - 10 = -3
    # s_full:  10 - 1 - 2 = 7
    # s_keep:  10 + 1 + 2 = 13
    # s_end:   10 + 1 + 2 + 10 = 23

    p = ShiftProfile(
        left_obs,
        vehicle_width=2.0,
        avoidance_maneuver_length=10.0,
        longitudinal_margin_front=2.0,
        longitudinal_margin_rear=2.0,
    )

    # Before start (-3.0)
    assert p.get_lat(-5.0) == 0.0

    # At start
    assert p.get_lat(-3.0) == pytest.approx(0.0, abs=1e-3)

    # Mid ramp (between -3 and 7) -> midpoint = 2
    mid = (-3.0 + 7.0) / 2.0
    val = p.get_lat(mid)
    assert abs(val) > 0
    assert abs(val) < abs(p.target_lat)

    # Full avoid (at 7.0 + epsilon)
    assert p.get_lat(8.0) == pytest.approx(p.target_lat)

    # End (at 23.0)
    assert p.get_lat(24.0) == 0.0


def test_merge_same_side():
    # Two obstacles on left.
    # Obs1: l=1. Target l ~ -1
    # Obs2: l=2. Target l ~ 0 (Wait, if obs is further left, we need less shift to right? No.)
    # If Obs1 at l=1. Safe right boundary < 1 - Clear.
    # If Obs2 at l=2. Safe right boundary < 2 - Clear.
    # So Obs1 is more restrictive (closer to center).

    obs1 = TargetObstacle(
        id="1",
        s=10.0,
        lat=1.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=1.0,
        right_boundary_dist=3.0,
    )
    obs2 = TargetObstacle(
        id="2",
        s=10.0,
        lat=2.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=0.5,
        right_boundary_dist=3.5,
    )

    p1 = ShiftProfile(obs1, vehicle_width=2.0)  # Avoid Right
    p2 = ShiftProfile(obs2, vehicle_width=2.0)  # Avoid Right

    # p1 target_l will be approx 1 - (0.5 + 1.0 + 0.5) = -1.0
    # p2 target_l will be approx 2 - 2.0 = 0.0
    # So p1 requires l < -1. p2 requires l < 0.
    # Combined requires l < -1. So max shift.

    # Check manual calc
    # Clear = 0.5 + 1.0 + 0.5 = 2.0
    # p1 target = 1.0 - 2.0 = -1.0
    # p2 target = 2.0 - 2.0 = 0.0

    s = np.array([10.0])
    l_tgt, coll = merge_profiles(s, [p1, p2])

    assert not coll
    assert l_tgt[0] == pytest.approx(-1.0)  # More negative one


def test_merge_slalom():
    # Obs1 Left (l=2) -> Req l < 0
    # Obs2 Right (l=-2) -> Req l > 0

    obs1 = TargetObstacle(
        id="1",
        s=10.0,
        lat=2.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=1.0,
        right_boundary_dist=3.0,
    )
    obs2 = TargetObstacle(
        id="2",
        s=10.0,
        lat=-2.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=3.0,
        right_boundary_dist=1.0,
    )

    p1 = ShiftProfile(obs1, vehicle_width=2.0)  # Target l = 0 (Left req < 0)
    p2 = ShiftProfile(obs2, vehicle_width=2.0)  # Target l = 0 (Right req > 0)

    # Both active at s=10
    s = np.array([10.0])
    l_tgt, coll = merge_profiles(s, [p1, p2])

    assert not coll
    assert l_tgt[0] == pytest.approx(0.0)


def test_merge_collision():
    # Obs1 Left (l=1) -> Req l < -1
    # Obs2 Right (l=-1) -> Req l > 1
    # Impossible interval: 1 < l < -1 -> Empty set.

    obs1 = TargetObstacle(
        id="1",
        s=10.0,
        lat=1.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=1.0,
        right_boundary_dist=3.0,
    )
    obs2 = TargetObstacle(
        id="2",
        s=10.0,
        lat=-1.0,
        length=2.0,
        width=1.0,
        left_boundary_dist=3.0,
        right_boundary_dist=1.0,
    )

    p1 = ShiftProfile(obs1, vehicle_width=2.0)
    p2 = ShiftProfile(obs2, vehicle_width=2.0)

    s = np.array([10.0])
    _, coll = merge_profiles(s, [p1, p2])

    assert coll
