import numpy as np
import pytest
from planning_utils.types import ReferencePath, ReferencePathPoint
from static_avoidance_planner.frenet_converter import FrenetConverter


def create_straight_path(length=100.0, step=1.0):
    points = []
    for x in np.arange(0, length + step, step):
        points.append(ReferencePathPoint(x=x, y=0.0, yaw=0.0, velocity=10.0))
    return ReferencePath(points=points)


def create_curved_path(radius=50.0, angle_deg=90):
    points = []
    angle_rad = np.deg2rad(angle_deg)
    step_rad = np.deg2rad(1.0)
    for theta in np.arange(0, angle_rad + step_rad, step_rad):
        x = radius * np.sin(theta)
        y = radius * (1 - np.cos(theta))
        # Yaw is tangent: dx = r cos theta, dy = r sin theta. tan = dy/dx = tan theta
        # But wait.
        # x = r sin t, y = r - r cos t
        # dx/dt = r cos t
        # dy/dt = r sin t
        # yaw = t
        points.append(ReferencePathPoint(x=x, y=y, yaw=theta, velocity=10.0))
    return ReferencePath(points=points)


def test_straight_path_projection():
    path = create_straight_path()
    converter = FrenetConverter(path)

    # On path
    s, lat = converter.global_to_frenet(10.0, 0.0)
    assert s == pytest.approx(10.0, abs=1e-3)
    assert lat == pytest.approx(0.0, abs=1e-3)

    # Left of path
    s, lat = converter.global_to_frenet(10.0, 2.0)
    assert s == pytest.approx(10.0, abs=1e-3)
    assert lat == pytest.approx(2.0, abs=1e-3)

    # Right of path
    s, lat = converter.global_to_frenet(20.5, -1.5)
    assert s == pytest.approx(20.5, abs=1e-3)
    assert lat == pytest.approx(-1.5, abs=1e-3)


def test_frenet_to_global_straight():
    path = create_straight_path()
    converter = FrenetConverter(path)

    x, y = converter.frenet_to_global(10.0, 2.0)
    assert x == pytest.approx(10.0, abs=1e-3)
    assert y == pytest.approx(2.0, abs=1e-3)


def test_round_trip():
    path = create_curved_path()
    converter = FrenetConverter(path)

    test_points = [(10.0, 5.0), (20.0, -3.0), (5.0, 0.0)]

    for tx, ty in test_points:
        s, lat = converter.global_to_frenet(tx, ty)
        rx, ry = converter.frenet_to_global(s, lat)

        assert rx == pytest.approx(tx, abs=5e-2)
        assert ry == pytest.approx(ty, abs=5e-2)


def test_extrapolation():
    path = create_straight_path(length=10.0)  # s: 0 to 10
    converter = FrenetConverter(path)

    # Before start
    s, _ = converter.global_to_frenet(-2.0, 1.0)
    # Projection should be at s=0
    assert s == pytest.approx(-2.0, abs=1.0)  # It might project to s<0 if linear, or s=0 if clamped
    # My implementation uses vector projection on first segment, so s can be negative.

    # After end
    s, _ = converter.global_to_frenet(12.0, 0.0)
    # Should project to s=12.0 on the last tangent
    assert s == pytest.approx(12.0, abs=0.5)
