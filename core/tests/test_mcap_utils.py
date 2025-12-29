"""Tests for dynamic MCAP data extraction utilities."""

import json

from core.utils.mcap_utils import (
    extract_dashboard_state,
    parse_mcap_message,
)


def test_parse_mcap_message_odometry():
    # Odometry is in core.data.ros
    payload = {
        "header": {"stamp": {"sec": 0, "nanosec": 0, "nsec": 0}, "frame_id": "map"},
        "pose": {
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        },
        "twist": {"twist": {"linear": {"x": 5.0}}},
    }
    data = json.dumps(payload).encode()
    msg = parse_mcap_message("Odometry", data)

    # Should be parsed as Pydantic model
    assert hasattr(msg, "pose")
    assert msg.pose.pose.position.x == 1.0


def test_extract_dashboard_state_odometry():
    payload = {
        "header": {"stamp": {"sec": 0, "nanosec": 0, "nsec": 0}, "frame_id": "map"},
        "pose": {
            "pose": {
                "position": {"x": 10.0, "y": 20.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        },
        "twist": {"twist": {"linear": {"x": 3.0}}},
    }
    msg = parse_mcap_message("nav_msgs/Odometry", json.dumps(payload).encode())
    state = extract_dashboard_state(msg)

    assert state.get("x") == 10.0
    assert state.get("y") == 20.0
    assert state.get("yaw") == 0.0
    assert state.get("velocity") == 3.0


def test_extract_dashboard_state_ackermann():
    payload = {
        "lateral": {"steering_tire_angle": -0.5},
        "longitudinal": {"acceleration": 2.0},
    }
    msg = parse_mcap_message("AckermannControlCommand", json.dumps(payload).encode())
    state = extract_dashboard_state(msg)

    assert state.get("acceleration") == 2.0
    assert state.get("steering") == -0.5


def test_extract_dashboard_state_obstacles():
    obs_list = [{"id": "obs1", "x": 1.0}]
    # Logged as std_msgs/String with JSON data
    payload = {"data": json.dumps(obs_list)}
    msg = parse_mcap_message("String", json.dumps(payload).encode())
    state = extract_dashboard_state(msg)

    assert "obstacles" in state
    assert state["obstacles"] == obs_list


def test_extract_dashboard_state_flat_dict():
    # Direct dictionary extraction (duck typing)
    payload = {"x": 5.0, "y": 6.0, "acceleration": 1.1}
    state = extract_dashboard_state(payload)

    assert state["x"] == 5.0
    assert state["y"] == 6.0
    assert state["acceleration"] == 1.1
