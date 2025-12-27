import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.data.dashboard import DashboardData
from core.utils import extract_dashboard_state, parse_mcap_message
from mcap.reader import make_reader

logger = logging.getLogger(__name__)


# Define mapping from MCAP topics (source) to DashboardData keys (target)
TOPIC_MAPPING = {
    "/localization/kinematic_state": "vehicle",
    "/control/command/control_cmd": "action",
    "/simulation/info": "metadata",
    "/obstacles": "obstacles",
}

# Dynamic topics (prefix based)
DYNAMIC_TOPIC_PREFIXES = ["/mppi_", "/pure_pursuit_", "/pid_"]

# Blacklisted topics that are too large for the dashboard overview
BLACKLISTED_TOPICS = {
    "/mppi_candidates",  # Very large visualization data
    "/lidar_scan",  # Raw sensor data
    "/perception/lidar/scan",
}


# Removed local _get_yaw_from_quat, using core.utils instead


def load_simulation_data(mcap_path: Path, vehicle_params: dict[str, Any] | Any) -> DashboardData:
    """Load simulation data from MCAP file into a structured, column-oriented format."""

    # Initialize containers
    timestamps: list[float] = []

    # Use defaultdict for generic columns mapping
    # structure: { "x": [], "y": [], ... } inside the function before packing into TypedDict
    vehicle_data: defaultdict[str, list[float]] = defaultdict(list)
    action_data: defaultdict[str, list[float]] = defaultdict(list)
    # ad_logs: simple list of dicts (row-oriented part for dynamic data)
    ad_logs_list: list[dict[str, Any]] = []

    metadata: dict[str, Any] = {}
    obstacles_data: list[dict[str, Any]] = []

    if not mcap_path.exists():
        logger.warning(f"MCAP file not found: {mcap_path}")
        return _empty_dashboard_data(vehicle_params)

    logger.info(f"Reading simulation data from MCAP: {mcap_path}")

    # Temporary storage for current time step
    current_timestamp = -1.0
    time_tolerance = 1e-4

    # State holders for the current grouping
    current_vehicle = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "velocity": 0.0}
    current_action = {"acceleration": 0.0, "steering": 0.0}
    current_ad_logs: dict[str, Any] = {}

    data_seen_in_step = False

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

            # Filter topics
            available_topics = set()
            if summary and summary.channels:
                for c in summary.channels.values():
                    # Skip blacklisted topics
                    if c.topic in BLACKLISTED_TOPICS:
                        continue
                    # Check mapping or dynamic prefix
                    if c.topic in TOPIC_MAPPING or any(
                        c.topic.startswith(p) for p in DYNAMIC_TOPIC_PREFIXES
                    ):
                        available_topics.add(c.topic)

            logger.info(f"Reading topics: {list(available_topics)}")

            for schema, channel, message in reader.iter_messages(topics=list(available_topics)):
                try:
                    json.loads(message.data)
                except Exception:
                    continue

                ts = message.log_time / 1e9
                topic = channel.topic
                is_ad_log = any(topic.startswith(prefix) for prefix in DYNAMIC_TOPIC_PREFIXES)

                # Check for new step
                if current_timestamp < 0:
                    current_timestamp = ts

                if abs(ts - current_timestamp) > time_tolerance:
                    # Commit step
                    if data_seen_in_step:
                        _append_step_columns(
                            timestamps,
                            vehicle_data,
                            action_data,
                            ad_logs_list,
                            current_timestamp,
                            current_vehicle,
                            current_action,
                            current_ad_logs,
                        )
                        # Reset ad logs for next step (events), but keep vehicle state (latch)
                        current_ad_logs = {}

                    current_timestamp = ts
                    data_seen_in_step = False

                # Update state
                data_seen_in_step = True

                # Parse using schema and model
                # Skip validation for ad_logs to speed up processing
                validate = not is_ad_log
                msg = parse_mcap_message(schema.name, message.data, validate=validate)
                extracted = extract_dashboard_state(msg)

                if schema.name in ["Odometry", "nav_msgs/Odometry"]:
                    current_vehicle.update(extracted)
                elif schema.name in [
                    "AckermannDriveStamped",
                    "ackermann_msgs/AckermannDriveStamped",
                ]:
                    current_action.update(extracted)
                elif topic == "/simulation/info" and isinstance(msg, dict):
                    metadata.update(msg)
                elif topic == "/obstacles" or schema.name in ["String", "std_msgs/String"]:
                    if "obstacles" in extracted and not obstacles_data:
                        obstacles_data = extracted["obstacles"]
                        logger.info(f"Loaded obstacles (count={len(obstacles_data)})")
                else:
                    # Generic AD logs
                    current_ad_logs[topic] = msg.model_dump() if hasattr(msg, "model_dump") else msg

            # Commit last step
            if data_seen_in_step:
                _append_step_columns(
                    timestamps,
                    vehicle_data,
                    action_data,
                    ad_logs_list,
                    current_timestamp,
                    current_vehicle,
                    current_action,
                    current_ad_logs,
                )

        logger.info(f"Loaded {len(timestamps)} steps from MCAP")

    except Exception as e:
        logger.error(f"Failed to read MCAP file: {e}")

    return {
        "timestamps": timestamps,
        "vehicle": dict(vehicle_data),
        "action": dict(action_data),
        "obstacles": obstacles_data,
        "metadata": metadata,
        "ad_logs": ad_logs_list,
        "vehicle_params": vehicle_params,
    }


def _append_step_columns(
    timestamps, vehicle_data, action_data, ad_logs_list, ts, vehicle, action, ad_logs
):
    timestamps.append(ts)

    for k, v in vehicle.items():
        vehicle_data[k].append(v)

    for k, v in action.items():
        action_data[k].append(v)

    # Use valid dict, clean keys if needed
    cleaned_ad_logs = {"data": {k.strip("/"): v for k, v in ad_logs.items()}} if ad_logs else None
    ad_logs_list.append(cleaned_ad_logs)


def _empty_dashboard_data(vehicle_params):
    return {
        "timestamps": [],
        "vehicle": {},
        "action": {},
        "obstacles": [],
        "metadata": {},
        "ad_logs": [],
        "vehicle_params": vehicle_params,
    }
