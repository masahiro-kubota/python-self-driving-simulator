"""MCAP data extraction utilities with dynamic model discovery and structural extraction."""

import importlib
import json
import logging
import math
import pkgutil
from typing import Any

from pathlib import Path
from typing import Any, Generator, Tuple, List, Optional
from types import SimpleNamespace

from pydantic import BaseModel
from rosbags.highlevel import AnyReader, AnyReaderError
from rosbags.rosbag2.errors import ReaderError
from mcap.reader import make_reader

import core.data

logger = logging.getLogger(__name__)

# Global cache for schema-to-model mapping
_SCHEMA_TO_MODEL_CACHE: dict[str, type[BaseModel]] = {}


def msg_to_dict(obj: Any) -> Any:
    """Convert rosbags message object to dictionary."""
    if hasattr(obj, '__dict__'):
        return {k: msg_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, list):
        return [msg_to_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        if hasattr(obj, '_fields'):
            return {k: msg_to_dict(getattr(obj, k)) for k in obj._fields}
        return [msg_to_dict(x) for x in obj]
    else:
        return obj


def dict_to_namespace(d: Any) -> Any:
    """Convert dictionary to SimpleNamespace recursively."""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def read_messages(
    mcap_path: str, topics: List[str], as_namespace: bool = True
) -> Generator[Tuple[str, Any, int], None, None]:
    """
    Reads messages from an MCAP file, handling both ROS2 (CDR) and JSON encoding.

    Args:
        mcap_path: Path to the MCAP file.
        topics: List of topics to extract.
        as_namespace: If True, converts dicts/JSON to SimpleNamespace for dot notation.

    Yields:
        (topic, message_object, timestamp_ns)
    """
    path = Path(mcap_path)
    if not path.exists():
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    # Try AnyReader (ROS2 focus)
    use_generic = False
    try:
        with AnyReader([path]) as reader:
            # Filter connections
            connections = [x for x in reader.connections if x.topic in topics]
            
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                try:
                    if connection.digest == 'json' or 'json' in connection.encoding:
                        msg = json.loads(rawdata)
                        if as_namespace:
                            msg = dict_to_namespace(msg)
                    else:
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        # rosbags usually returns objects with dot access already
                        
                    yield connection.topic, msg, timestamp
                except Exception as e:
                    # Log error but continue
                    pass
                    
    except (ReaderError, AnyReaderError):
        use_generic = True
    except Exception:
        use_generic = True
        
    # Fallback to generic MCAP reader
    if use_generic:
        with open(path, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=topics):
                try:
                    if schema.encoding in ['json', 'jsonschema']:
                        msg = json.loads(message.data)
                        if as_namespace:
                            msg = dict_to_namespace(msg)
                    else:
                        # Skip unsupported encodings in generic mode
                        continue
                        
                    yield channel.topic, msg, message.log_time
                except Exception:
                    pass


def _discover_models():
    """Dynamically discover Pydantic models in core.data and its subpackages."""
    if _SCHEMA_TO_MODEL_CACHE:
        return

    # Scan core.data and subpackages
    prefix = core.data.__name__ + "."
    for loader, module_name, is_pkg in pkgutil.walk_packages(core.data.__path__, prefix):
        try:
            module = importlib.import_module(module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                    # Register by class name
                    _SCHEMA_TO_MODEL_CACHE[attr.__name__] = attr
                    # Also register with ROS-style prefix if available (placeholder logic)
                    # For now, we mainly rely on class names matching schema names
        except Exception as e:
            logger.warning(f"Failed to import module {module_name} for model discovery: {e}")

    # Add common ROS aliases if not already present
    aliases = {
        "nav_msgs/Odometry": "Odometry",
        "visualization_msgs/MarkerArray": "MarkerArray",
        "std_msgs/String": "String",
        "tf2_msgs/TFMessage": "TFMessage",
    }
    for alias, target in aliases.items():
        if target in _SCHEMA_TO_MODEL_CACHE:
            _SCHEMA_TO_MODEL_CACHE[alias] = _SCHEMA_TO_MODEL_CACHE[target]


def parse_mcap_message(
    schema_name: str, data: bytes, validate: bool = True
) -> BaseModel | dict[str, Any]:
    """Parse MCAP message data using dynamically discovered schemas.

    Args:
        schema_name: Schema name from MCAP channel/schema.
        data: Raw byte data (usually JSON) from MCAP message.
        validate: Whether to validate using Pydantic models.

    Returns:
        Pydantic model instance if schema is matched and validate=True, otherwise raw dict.
    """
    _discover_models()

    try:
        payload = json.loads(data)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to decode JSON from MCAP message: {e}")
        return {}

    model_cls = _SCHEMA_TO_MODEL_CACHE.get(schema_name)
    if model_cls and validate:
        try:
            return model_cls.model_validate(payload)
        except Exception as e:
            # Fallback to dict if validation fails (e.g. schema slightly different)
            logger.debug(f"Pydantic validation failed for {schema_name}: {e}")
            return payload

    return payload


def get_recursive_attr(obj: Any, path: str, default: Any = None) -> Any:
    """Safely get nested attributes or dictionary keys.

    Args:
        obj: Object or dictionary.
        path: Dot-separated path (e.g., 'pose.pose.position.x').
        default: Default value if not found.
    """
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)

        if current is None:
            return default
    return current


def extract_dashboard_state(msg: Any) -> dict[str, Any]:
    """Extract flat dashboard-friendly state using structural 'duck-typing'.

    Focuses on 'finding' position, velocity, and drive commands regardless of exact type.

    Args:
        msg: Pydantic model instance or dictionary.

    Returns:
        Dictionary containing extracted fields ('x', 'y', 'yaw', 'velocity', etc.).
    """
    result = {}

    # 1. Try to extract Position/Orientation (Odometry-like or Pose-like)
    # Check for ROS Odometry structure
    x = get_recursive_attr(msg, "pose.pose.position.x")
    y = get_recursive_attr(msg, "pose.pose.position.y")
    if x is not None and y is not None:
        result["x"] = x
        result["y"] = y
        result["z"] = get_recursive_attr(msg, "pose.pose.position.z", 0.0)

        # Orientation for Yaw
        ow = get_recursive_attr(msg, "pose.pose.orientation.w")
        ox = get_recursive_attr(msg, "pose.pose.orientation.x")
        oy = get_recursive_attr(msg, "pose.pose.orientation.y")
        oz = get_recursive_attr(msg, "pose.pose.orientation.z")
        if all(v is not None for v in [ow, ox, oy, oz]):
            result["yaw"] = math.atan2(2 * (ow * oz + ox * oy), 1 - 2 * (oy * oy + oz * oz))

    # 2. Try to extract Velocity
    v = get_recursive_attr(msg, "twist.twist.linear.x")
    if v is not None:
        result["velocity"] = v
    elif "velocity" in result:  # already set?
        pass

    # 3. Try to extract Control Commands (Autoware AckermannControlCommand)
    accel = get_recursive_attr(msg, "longitudinal.acceleration")
    steer = get_recursive_attr(msg, "lateral.steering_tire_angle")

    if accel is not None:
        result["acceleration"] = accel
    if steer is not None:
        result["steering"] = steer

    # 4. Fallback for flat dictionary (Backward compatibility or simple logs)
    if isinstance(msg, dict):
        for key in ["x", "y", "z", "yaw", "velocity", "acceleration", "steering"]:
            if key not in result and key in msg:
                result[key] = msg[key]

        # Obstacles special handling (String data)
        if "data" in msg and isinstance(msg["data"], str):
            try:
                data = json.loads(msg["data"])
                if isinstance(data, list):
                    result["obstacles"] = data
            except Exception:
                pass

    # 5. Pydantic specific fallback for String message
    if hasattr(msg, "data") and isinstance(msg.data, str):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                result["obstacles"] = data
        except Exception:
            pass

    return result
