import importlib
import inspect
from typing import Any

from core.data import VehicleParameters
from core.interfaces.node import Node
from core.utils.paths import get_project_root


def create_node(
    node_type: str,
    name: str,
    rate_hz: float,
    params: dict[str, Any],
    vehicle_params: VehicleParameters,
) -> Node:
    """Create a Node instance dynamically.

    Args:
        node_type: Class path (e.g., "package.module.ClassName")
        name: Node instance name
        rate_hz: Execution frequency in Hz
        params: Node configuration parameters
        vehicle_params: Vehicle parameters to inject if required by Node

    Returns:
        Instantiated Node
    """
    # Resolve path parameters
    path_keys = {"track_path", "model_path", "scaler_path"}
    workspace_root = get_project_root()

    # Copy params to avoid mutation
    node_params = params.copy()

    for key, value in node_params.items():
        if key in path_keys and isinstance(value, str):
            # Resolve relative paths against workspace root
            # Only if it looks like a relative path? get_project_root / value handles both.
            node_params[key] = str(workspace_root / value)

    # Import class
    try:
        module_name, class_name = node_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ValueError(
            f"Invalid node type: {node_type}. "
            f"Must be in 'module.ClassName' format and importable. Error: {e}"
        ) from e

    if not issubclass(cls, Node):
        raise TypeError(f"Class {cls} is not a subclass of Node")

    # Inject vehicle_params if needed
    sig = inspect.signature(cls.__init__)

    # Check signature for vehicle_params
    if "vehicle_params" in sig.parameters:
        return cls(config=node_params, rate_hz=rate_hz, vehicle_params=vehicle_params)
    else:
        # Standard Node signature (name, rate_hz, config) ??
        # Or (config, rate_hz) depending on implementation conventions.
        # Based on refactor work, most new nodes follow (config, rate_hz, [vehicle_params]).
        # But base Node is (name, rate_hz, config).
        # We assume the Node implementation handles its own naming or construction details.

        # If the Node does NOT take vehicle_params, we try standard instantiation.
        # Checking against PurePursuitNode/PIDNode signatures we made:
        # def __init__(self, config: dict, rate_hz: float, vehicle_params: ...)

        # If we encounter a Node that follows the BASE Node signature:
        # def __init__(self, name: str, rate_hz: float, config: Any = None)

        # Let's check parameters names to be safe.
        kwargs = {}
        if "config" in sig.parameters:
            kwargs["config"] = node_params

        if "name" in sig.parameters:
            kwargs["name"] = name

        if "rate_hz" in sig.parameters:
            kwargs["rate_hz"] = rate_hz

        return cls(**kwargs)
