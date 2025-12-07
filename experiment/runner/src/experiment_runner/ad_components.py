"""Standard AD Component implementation adhering to Node Provider pattern."""

from typing import Any

from core.interfaces import ADComponent, Controller, Planner
from core.interfaces.node import Node
from core.nodes import ControlNode, PlanningNode, SensorNode


class StandardADComponent(ADComponent):
    """Standard AD Component with separate Planner and Controller nodes."""

    def __init__(self, vehicle_params: Any, **kwargs: Any) -> None:
        """Initialize standard component.

        Args:
            vehicle_params: Vehicle parameters
            **kwargs: Configuration parameters
                      Expects 'updates' dict for node rates.
                      Expects 'planning' dict for planner config.
                      Expects 'control' dict for control config.
                      Expects 'perception' dict for perception config (optional).
        """
        # Parse configuration
        self.config = kwargs
        self.updates_config = kwargs.get("updates", {})

        # Call super to create planner/controller via _create_* methods
        # But wait, ADComponent.__init__ calls _create_planner/_create_controller immediately.
        # We need to ensure we can parse what class to instantiate.
        # The Runner previously handled logic to split "type" and instantiate.
        # Now ADComponent is responsible for its own internals.

        # However, to be dynamic (load any planner/controller class),
        # this class needs to know WHICH planner/controller to load.
        # We can pass that in kwargs.

        super().__init__(vehicle_params, **kwargs)

        # Initialize nodes
        self.nodes: list[Node] = []
        self._setup_nodes()

    def _create_planner(self, **kwargs: Any) -> Planner:
        """Create planner instance."""
        plan_config = kwargs.get("planning", {})
        if not plan_config:
            raise ValueError("Missing 'planning' configuration in params")

        # Check if already instantiated (passed by runner)
        if isinstance(plan_config, Planner):
            return plan_config

        # Otherwise, instantiate from config dict
        cls_path = plan_config.get("type")
        params = plan_config.get("params", {})
        params["vehicle_params"] = self.vehicle_params
        return self._instantiate(cls_path, params)

    def _create_controller(self, **kwargs: Any) -> Controller:
        """Create controller instance."""
        ctrl_config = kwargs.get("control", {})
        if not ctrl_config:
            raise ValueError("Missing 'control' configuration in params")

        # Check if already instantiated (passed by runner)
        if isinstance(ctrl_config, Controller):
            return ctrl_config

        # Otherwise, instantiate from config dict
        cls_path = ctrl_config.get("type")
        params = ctrl_config.get("params", {})
        params["vehicle_params"] = self.vehicle_params
        return self._instantiate(cls_path, params)

    def _instantiate(self, class_path: str, params: dict[str, Any]) -> Any:
        import importlib

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # Resolve path params (simple version)
        # In actual runner we had get_project_root resolution.
        # Ideally we should keep that utility or pass resolved paths.
        # For this PoC, we assume resolved paths or use basic string paths.
        from core.utils import get_project_root

        path_keys = {"track_path", "model_path", "scaler_path"}
        resolved_params = {}
        for key, value in params.items():
            if key in path_keys and isinstance(value, str):
                resolved_params[key] = get_project_root() / value
            else:
                resolved_params[key] = value

        return cls(**resolved_params)

    def _setup_nodes(self) -> None:
        """Setup execution nodes."""
        # Get rates
        plan_hz = self.updates_config.get("planning_hz", 10.0)
        ctrl_hz = self.updates_config.get("control_hz", 30.0)
        sens_hz = self.updates_config.get(
            "perception_hz", 50.0
        )  # Default sensor rate if not specified

        # Create Nodes
        # Note: SensorNode is currently generic. In future `perception` config could specify a specific PerceptionNode.
        self.nodes = [
            SensorNode(rate_hz=sens_hz),
            PlanningNode(self.planner, rate_hz=plan_hz),
            ControlNode(self.controller, rate_hz=ctrl_hz),
        ]

    def get_schedulable_nodes(self) -> list[Node]:
        """Get list of schedulable nodes."""
        return self.nodes
